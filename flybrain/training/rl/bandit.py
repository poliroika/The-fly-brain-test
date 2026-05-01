"""Contextual bandits over the discrete (kind, agent) action space
(README §12.3 — "для тестового достаточно contextual bandit").

These are not policy-gradient methods — they're closed-form / sampled
posterior solvers that consume the controller's state-vector as
context and output a flat action index. A small wrapper at the call
site maps that index back to a ``(kind, agent)`` action dict.

Two algorithms ship:

* :class:`LinUCBBandit`   — classic linear UCB with a Sherman-Morrison
                             rank-1 update for ``A^{-1}``.
* :class:`ThompsonBandit` — Bayesian linear regression with a
                             Gaussian-Wishart-ish prior; samples a
                             coefficient vector each step (Thompson
                             sampling).

Both algorithms are pure-numpy so they don't pull torch into
inference paths. They're picklable and small (< few hundred KB of
state per arm even for high-dim contexts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _flatten_context(ctx: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(ctx, dtype=np.float64).ravel()
    if arr.ndim != 1:
        raise ValueError(f"context must flatten to 1-D, got shape {arr.shape}")
    return arr


@dataclass
class LinUCBBandit:
    """Linear UCB (Li et al., 2010) over ``num_arms`` discrete actions.

    Each arm holds an independent ridge-regression model with
    parameter ``A_a, b_a``::

        theta_a = A_a^{-1} @ b_a
        UCB_a   = theta_a^T x  +  alpha * sqrt(x^T A_a^{-1} x)

    The arm with the highest UCB is selected. After observing the
    reward, the chosen arm's ``A`` and ``b`` are updated.
    """

    num_arms: int
    context_dim: int
    alpha: float = 1.0
    """Exploration coefficient. Larger = more exploration."""
    ridge: float = 1.0
    """Ridge parameter for the prior ``A_0 = ridge * I``."""
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    A_inv: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        eye = np.eye(self.context_dim, dtype=np.float64) / self.ridge
        self.A_inv = np.tile(eye[None, :, :], (self.num_arms, 1, 1))
        self.b = np.zeros((self.num_arms, self.context_dim), dtype=np.float64)

    def select(
        self,
        context: np.ndarray | list[float],
        action_mask: np.ndarray | list[bool] | None = None,
    ) -> int:
        x = _flatten_context(context)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"expected context of dim {self.context_dim}, got {x.shape[0]}")
        # theta_a = A_a^{-1} b_a
        thetas = np.einsum("aij,aj->ai", self.A_inv, self.b)
        means = thetas @ x  # (num_arms,)
        # variance term: x^T A_a^{-1} x
        covs = np.einsum("i,aij,j->a", x, self.A_inv, x)
        ucb = means + self.alpha * np.sqrt(np.clip(covs, 0.0, None))

        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape[0] != self.num_arms:
                raise ValueError("action_mask shape mismatch")
            ucb = np.where(mask, ucb, -np.inf)
            if not np.any(mask):
                return int(self.rng.integers(self.num_arms))

        # Tie-break uniformly at random among argmax to avoid bias.
        best = np.flatnonzero(ucb == ucb.max())
        return int(self.rng.choice(best))

    def update(
        self,
        arm: int,
        context: np.ndarray | list[float],
        reward: float,
    ) -> None:
        x = _flatten_context(context)
        # Sherman-Morrison rank-1 update: A_inv -= (A_inv x)(x^T A_inv) / (1 + x^T A_inv x)
        Ax = self.A_inv[arm] @ x
        denom = 1.0 + float(x @ Ax)
        if denom <= 0.0:
            # fall back to a small ridge bump
            self.A_inv[arm] = np.linalg.inv(np.linalg.inv(self.A_inv[arm]) + np.outer(x, x))
        else:
            self.A_inv[arm] -= np.outer(Ax, Ax) / denom
        self.b[arm] += reward * x


@dataclass
class ThompsonBandit:
    """Bayesian linear-regression Thompson sampling over ``num_arms``.

    Each arm has posterior ``N(mu_a, sigma^2 A_a^{-1})`` with
    ``A_a = ridge*I + sum x x^T`` and
    ``b_a = sum r * x``. We sample
    ``theta ~ N(A_a^{-1} b_a, sigma^2 A_a^{-1})`` and pick the
    argmax over ``theta_a^T x``.
    """

    num_arms: int
    context_dim: int
    sigma: float = 0.5
    ridge: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    A: np.ndarray = field(init=False)
    A_inv: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        eye = np.eye(self.context_dim, dtype=np.float64) * self.ridge
        self.A = np.tile(eye[None, :, :], (self.num_arms, 1, 1))
        eye_inv = np.eye(self.context_dim, dtype=np.float64) / self.ridge
        self.A_inv = np.tile(eye_inv[None, :, :], (self.num_arms, 1, 1))
        self.b = np.zeros((self.num_arms, self.context_dim), dtype=np.float64)

    def select(
        self,
        context: np.ndarray | list[float],
        action_mask: np.ndarray | list[bool] | None = None,
    ) -> int:
        x = _flatten_context(context)
        if x.shape[0] != self.context_dim:
            raise ValueError(f"expected context of dim {self.context_dim}, got {x.shape[0]}")
        # mu = A_inv @ b
        mu = np.einsum("aij,aj->ai", self.A_inv, self.b)
        # Sample theta_a ~ N(mu_a, sigma^2 A_inv_a)
        scores = np.zeros(self.num_arms, dtype=np.float64)
        for a in range(self.num_arms):
            cov = (self.sigma**2) * self.A_inv[a]
            try:
                theta = self.rng.multivariate_normal(mu[a], cov, check_valid="ignore")
            except (np.linalg.LinAlgError, ValueError):
                # If cov isn't PSD (numerical drift), fall back to mean.
                theta = mu[a]
            scores[a] = float(theta @ x)

        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape[0] != self.num_arms:
                raise ValueError("action_mask shape mismatch")
            scores = np.where(mask, scores, -np.inf)
            if not np.any(mask):
                return int(self.rng.integers(self.num_arms))

        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))

    def update(
        self,
        arm: int,
        context: np.ndarray | list[float],
        reward: float,
    ) -> None:
        x = _flatten_context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        # Sherman-Morrison rank-1 inverse update (same as LinUCB).
        Ax = self.A_inv[arm] @ x
        denom = 1.0 + float(x @ Ax)
        if denom <= 0.0:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        else:
            self.A_inv[arm] -= np.outer(Ax, Ax) / denom


__all__ = ["LinUCBBandit", "ThompsonBandit"]
