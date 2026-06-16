use flybrain_core::verify::VerificationResult;

/// Running budget counters for one MAS execution.
#[derive(Debug, Clone, Default)]
pub struct BudgetState {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub llm_calls: u32,
    pub cost_rub: f32,
    pub latency_ms: u64,
}

impl BudgetState {
    pub fn add_call(&mut self, tokens_in: u64, tokens_out: u64, cost_rub: f32, latency_ms: u64) {
        self.tokens_in += tokens_in;
        self.tokens_out += tokens_out;
        self.llm_calls += 1;
        self.cost_rub += cost_rub;
        self.latency_ms += latency_ms;
    }
}

/// Hard/soft budget limits. `hard_*` failing causes the runtime to terminate;
/// `soft_*` failing produces a warning and a negative reward delta.
#[derive(Debug, Clone)]
pub struct BudgetVerifier {
    pub hard_cost_rub: f32,
    pub hard_tokens: u64,
    pub hard_calls: u32,
    pub soft_cost_rub: f32,
}

impl BudgetVerifier {
    pub fn from_hard_cap_rub(hard_cap_rub: f32) -> Self {
        Self {
            hard_cost_rub: hard_cap_rub,
            hard_tokens: u64::MAX,
            hard_calls: u32::MAX,
            soft_cost_rub: hard_cap_rub * 0.8,
        }
    }

    pub fn check(&self, state: &BudgetState) -> VerificationResult {
        if state.cost_rub > self.hard_cost_rub {
            return VerificationResult {
                passed: false,
                score: 0.0,
                errors: vec![format!(
                    "hard cost cap exceeded: {:.2}/{:.2} ₽",
                    state.cost_rub, self.hard_cost_rub
                )],
                warnings: vec![],
                failed_component: Some("budget".into()),
                suggested_next_agent: None,
                reward_delta: -1.0,
            };
        }
        if state.tokens_in + state.tokens_out > self.hard_tokens {
            return VerificationResult {
                passed: false,
                score: 0.0,
                errors: vec![format!(
                    "token cap exceeded: {}/{}",
                    state.tokens_in + state.tokens_out,
                    self.hard_tokens
                )],
                warnings: vec![],
                failed_component: Some("budget".into()),
                suggested_next_agent: None,
                reward_delta: -1.0,
            };
        }
        if state.llm_calls > self.hard_calls {
            return VerificationResult::fail(format!(
                "call cap exceeded: {}/{}",
                state.llm_calls, self.hard_calls
            ));
        }

        let mut warnings = Vec::new();
        if state.cost_rub > self.soft_cost_rub {
            warnings.push(format!(
                "soft cost threshold reached: {:.2} ₽ > {:.2} ₽",
                state.cost_rub, self.soft_cost_rub
            ));
        }

        VerificationResult {
            passed: true,
            score: (1.0 - state.cost_rub / self.hard_cost_rub).clamp(0.0, 1.0),
            errors: vec![],
            warnings,
            failed_component: None,
            suggested_next_agent: None,
            reward_delta: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_under_cap() {
        let v = BudgetVerifier::from_hard_cap_rub(2000.0);
        let mut s = BudgetState::default();
        s.add_call(100, 50, 5.0, 10);
        assert!(v.check(&s).passed);
    }

    #[test]
    fn fails_over_hard_cap() {
        let v = BudgetVerifier::from_hard_cap_rub(10.0);
        let mut s = BudgetState::default();
        s.add_call(1000, 500, 50.0, 100);
        let r = v.check(&s);
        assert!(!r.passed);
        assert_eq!(r.failed_component.as_deref(), Some("budget"));
    }

    #[test]
    fn warns_over_soft_cap() {
        let v = BudgetVerifier::from_hard_cap_rub(100.0);
        let mut s = BudgetState::default();
        s.add_call(1000, 500, 85.0, 100);
        let r = v.check(&s);
        assert!(r.passed);
        assert!(!r.warnings.is_empty());
    }
}
