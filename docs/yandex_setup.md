# Yandex AI Studio setup

This document is the operator runbook for getting FlyBrain Optimizer
talking to Yandex AI Studio.

## What you need

1. A Yandex Cloud account with a folder.
2. A service account with the `ai.languageModels.user` and
   `ai.embeddings.user` roles.
3. An API key for that service account.

## Provisioning (manual, no Terraform)

```bash
# 1. Create the service account.
yc iam service-account create --name flybrain-sa

# 2. Grant LLM + embedding roles.
SA_ID=$(yc iam service-account get --name flybrain-sa --format json | jq -r .id)
FOLDER_ID=$(yc config get folder-id)

for ROLE in ai.languageModels.user ai.embeddings.user; do
  yc resource-manager folder add-access-binding "$FOLDER_ID" \
    --role "$ROLE" \
    --service-account-id "$SA_ID"
done

# 3. Create an API key.
yc iam api-key create --service-account-id "$SA_ID" --format json \
  | jq -r .secret > .api_key.local

# 4. Store env vars.
export YANDEX_FOLDER_ID="$FOLDER_ID"
export YANDEX_API_KEY="$(cat .api_key.local)"
```

## Provisioning (Terraform)

See `infra/terraform/README.md`. Phase 0 ships the skeleton; you opt in to
`apply` before Phase 7 or in Phase 12.

## Configuration

`configs/llm/yandex.yaml` reads `YANDEX_FOLDER_ID` (or lowercase `folder_id`)
and `YANDEX_API_KEY` from env vars. The agent → tier mapping (`lite` /
`pro`) is also there.

## Smoke test

```bash
export FLYBRAIN_RUN_LIVE_LLM=1
. .venv/bin/activate
pytest tests/python/integration/test_yandex_live.py -q
```

A single run costs about 0.5 ₽ (lite, ~10 input + ~3 output tokens).

## Pricing assumptions

The defaults in `flybrain/llm/pricing.py` use:

* `yandexgpt-lite/latest`: 0.40 ₽ / 1k tokens
* `yandexgpt/latest` (Pro): 1.20 ₽ / 1k tokens
* `text-search-{doc,query}/latest`: 0.10 ₽ / 1k tokens

These are upper-bound estimates. Override per environment in
`configs/llm/yandex.yaml` if Yandex updates pricing.

## Budget

Total session budget locked at 2000 ₽ (split 200 / 900 / 900 across
dev / train / eval). The `BudgetController` (Rust) and `BudgetTracker`
(Python) both enforce a hard cap; the SQLite cache absorbs retries and
re-runs of identical (model, temperature, messages) tuples.

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: YANDEX_FOLDER_ID is not set` | env var missing | `export YANDEX_FOLDER_ID=...` (or lowercase `folder_id`) |
| `RuntimeError: yandex-ai-studio-sdk is not installed` | dev extra not installed | `uv pip install yandex-ai-studio-sdk` or `make install` |
| `BudgetExceededError` | hard cap reached | raise `hard_cap_rub` in `configs/llm/yandex.yaml` or wait for next session |
| 4xx from SDK | bad folder id / key | re-check provisioning step, role bindings |
| 429 from SDK | rate limit | reduce `concurrency.max_in_flight`; Phase 0 default is 4 |
