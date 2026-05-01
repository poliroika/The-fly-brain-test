# FlyBrain Optimizer — план реализации v3 (Rust + Python + Yandex AI Studio + DataSphere)

Финальный план с учётом всех ваших ответов. Все решения зафиксированы.

| Решение | Выбор |
|---|---|
| Язык ядра | **Rust** (hot path) + **Python** (ML, LLM, тренировка), maturin mixed layout |
| LLM-провайдер | **Yandex AI Studio** через `yandex-cloud-ml-sdk` |
| LLM модели | **Гибрид**: `yandexgpt-lite/latest` для быстрых агентов, `yandexgpt/latest` (Pro) для Critic/Judge/Verifier/Planner |
| Embeddings | **Yandex** `text-search-doc` + `text-search-query` |
| Бюджет | **2000 ₽ всего** (200 dev / 900 train / 900 eval), `hard_cap_rub: 2000` |
| Источник connectome | **Zenodo tar + synthetic fallback** |
| Бенчмарки | **Минимальный сет**: HumanEval + GSM8K + BBH-mini + synthetic routing |
| Инфраструктура | **Yandex DataSphere / Compute Cloud** (Dockerfile + terraform) |
| Формат отчёта | **ML-исследователь** (~10–15 стр.) |
| Credentials | `YANDEX_API_KEY` + `folder_id` (предоставлены, session-only) |

Дифф от v2: добавлены §4.5 (инфраструктура), §4.6 (mapping агентов на lite/Pro), жёсткий бюджет 2000 ₽ и его mitigation’ы
в §7; время +1–2 дня на Dockerfile + terraform в Phase 12 (или +1 в Phase 0, если хотим CI билдить образ сразу).

---

## 1. Что в Rust, что в Python — и почему

Главный принцип: **Rust для всего, что не зависит от LLM**, **Python для всего, что зависит**.

### Rust (cargo workspace, ~6 крейтов)

| Крейт | Назначение | Почему Rust |
|---|---|---|
| `flybrain-core` | базовые типы (`AgentSpec`, `TraceStep`, `Trace`, `GraphAction`, `VerificationResult`, `AgentGraph`), сериализация в JSONL/MessagePack | детерминизм, скорость, общий source-of-truth для всех остальных крейтов |
| `flybrain-graph` | загрузка connectome (Zenodo tar parsing, Codex CSV), compression (Louvain/Leiden/spectral/region_agg/celltype_agg) на `petgraph`, статистика | большие графы (54M edges), CPU-bound; Python будет тормозить |
| `flybrain-runtime` | message bus, agent graph mutate API, scheduler tick, trace JSONL writer, dynamic graph hashing | hot loop вызывается на каждом шаге MAS; Rust убирает GIL |
| `flybrain-verify` | детерминированные verifier’ы: schema (jsonschema-rs), tool_use, budget, trace (loop detection, redundancy) | их вызывают на каждом шаге, должны быть быстрыми |
| `flybrain-py` | PyO3-биндинги, экспортирующие всё выше как `flybrain_native` | единственный entrypoint из Python |
| `flybrain-cli` | binary `flybrain` (build / sim / bench / report) | удобно иметь self-contained CLI |

Сборка через **maturin**, mixed Rust/Python project layout (см. §3).

### Python

| Пакет | Назначение | Почему Python |
|---|---|---|
| `flybrain.llm` | `LLMClient` ABC + `YandexClient`, `MockClient` | `yandex-cloud-ml-sdk` — Python-only |
| `flybrain.agents` | `AgentSpec`-инстансы (через биндинги) + промпты + Python-обёртка `Agent.step()` | агент дёргает LLM → должен быть на Python |
| `flybrain.controller` | GNN / RNN / LearnedRouter на PyTorch + PyG | весь ML-экосистема на Python; `tch-rs`/`burn` ещё не дотягивают |
| `flybrain.embeddings` | task / agent / trace / graph / fly emb на sentence-transformers | то же |
| `flybrain.training` | SSL / sim / IL / RL / bandit / PPO / offline RL | PyTorch + Hydra |
| `flybrain.benchmarks` | HumanEval, GSM8K, BBH-mini, synthetic routing | стандартные датасет-лоадеры на Python |
| `flybrain.eval` | metrics, tables, reports | pandas / matplotlib |

### Граница Rust ↔ Python

PyO3 экспортирует:

```
flybrain_native.types        # FlyGraph, AgentSpec, Trace, GraphAction, VerificationResult
flybrain_native.graph        # build_compressed(source, method, K) -> FlyGraph
flybrain_native.runtime      # AgentGraph, MessageBus, TraceWriter
flybrain_native.verify       # SchemaVerifier, BudgetVerifier, TraceVerifier
flybrain_native.io           # load/save fly_graph_*.pt-equivalent (.fbg формат)
```

Обратные вызовы (Python → Rust → Python): в Rust-`Scheduler` агенты регистрируются как Python-callbacks
(`PyObject` с методом `step`), вызываются через `Python::with_gil`. Это даёт нам Rust-овский цикл, который
оркестрирует Python-овские агенты — типичный паттерн PyO3.

---

## 2. Архитектура верхнего уровня

```
                    ┌────────────────────────────┐
                    │  configs/ (Hydra)          │  Python
                    └──────────────┬─────────────┘
                                   │
   ┌───────────────┐  ┌────────────▼────────────┐  ┌────────────────┐
   │ Zenodo tar    │  │  flybrain-graph (Rust)  │  │ Synthetic fly  │
   │ Codex CSV     │─▶│  parse → compress → save│◀─│ generator      │
   └───────────────┘  └────────────┬────────────┘  └────────────────┘
                                   │ fly_graph_K.fbg
                    ┌──────────────▼─────────────┐
                    │  flybrain.embeddings (Py)  │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐    ┌──────────────────────┐
                    │  flybrain.controller (Py)  │◀──▶│  flybrain.training   │
                    │  GNN | RNN | LearnedRouter │    │  SSL/Sim/IL/RL       │
                    └──────────────┬─────────────┘    └──────────────────────┘
                                   │ GraphAction (Rust type)
                    ┌──────────────▼─────────────┐
                    │  flybrain-runtime (Rust)   │
                    │  scheduler + agent graph   │
                    │  + trace writer            │
                    └──────────────┬─────────────┘
                                   │ Python callbacks
                    ┌──────────────▼─────────────┐
                    │  flybrain.agents (Py)      │
                    │  Agent.step() → LLM        │
                    └──────────────┬─────────────┘
                                   │ messages
                    ┌──────────────▼─────────────┐
                    │  flybrain.llm.YandexClient │
                    │  yandex-cloud-ml-sdk       │
                    └──────────────┬─────────────┘
                                   │ traces
                    ┌──────────────▼─────────────┐
                    │  flybrain-verify (Rust) +  │
                    │  flybrain.verification.llm │
                    │  (factual/reasoning)       │
                    └──────────────┬─────────────┘
                                   │ reward
                    ┌──────────────▼─────────────┐
                    │  flybrain.training         │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │  flybrain.eval + reports   │
                    └────────────────────────────┘
```

**Verification split**: детерминированные verifier’ы (schema, tool_use, budget, trace) — в Rust для скорости.
LLM-зависимые (factual citation matching, reasoning judge) — в Python, т.к. зовут LLM.

---

## 3. Структура репозитория

Используем **maturin mixed layout** + **cargo workspace**.

```
The-fly-brain-test/
├── README.md                          # как есть
├── PLAN.md                            # этот документ
├── Cargo.toml                         # workspace root
├── Cargo.lock
├── pyproject.toml                     # build-backend = "maturin"
├── uv.lock
├── .python-version                    # 3.11
├── rust-toolchain.toml                # stable
├── .pre-commit-config.yaml            # ruff + mypy + cargo fmt + clippy
├── .github/workflows/
│   ├── rust.yml                       # cargo fmt/clippy/test (Linux/macOS)
│   ├── python.yml                     # uv + ruff + mypy + pytest
│   └── ci.yml                         # build wheel via maturin + smoke
├── Makefile                           # make {install,build,lint,test,sim,bench,report}
│
├── crates/                            # Rust workspace members
│   ├── flybrain-core/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, types.rs, agent_graph.rs, trace.rs, action.rs, ids.rs}
│   ├── flybrain-graph/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, sources/{zenodo.rs, codex.rs, synthetic.rs},
│   │            compression/{louvain.rs, leiden.rs, spectral.rs,
│   │                         region_agg.rs, celltype_agg.rs, subgraph.rs},
│   │            stats.rs, builder.rs, format.rs}
│   ├── flybrain-runtime/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, message_bus.rs, scheduler.rs, executor.rs,
│   │            trace_writer.rs, hash.rs}
│   ├── flybrain-verify/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, schema.rs, tool_use.rs, budget.rs, trace.rs,
│   │            unit_test.rs, pipeline.rs}
│   ├── flybrain-cli/
│   │   ├── Cargo.toml
│   │   └── src/main.rs                 # `flybrain build|sim|bench|report`
│   └── flybrain-py/                    # PyO3 module → flybrain_native
│       ├── Cargo.toml                  # crate-type = ["cdylib"]; pyo3 0.27 abi3-py311
│       └── src/{lib.rs, types_py.rs, graph_py.rs, runtime_py.rs,
│                verify_py.rs, io_py.rs}
│
├── flybrain/                          # Python source (mixed layout)
│   ├── __init__.py                    # re-exports flybrain_native + Python API
│   ├── config.py                      # Hydra entrypoints
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                    # LLMClient ABC, LLMResponse
│   │   ├── yandex_client.py           # AsyncYCloudML wrapper
│   │   ├── mock_client.py             # детерминированные ответы
│   │   ├── cache.py                   # SQLite кэш по hash(messages)
│   │   └── pricing.py                 # cost-tracking (RUB → USD-эквивалент)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── registry.py                # name → AgentSpec (импорт из flybrain_native)
│   │   ├── base.py                    # class Agent: spec, llm, step()
│   │   ├── prompts/                   # *.md по агенту
│   │   └── specs/{planner.py, coder.py, debugger.py, verifier.py, ...}
│   │
│   ├── runtime/                       # тонкая Python-обёртка над flybrain-runtime
│   │   ├── __init__.py
│   │   ├── runner.py                  # MAS.run(task, controller) → Trace
│   │   ├── tools/                     # Python-инструменты
│   │   │   ├── python_exec.py
│   │   │   ├── web_search.py          # Yandex Search API опционально
│   │   │   ├── file_tool.py
│   │   │   └── unit_tester.py
│   │   ├── memory/{episodic.py, vector.py}
│   │   └── retriever/bm25.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── task_emb.py                # Yandex text-search-doc embeddings
│   │   ├── agent_emb.py
│   │   ├── trace_emb.py
│   │   ├── graph_emb.py               # GCN over current agent graph
│   │   └── fly_emb.py                 # node2vec / GraphSAGE pretrain
│   │
│   ├── controller/
│   │   ├── __init__.py
│   │   ├── action_space.py            # masking над flybrain_native.GraphAction
│   │   ├── base.py                    # FlyBrainController ABC
│   │   ├── gnn_controller.py          # variant A
│   │   ├── rnn_controller.py          # variant B (A_fly @ h_{t-1})
│   │   ├── learned_router.py          # variant C + fly regularizer
│   │   ├── heads.py
│   │   └── policy.py
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── llm/                       # LLM-зависимые verifier’ы
│   │   │   ├── factual.py             # citation matching через LLM judge
│   │   │   └── reasoning.py           # step-level reasoning lint
│   │   └── pipeline.py                # композирует Rust- и Python-verifier’ы
│   │
│   ├── sim/
│   │   ├── task_generator.py
│   │   ├── synthetic_mas.py
│   │   ├── optimal_routes.py
│   │   └── reward_model.py
│   │
│   ├── training/
│   │   ├── ssl_graph.py
│   │   ├── simulation_pretrain.py
│   │   ├── imitation.py
│   │   ├── rl/{rewards.py, bandit.py, reinforce.py, ppo.py}
│   │   ├── offline_rl.py
│   │   ├── replay.py
│   │   ├── checkpoint.py
│   │   └── loop.py
│   │
│   ├── baselines/
│   │   ├── manual_graph.py
│   │   ├── fully_connected.py
│   │   ├── random_sparse.py
│   │   ├── degree_preserving.py
│   │   ├── learned_router_no_fly.py
│   │   └── flybrain_no_train.py
│   │
│   ├── benchmarks/
│   │   ├── humaneval.py
│   │   ├── gsm8k.py
│   │   ├── bbh_mini.py
│   │   ├── synthetic_routing.py
│   │   └── runner.py                  # parallel + retry + cache
│   │
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── tables.py
│   │   └── reports.py
│   │
│   └── cli.py                         # `flybrain-py` Python entrypoint (parallel to Rust CLI)
│
├── configs/                           # Hydra (как в v1, без изменений)
│   ├── default.yaml
│   ├── llm/yandex.yaml                # folder_id, model_uri, max_tokens
│   ├── graph/{zenodo.yaml, synthetic.yaml, compression/*.yaml}
│   ├── agents/{full_25.yaml, minimal_15.yaml}
│   ├── controller/{gnn.yaml, rnn_fly.yaml, learned_router.yaml}
│   ├── training/{ssl.yaml, simulation.yaml, imitation.yaml, rl_ppo.yaml}
│   └── eval/{humaneval.yaml, gsm8k.yaml, bbh_mini.yaml, full_min.yaml}
│
├── data/
│   ├── flybrain/                      # выходы graph builder’а
│   │   ├── fly_graph_64.fbg
│   │   ├── fly_graph_128.fbg
│   │   ├── node_metadata.json
│   │   └── edge_metadata.json
│   ├── traces/sample/                 # 2-3 cherry-picked execution traces
│   └── benchmarks/                    # downloader пишет сюда
│
├── tests/
│   ├── rust/                          # cargo test внутри каждого крейта
│   ├── python/
│   │   ├── unit/{test_llm_yandex.py, test_controller_*.py, test_metrics.py, ...}
│   │   ├── integration/{test_mas_runtime_mock.py, test_sim_pretraining.py,
│   │                     test_rl_smoke.py, test_native_bindings.py}
│   │   └── fixtures/{tiny_fly_graph.fbg, traces/...}
│   └── conftest.py
│
├── notebooks/
│   ├── 01_explore_connectome.ipynb
│   ├── 02_compression_ablation.ipynb
│   ├── 03_trace_analysis.ipynb
│   └── 04_results_dashboard.ipynb
│
├── docs/
│   ├── architecture.md
│   ├── data_contracts.md
│   ├── rust_python_boundary.md        # NEW: PyO3 типы и переходы
│   ├── yandex_setup.md                # NEW: получение folder_id / API key
│   ├── training_recipes.md
│   ├── experiment_log.md
│   └── final_report.md                # короткий research report (~10–15 стр.)
│
├── scripts/
│   ├── download_zenodo_connectome.sh
│   ├── build_graph.py                 # тонкий wrapper над flybrain-cli
│   ├── collect_expert_traces.py
│   ├── run_baselines.py
│   └── make_report.py
│
└── results/                           # gitignore, кроме sample
    ├── traces/
    ├── checkpoints/
    └── tables/
```

Сборка:
```bash
make install     # uv sync + maturin develop --release
make build       # maturin build --release (wheel в target/wheels/)
make lint        # cargo fmt+clippy + ruff + mypy
make test        # cargo test + pytest
make sim         # flybrain sim (Rust CLI) или flybrain-py sim
make bench       # flybrain-py bench --config eval/full_min.yaml
make report      # python scripts/make_report.py
```

---

## 4. Yandex AI Studio — интеграция

### Регистрация и аутентификация
1. Создать Yandex Cloud account, billing.
2. В консоли создать **service account**, выдать роли `ai.languageModels.user` и `ai.editor`.
3. Получить **folder_id** и **API key** для service account.
4. Положить в `.env`:
   ```
   YANDEX_FOLDER_ID=...
   YANDEX_API_KEY=...
   YANDEX_MODEL_URI=gpt://<folder_id>/yandexgpt/latest
   YANDEX_EMB_DOC=text-search-doc/latest
   YANDEX_EMB_QUERY=text-search-query/latest
   ```

### Python SDK
```python
from yandex_cloud_ml_sdk import AsyncYCloudML
sdk = AsyncYCloudML(folder_id=..., auth=...)
result = await sdk.models.completions("yandexgpt").configure(temperature=0.0).run(messages)
emb   = await sdk.models.text_embeddings("text-search-doc").run(text)
```

### Что делает `flybrain.llm.YandexClient`
- Реализует `LLMClient.complete(messages, tools=None) -> LLMResponse`.
- Конвертирует наши `Message` → Yandex format и обратно.
- Tracking: `tokens_in / tokens_out / latency / cost_rub` сохраняется в `Trace`.
- Кэширует ответы по `hash(messages, model, temperature)` в SQLite — критично для воспроизводимости и
  экономии бюджета на повторных прогонах.
- Function-calling: SDK поддерживает tool calls — мапим в наш `ToolExecutor`.
- Retry / rate limits: exponential backoff, configurable concurrency.

### Embeddings
- `text-search-doc` для индексации (agent role descriptions, retriever corpus).
- `text-search-query` для запросов (task embeddings).
- Dim = 256 (актуальная YandexGPT embedding dimension), кешируем в `data/embeddings_cache/`.

### Бюджетная политика (жёсткая, 2000 ₽)
В Hydra-конфиге `llm/yandex.yaml`:
```yaml
dev_budget_rub: 200      # smoke + отладка Yandex client
train_budget_rub: 900    # Phase 7: ~300–500 expert traces (вместо 1000–2000 в v1/v2)
eval_budget_rub: 900     # Phase 10: ~80–100 задач на бенчмарк (вместо полных dataset’ов)
hard_cap_rub: 2000       # общий потолок, пайплайн падает выше
```

С таким бюджетом **SQLite-кэш ответов и offline RL из traces становятся обязательными**, не опциональными. Каждый retry,
каждый повторный прогон бенчмарка берёт ответы из кэша. Мы ожидаем реальный cost-per-task ~5–10 ₽ на lite
и ~30–60 ₽ на Pro (zhgpt 32k context). 900 ₽ / 7 ₽ ≈ 130 train tasks — рабочее число для IL.

Все агенты идут через `BudgetController` (Rust verifier), который ловит превышение лимитов до
LLM-вызова и заставляет controller `terminate()`.

### 4.5. Инфраструктура: Yandex DataSphere / Compute Cloud

Тренировка и бенчмарки запускаются не локально, а на Yandex Cloud. Поэтому добавляем:

```
infra/
├── Dockerfile                       # multi-stage: rust builder → python runtime
├── .dockerignore
├── docker-compose.yaml              # локальный прогон перед пушем в Yandex
├── terraform/
│   ├── main.tf                      # service account, IAM roles, Container Registry, S3 bucket
│   ├── variables.tf
│   ├── outputs.tf                   # registry_url, sa_id, bucket_name
│   └── versions.tf                  # provider yandex-cloud >= 0.150
└── datasphere/
    ├── project.yaml                 # DataSphere project + community settings
    ├── environment.yaml             # образ + вычислительные конфиги (g1.1, g2.1)
    └── jobs/                        # decl. job specs для фаз 6–10
        ├── sim_pretrain.yaml
        ├── collect_traces.yaml
        ├── imitation.yaml
        ├── ppo.yaml
        └── bench_full_min.yaml
```

Dockerfile-схема:
```dockerfile
# Stage 1: build wheel из Rust + Python
FROM rust:1.83-slim AS rust-builder
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip
RUN pip install maturin
WORKDIR /build
COPY . .
RUN maturin build --release --strip

# Stage 2: runtime
FROM python:3.11-slim
COPY --from=rust-builder /build/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl
RUN pip install yandex-cloud-ml-sdk torch torch-geometric ...
WORKDIR /workspace
COPY flybrain /workspace/flybrain
COPY configs /workspace/configs
COPY scripts /workspace/scripts
ENTRYPOINT ["python", "-m", "flybrain.cli"]
```

Terraform поднимает минимум:
- service account `flybrain-sa` с ролями `ai.languageModels.user`, `ai.editor`,
  `container-registry.images.puller`, `storage.editor`, `datasphere.user`.
- Container Registry для образа.
- S3 bucket для сырых connectome-данных и traces (10–20 GB).
- DataSphere community + project (опционально — можно руками).

DataSphere job spec пример (`jobs/imitation.yaml`):
```yaml
name: flybrain-imitation
desc: "Phase 7: imitation learning from expert traces"
cmd: python -m flybrain.training.imitation +run.config=imitation
env:
  vars:
    YANDEX_FOLDER_ID: ${secrets.YANDEX_FOLDER_ID}
    YANDEX_API_KEY:   ${secrets.YANDEX_API_KEY}
flags:
  attach-project-disk: true
inputs:
  - data/traces/expert/
  - data/flybrain/fly_graph_64.fbg
outputs:
  - results/checkpoints/imitation_${timestamp}.pt
cloud-instance-type: g2.1   # 1 × V100, 32 ГБ RAM
```

### 4.6. Mapping агентов на LLM-модели (гибрид)

Подход: lite — дефолт, Pro — только для «взрослых» ролей, где нужны reasoning или judgment.

```yaml
# configs/llm/yandex.yaml
models:
  lite:
    uri: gpt://${oc.env:YANDEX_FOLDER_ID}/yandexgpt-lite/latest
    max_tokens: 2000
    temperature_default: 0.3
  pro:
    uri: gpt://${oc.env:YANDEX_FOLDER_ID}/yandexgpt/latest
    max_tokens: 4000
    temperature_default: 0.1

agent_to_model:
  # быстрые / механические роли → lite
  Coder:              lite
  Refiner:            lite
  Retriever:          lite
  MemoryReader:       lite
  MemoryWriter:       lite
  ToolExecutor:       lite
  SearchAgent:        lite
  TaskDecomposer:     lite
  Researcher:         lite
  Finalizer:          lite
  Debugger:           lite
  TestRunner:         lite
  ContextCompressor:  lite
  CitationChecker:    lite
  SchemaValidator:    lite      # rule-heavy, lite хватит
  ConstraintChecker:  lite
  FailureRecovery:    lite
  BudgetController:   lite      # простая логика
  MathSolver:         lite      # первый проход

  # reasoning / judgment → Pro
  Planner:            pro       # тяжёлый task decomposition
  Critic:             pro
  Judge:              pro
  Verifier:           pro       # final-stage verifier
```

Ожидаемые цены (Yandex AI Studio, 2025–апрель рефропрайсинг):
- yandexgpt-lite: ~0.40 ₽ / 1k tokens (input+output усреднённо)
- yandexgpt (Pro): ~1.20 ₽ / 1k tokens
- text-search-doc/query: ~0.10 ₽ / 1k tokens

Для trace из ~10 шагов с средним промптом 1500 tokens и ответом 500 tokens выходит ~6 ₽ за задачу на lite,
~18 ₽ на Pro. Гибридная конфигурация (большинство вызовов в lite, 1–2 в Pro на trace) — в районе 8–12 ₽ задачу.
При 900 ₽ train budget это даёт ~75–100 expert traces, что на границе минимума для IL. **Mitigation**: aggressive
simulation pretraining на Phase 6 компенсирует маленькую expert dataset на Phase 7.

---

## 5. Порядок работ — обновлённые фазы

Оценки в **человеко-днях для одного инженера**, full-time.

### Phase 0 — Bootstrap (5 дней) ⬆️ +1 vs v2
- Cargo workspace + maturin mixed layout + `pyproject.toml` с `build-backend = "maturin"`.
- Skeleton всех 6 Rust-крейтов; пустой `flybrain_native` exposes hello-world; Python импортирует.
- `flybrain-core::types` со всеми типами + Serde + PyO3-биндинги.
- `flybrain.llm.MockClient` + `flybrain.llm.YandexClient` (только chat completion + бюджет-трекинг + SQLite-кэш).
- Hydra-конфиги вкл. `llm/yandex.yaml` (lite/Pro mapping).
- `infra/Dockerfile` (multi-stage rust→python) + `make build-image`.
- `infra/terraform/` (service account, IAM, Container Registry, S3 bucket) — скелет без применения.
- Dual CI (rust.yml + python.yml + maturin wheel build + docker build).
- **Exit:** `make install && make test` зелёный; `flybrain --help` (Rust CLI) работает; `python -c "import flybrain_native"` импортирует; образ билдится.

### Phase 1 — FlyBrain Graph (6–8 дней) ⬆️ +2 vs v1
- Rust:
  1. `synthetic.rs` — fly-inspired generator (день 1, нужен сразу).
  2. `zenodo.rs` — скачивание + парсинг tar (parquet/csv).
  3. `compression/{region_agg, celltype_agg, louvain, leiden, spectral}.rs` на `petgraph` + `petgraph-leiden`.
  4. `builder.rs` orchestrator + `format.rs` (двоичный `.fbg` формат).
  5. PyO3-биндинги в `flybrain-py/src/graph_py.rs`.
- Python:
  6. `notebooks/01_explore_connectome.ipynb` — distrib плотности, modularity, степеней.
- **Exit:** для K∈{32,64,128,256} есть `.fbg` + `node_metadata.json`; cargo-тесты на стабильность compression при фиксированном seed.

### Phase 2 — MAS Runtime + Agents (7–9 дней) ⬆️ +2 vs v1
- Rust:
  1. `flybrain-runtime/{message_bus, agent_graph, scheduler, trace_writer, hash}`.
  2. PyO3-биндинги для `AgentGraph.apply(action)`, `Scheduler.tick(callback)`, `TraceWriter`.
- Python:
  3. `flybrain.runtime.tools/{python_exec, web_search, file_tool, unit_tester}` с timeout/retry.
  4. `flybrain.runtime.memory/{episodic, vector}` + `retriever/bm25`.
  5. ≥20 `AgentSpec` в `flybrain.agents.specs/` + промпты в `agents/prompts/`.
  6. `flybrain.runtime.runner.MAS.run(task, controller)` — orchestrates: Rust scheduler ↔ Python agent.step ↔ Yandex LLM.
- **Exit:** `tests/python/integration/test_mas_runtime_mock.py` гоняет 3 типа задач (coding/math/research) на mock-LLM, traces валидные.

### Phase 3 — Verification Layer (4–5 дней) ⬆️ +1 vs v1, ║ с Phase 2
- Rust:
  1. `flybrain-verify/{schema (jsonschema-rs), tool_use, budget, trace, unit_test}`.
  2. PyO3-биндинги.
- Python:
  3. `flybrain.verification.llm/{factual, reasoning}` через Yandex LLM judge.
  4. `flybrain.verification.pipeline` — оркестрирует Rust + Python verifier’ы по `task_type`.
- **Exit:** unit-тесты на pass/fail для каждого verifier’а; `VerificationResult` стабилен.

### Phase 4 — Embeddings (2–3 дня)
- `task_emb.py` — Yandex `text-search-query` с SQLite-кешем.
- `agent_emb.py` — Yandex `text-search-doc` для role descriptions.
- `trace_emb.py` — pooling над step-эмбеддингами + handcrafted features.
- `graph_emb.py` — маленькая GCN над текущим agent graph.
- `fly_emb.py` — node2vec на fly graph + GraphSAGE pretrain (по бюджету).
- **Exit:** `ControllerState.from_runtime(...)` собирается за <50 ms на CPU.

### Phase 5 — Controller (4–5 дней)
- `action_space.py` + masking (через `flybrain_native.GraphAction`).
- `gnn_controller.py` (variant A, основной).
- `rnn_controller.py` (variant B, `A_fly` как sparse weight).
- `learned_router.py` (variant C, + fly regularizer как loss term).
- `heads.py` (action / value / aux verifier prediction).
- **Exit:** все три варианта проходят shape/grad smoke-tests; могут быть инициализированы fly graph’ом.

### Phase 6 — Simulation Pretraining (3 дня)
- `sim/task_generator.py` — синтетика по шаблонам coding/math/research/tool-use.
- `sim/synthetic_mas.py` — `success_prob[task_type]`, `cost`, `error_rate` на агента.
- `sim/optimal_routes.py` — ground-truth маршруты из README §12.1.
- `training/simulation_pretrain.py` — supervised на (state → optimal action).
- **Exit:** controller сходится за <10 минут на CPU и решает sim-задачи на ≥0.85 success.

### Phase 7 — Expert Traces + Imitation (4 дня) ⬆️ +1 vs v1
- `scripts/collect_expert_traces.py`: гонит fully-connected или manual MAS на YandexGPT на subset benchmark’ов. Бюджет: 200–500 traces × 4 task type → ~1000–2000 traces. Кэш ответов критичен.
- `training/imitation.py` — supervised cloning поверх sim-инициализированного controller.
- **Exit:** на held-out subset IL-controller бьёт sim-only по success rate и/или по cost.

### Phase 8 — RL / Bandit Finetuning (4–5 дней)
- `training/rl/rewards.py` — формула из README §12.3.
- `bandit.py` (LinUCB / Thompson) → `reinforce.py` → `ppo.py`.
- `offline_rl.py` для re-training из старых traces.
- **Exit:** PPO на 10–20% benchmark’ов сходится без degradation.

### Phase 9 — Baselines (2 дня)
9 baseline’ов из README §15 как разные controller-классы или статические graph’ы.
- **Exit:** `scripts/run_baselines.py --suite full_min` гоняется одной командой.

### Phase 10 — Benchmarks + Evaluation (4 дня) ⬇️ -1 vs v1 (минимальный сет)
- HumanEval, GSM8K, BBH-mini, synthetic routing — 4 лоадера вместо 8.
- `benchmarks/runner.py` — параллелизм, ретраи, кэш Yandex-ответов.
- `eval/{metrics, tables, reports}.py`.
- **Exit:** одна команда `flybrain-py bench --config eval/full_min.yaml` собирает все цифры в `results/`.

### Phase 11 — Experiments & Report (3 дня)
- 5 экспериментов §18 как отдельные Hydra-runs.
- `notebooks/04_results_dashboard.ipynb` + `docs/final_report.md` (10–15 стр., ML-исследовательский формат).
- 2–3 cherry-picked execution traces в `data/traces/sample/`.
- **Exit:** все 13 пунктов §19 закрыты.

### Phase 12 — Polish & Stretch (3–5 дней) ║ опционально
- Полный terraform apply + DataSphere project bootstrap (если не сделали в Phase 0).
- Перенести агент-step в Rust с использованием `tch-rs` для inline GNN (если профайл оправдает).
- FlyGym/NeuroMechFly embodied pretrain.
- Дополнительные ablation’ы по compression methods и K.
- Yandex Search API как retriever tool.

---

## 6. Сводная оценка времени

| Фаза | Содержание | Оценка | Δ vs v1 |
|---|---|---|---|
| 0 | Bootstrap (cargo workspace + maturin + Yandex client + Dockerfile + terraform skeleton) | 5 | +1 |
| 1 | FlyBrain graph builder (Rust) | 6–8 | +2 |
| 2 | MAS runtime (Rust core + Python agents) + 20+ agents | 7–9 | +2 |
| 3 | Verification (Rust deterministic + Python LLM-judge) | 4–5 | +1 |
| 4 | Embeddings (Yandex emb API) | 2–3 | 0 |
| 5 | Controller (3 варианта) | 4–5 | 0 |
| 6 | Simulation pretraining | 3 | 0 |
| 7 | Expert traces (Yandex) + imitation | 4 | +1 |
| 8 | RL / bandit | 4–5 | 0 |
| 9 | Baselines | 2 | 0 |
| 10 | Benchmarks (минимальный сет) + eval | 4 | -1 |
| 11 | Experiments + report | 3 | 0 |
| 12 | Polish / stretch (опц., вкл. terraform apply + DataSphere bootstrap) | 3–5 | +1 |
| **Итого без stretch** | | **48–56** | **+1 от v2** |
| **С 25% буфером** | | **~60–70** | |

При full-time — **3 календарных месяца** соло. Параллелится по веткам Phase 2/3 и Phase 6/7/8 — двое
инженеров (один на Rust, один на ML) уложатся в **~7 недель**, что близко к оптимуму при таком
архитектурном расщеплении.

**MVP-вариант (если время жёстко лимитировано):**
Phase 0 + 1 (только synthetic graph, без Zenodo) + 2 (15 агентов вместо 25) + 3 (только Rust verifier’ы) +
4 (только task_emb) + 5 (только GNN controller) + 6 + 9 (manual + fully connected + flybrain-no-train) +
10 (только GSM8K + HumanEval) + урезанный 11 = **~4 рабочих недели**.

---

## 7. Риски и mitigations (обновлены)

| Риск | Вероятность | Impact | Mitigation |
|---|---|---|---|
| Rust ↔ Python boundary тратит больше времени, чем экономит | средняя | высокий | минимизировать пересечение границы (batched calls); строгий критерий — Rust только там, где hot path или нужна детерминированность |
| `petgraph` Leiden/Louvain нет на crates.io в нужном виде | средняя | средний | fallback: вызвать Python `python-louvain` через PyO3 callback или сделать собственную реализацию Louvain (~2 дня доп.) |
| Yandex AI Studio rate limits / квоты на free tier | высокая | высокий | SQLite-кэш ответов + concurrency cap + offline RL из traces; `BudgetController` падает до превышения квот |
| **Бюджет 2000 ₽ слишком маленький** для полного бенчмарка + сбор traces | высокая | высокий | «буджетный» режим: ~80 задач на бенчмарк, ~100 expert traces, обязательный SQLite-кэш, упор на simulation pretraining (Phase 6) для компенсации малой expert dataset. При достижении hard-cap пайплайн пишет partial results, не падает |
| YandexGPT free-tier иногда бывает недоступен в пиковые часы | низкая | средний | exponential backoff + retry budget; offline RL из сохранённых traces |
| Terraform поднятие DataSphere и IAM обычно тянет ровно на день из-за ролей/quotas | средняя | средний | в Phase 0 делаем только скелет + `terraform plan`; реальный `apply` — в Phase 12 или перед Phase 7 |
| YandexGPT слабее на reasoning, чем GPT-4o, и проседает на BBH/MATH | средняя | средний | сравниваем не absolute scores, а delta vs baselines (manual graph vs FlyBrain) — это и есть ответ на гипотезу |
| Compression `K=64` теряет содержательную структуру | средняя | средний | держим 4 значения K + ablation §18.4 |
| Бюджет на real-traces съест всё время | высокая | высокий | mock-LLM по умолчанию + кэш + offline RL из сохранённых traces |
| RL не сходится | средняя | высокий | bandit → REINFORCE → PPO; в худшем случае останавливаемся на IL+bandit |
| Verifier reward hacking | средняя | высокий | `terminate` маскируется до min-verifier-score; trace-verifier ловит |
| Скоуп раздувается под влиянием README §20.3 | высокая | средний | строгий MVP-резак: всё, что не в §19, идёт в Phase 12 |
| FlyWire/Zenodo схема изменилась, парсер ломается | низкая | средний | pin версию dataset; в `synthetic.rs` всегда есть fallback |

---

## 8. Статус credentials и блокеры перед Phase 0

| Пункт | Статус |
|---|---|
| `YANDEX_API_KEY` (session-only) | ✅ получен |
| `folder_id` (session-only) | ✅ получен (формат `b1g…`) |
| Бюджет | ✅ 2000 ₽ всего (200/900/900) |
| LLM mapping | ✅ гибрид, см. §4.6 |
| Инфраструктура | ✅ DataSphere/Compute Cloud, Dockerfile + terraform в Phase 0 |
| connectome источник | ✅ Zenodo + synthetic fallback |
| бенчмарки | ✅ минимальный сет |
| формат отчёта | ✅ ML-исследователь |

Блокеров нет. План фиксирован. Ожидаю ваше «поехали» и открываю PR с Phase 0 (Bootstrap).

### Что попадёт в первый PR (Phase 0)

1. `Cargo.toml` workspace root + 6 скелетных крейтов в `crates/` (пустые lib.rs).
2. `pyproject.toml` с `build-backend = "maturin"` + зависимости (torch, torch-geometric, sentence-transformers, hydra-core, pydantic, yandex-cloud-ml-sdk, sqlite, pytest, ruff, mypy).
3. `flybrain-core::types` со всеми pydantic-equivalents на Rust + Serde + PyO3-биндинги.
4. `flybrain-py` экспортирует `flybrain_native` с типами; smoke-тест «импортируется и round-trip JSON».
5. `flybrain.llm.{base, mock_client, yandex_client}` + SQLite-кэш + budget-трекинг.
6. Пустые хинты-модули для всех будущих фаз (чтобы PyCharm/VSCode навигировали): `flybrain/{agents,runtime,embeddings,controller,verification,sim,training,baselines,benchmarks,eval}/__init__.py`.
7. `configs/` (Hydra): default + llm/yandex.yaml + graph/{zenodo,synthetic} + agents/minimal_15.yaml + controller/gnn.yaml.
8. `infra/Dockerfile` (multi-stage) + `infra/terraform/` (скелет без `apply`).
9. `.github/workflows/{rust.yml, python.yml, ci.yml}` + `Makefile`.
10. `tests/python/unit/test_llm_yandex.py` (на mock + cassette для одного реального вызова).
11. `docs/{architecture, data_contracts, rust_python_boundary, yandex_setup}.md`.
12. `README.md` обновлён со ссылкой на PLAN.md и инструкцией по setup.
