.DEFAULT_GOAL := help

UV       ?= uv
VENV     ?= .venv
PYTHON   ?= $(VENV)/bin/python
MATURIN  ?= NO_COLOR=1 $(VENV)/bin/maturin

# ANSI: just enough so `make help` is readable.
BLUE   := \033[34m
GREEN  := \033[32m
YELLOW := \033[33m
RESET  := \033[0m

.PHONY: help
help:
	@echo "$(BLUE)FlyBrain Optimizer — Make targets$(RESET)"
	@echo ""
	@echo "  $(GREEN)setup$(RESET)         Create venv, install dev deps, build Rust extension"
	@echo "  $(GREEN)install$(RESET)       Install Python deps into existing venv"
	@echo "  $(GREEN)develop$(RESET)       Rebuild Rust extension into the active venv"
	@echo "  $(GREEN)test$(RESET)          Run Rust + Python unit tests"
	@echo "  $(GREEN)test-rust$(RESET)     Cargo workspace tests (excluding PyO3 crate)"
	@echo "  $(GREEN)test-py$(RESET)       pytest unit tests"
	@echo "  $(GREEN)lint$(RESET)          ruff + mypy + cargo clippy + cargo fmt --check"
	@echo "  $(GREEN)fmt$(RESET)           Auto-format Rust + Python"
	@echo "  $(GREEN)image$(RESET)         Build the Docker runtime image"
	@echo "  $(GREEN)tf-init$(RESET)       terraform init (backend=false)"
	@echo "  $(GREEN)tf-validate$(RESET)   terraform fmt + validate"
	@echo "  $(GREEN)clean$(RESET)         Remove build artefacts (.venv, target/, dist/)"

$(VENV)/bin/activate:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --python $(VENV)/bin/python "maturin>=1.7,<2"

.PHONY: setup
setup: $(VENV)/bin/activate install develop

.PHONY: install
install: $(VENV)/bin/activate
	NO_COLOR=1 $(UV) pip install --python $(VENV)/bin/python -e ".[dev]" --no-build-isolation

.PHONY: develop
develop:
	$(MATURIN) develop --release --manifest-path crates/flybrain-py/Cargo.toml

.PHONY: test
test: test-rust test-py

.PHONY: test-rust
test-rust:
	cargo test --workspace --exclude flybrain-py

.PHONY: test-py
test-py:
	$(VENV)/bin/pytest tests/python/unit -q

.PHONY: lint
lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --exclude flybrain-py --all-targets -- -D warnings
	$(VENV)/bin/ruff check flybrain tests
	$(VENV)/bin/ruff format --check flybrain tests
	$(VENV)/bin/mypy flybrain

.PHONY: fmt
fmt:
	cargo fmt --all
	$(VENV)/bin/ruff check flybrain tests --fix
	$(VENV)/bin/ruff format flybrain tests

.PHONY: image
image:
	docker build -f infra/Dockerfile -t flybrain:dev .

.PHONY: tf-init
tf-init:
	terraform -chdir=infra/terraform init -backend=false

.PHONY: tf-validate
tf-validate:
	terraform -chdir=infra/terraform fmt -check
	terraform -chdir=infra/terraform validate

.PHONY: clean
clean:
	rm -rf $(VENV) target/ build/ dist/ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
