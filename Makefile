# Makefile — TDD-focused workflow for grokking-mechanism-test
# Usage: `make help`
#
# Conventions
# - Keep diffs small and tests first.
# - Fast feedback: fail fast on unit tests, run lint and type checks in the loop.
# - PyTorch is default; smoke runs stay CPU-only and short.

# ==== Configuration ====
PYTHON := python
PKG ?= scripts
TESTS ?= tests
SMOKE_CFG ?= configs/modular_addition.yaml

# ==== Meta ====
.PHONY: help default init lint type format format-check test test-fast test-watch coverage tdd check smoke train unit integration new-test clean

default: help

help:
	@echo "Targets:"
	@echo "  init           Install deps (and optional requirements-dev.txt)"
	@echo "  lint           Ruff + Black check"
	@echo "  type           Mypy type check"
	@echo "  format         Apply Black formatting"
	@echo "  format-check   Black in check mode"
	@echo "  test           Run full test suite"
	@echo "  test-fast      Fail-fast unit tests (-x --maxfail=1)"
	@echo "  test-watch     Watch tests with pytest-watch (if installed)"
	@echo "  coverage       Run pytest with coverage reports"
	@echo "  tdd            Lint + type + fail-fast unit tests (inner loop)"
	@echo "  check          Lint + type + tests (pre-push)"
	@echo "  smoke          CPU-only smoke training run with tiny epochs"
	@echo "  train          Example short train call (override ARGS=...)"
	@echo "  unit           Only unit tests (mark=unit)"
	@echo "  integration    Only integration tests (mark=integration)"
	@echo "  new-test NAME=feature  Scaffold tests/test_feature.py"
	@echo "  clean          Remove caches and build artifacts"

# ==== Setup ====
init:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

# ==== Quality gates ====
SRC := $(PKG) $(TESTS) scripts

lint:
	ruff check $(SRC)
	black --check $(SRC)
	isort --check-only $(SRC)

type:
	mypy $(PKG) $(TESTS)

format:
	isort $(SRC)
	black $(SRC)

format-check:
	black --check $(SRC)

# ==== Tests ====
test:
	pytest -q

test-fast:
	pytest -q -x --maxfail=1 -m "not integration"

test-watch:
	@if command -v ptw >/dev/null 2>&1; then \
		ptw $(TESTS) -- -q -x --maxfail=1 -m "not integration"; \
	else \
		echo "pytest-watch (ptw) not found. Install with: pip install pytest-watch"; \
	fi

coverage:
	pytest -q --cov=$(PKG) --cov=$(TESTS) --cov-report=term-missing --cov-report=xml

# Inner TDD loop: quick, strict, no long runs
tdd: lint type test-fast

# Pre-push: everything important
check: lint type test coverage

# ==== Training shortcuts ====
smoke:
	$(PYTHON) -m src.scripts.train --config $(SMOKE_CFG) --cpu-only --epochs 2 --seed 1337

train:
	$(PYTHON) -m src.scripts.train $(ARGS)

unit:
	pytest -q -m "unit"

integration:
	pytest -q -m "integration"

# ==== Scaffolding ====
# Create a basic unit test file: make new-test NAME=feature_x
new-test:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make new-test NAME=feature_name"; exit 1; \
	fi
	@mkdir -p $(TESTS)
	@if [ -f "$(TESTS)/test_$(NAME).py" ]; then \
		echo "$(TESTS)/test_$(NAME).py already exists"; \
	else \
		echo "Creating $(TESTS)/test_$(NAME).py"; \
		printf "%s\n" \
"import pytest" \
"" \
"pytestmark = pytest.mark.unit" \
"" \
"def test_$(NAME)_behavior():\n    # Arrange\n    # TODO: set up inputs\n\n    # Act\n    # TODO: call function under test\n\n    # Assert\n    # TODO: assert on outputs\n    assert True" \
		> $(TESTS)/test_$(NAME).py; \
	fi

# ==== Hygiene ====
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml dist build \
		$(PKG)/*.egg-info .benchmarks
	find . -type d -name "__pycache__" -exec rm -rf {} +
