# ObjDet Makefile
# Convenience scripts for development tasks

.PHONY: help install lint format test test-unit test-integration docs docs-serve docs-clean clean

# Default target
help:
	@echo "ObjDet Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies (dev + docs)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff check and pyrefly type checker"
	@echo "  make format           Run ruff format"
	@echo "  make pre-commit       Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build HTML documentation"
	@echo "  make docs-serve       Build and serve docs locally"
	@echo "  make docs-clean       Clean built documentation"
	@echo "  make docs-check       Build docs with warnings as errors"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts and caches"

# ============================================================================
# Setup
# ============================================================================
install:
	uv sync --frozen

install-dev:
	uv sync --frozen --all-groups

# ============================================================================
# Code Quality
# ============================================================================
lint:
	uv run ruff check src tests
	uv run pyrefly check src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

pre-commit:
	uv run pre-commit run --all-files

# ============================================================================
# Testing
# ============================================================================
test:
	uv run pytest tests -v

test-unit:
	uv run pytest tests/unit -v

test-cov:
	uv run pytest tests/unit -v --cov=src/objdet --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ============================================================================
# Documentation
# ============================================================================
DOCS_DIR := docs
DOCS_BUILD_DIR := $(DOCS_DIR)/_build

docs:
	uv run sphinx-build -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
	@echo "Documentation built: $(DOCS_BUILD_DIR)/html/index.html"

docs-serve: docs
	@echo "Serving docs at http://localhost:8000"
	python -m http.server 8000 -d $(DOCS_BUILD_DIR)/html

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)
	@echo "Cleaned documentation build directory"

docs-check:
	uv run sphinx-build -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html -W --keep-going
	@echo "Documentation built with warnings as errors"

docs-linkcheck:
	uv run sphinx-build -b linkcheck $(DOCS_DIR) $(DOCS_BUILD_DIR)/linkcheck
	@echo "Link check complete"

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned all build artifacts"
