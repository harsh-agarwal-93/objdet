# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-09

### 🚀 Features

- Phase 1 commit
- Complete Phase 2
- Commit completed phase 3
- Complete Phase 4
- Completed Phases 5 & 6
- Completed Phase 7
- Phase 8 Complete. Base repo plan implemented.
- Upgrade packages
- Fix inital linting issues and add rule ignores where needed
- Fix inital formating
- Fix typing
- Fix typing part 2 & remove ty
- Fix typing issues part 3
- Fix typing issues part 4 & replace pyright with pyrefly
- Resolve remaining ruff linting and formatting issues
- Update repo links and pre-commit tool versions
- **data:** Add LitData streaming dataset support and DependencyError
- **cli:** Add format-specific path arguments to preprocess command
- **configs:** Add full experiment configs with model, data, and trainer sections
- **data:** Add LITDATA enum value to DatasetFormat
- **tests:** Add comprehensive integration tests for CLI, serving, callbacks, and export
- Improve test coverage with integration tests and mocks
- **webapp:** Add Streamlit + FastAPI web application
- **webapp:** Add standalone Celery app and complete E2E testing
- **webapp:** Add Celery worker container to docker-compose
- **webapp:** Implement comprehensive testing suite for backend and frontend
- **tests:** Add comprehensive integration tests for webapp backend
- **tests:** Add frontend component tests for UI elements
- **webapp:** Add React frontend with Vite and Tailwind
- **frontend:** Add unit tests with Vitest + MSW
- **frontend:** Add E2E tests with Playwright
- **frontend:** Add E2E tests and API documentation
- **lint:** Add sonarqube config
- **webapp:** Link mlflow status to external url and add env example
- **ci:** Enable multiprocess testing with pytest-xdist
- **deploy:** Add healthchecks and consolidate mlflow

### 🐛 Bug Fixes

- Add pyproject.toml directive for updated schema
- Replace ty pre-commit hook with one for pyrefly
- **docs:** Resolve sphinx build errors and warnings
- **docs:** Allow docs/api directory and commit missing files
- Update LitData batch format test for StreamingDataLoader compatibility
- Add proper type annotations to test_base.py
- Resolve pyrefly and ruff issues
- **webapp:** Address pre-commit hook errors
- **tests:** Ensure celery tasks are registered in unit tests
- **webapp:** Docker fixes, remove streamlit, add preventive tests
- **infra:** Resolve docker security warnings
- **tests:** Resolve pre-commit type errors in backend and ml mocks
- **deploy:** Resolve docker compose startup and lint issues
- **sonarqube:** Resolve accessibility and code quality issues identified in scan

### 📚 Documentation

- Update docs and changelog
- Add information about preprocessing datasets
- Update changelog and fix formatting
- Fix MD040 lint error by adding language identifier to fenced code block
- Migrate installation instructions to use uv
- Add placeholder logos to fix build errors
- Update API documentation and add build automation
- Update changelog with API docs and build automation entries
- Update changelog and README with comprehensive test coverage
- Update CHANGELOG and rebuild Sphinx documentation
- Create GEMINI.md context guide and update test commands
- **changelog:** Update changelog
- Update README and add deployment guide
- **changelog:** Update changelog
- Update user guides for inference, models, and training

### ⚡ Performance

- **tests:** Add autouse fixtures to mock external services
- **docker:** Optimize ml and backend dockerfiles for size

### 🔧 Refactoring

- **directory structure:** Remove `src/objdet/data` from gitignore
- **data:** Use native LitData StreamingDataset and StreamingDataLoader (**BREAKING**)
- Clean up LitData collate and update batch type annotations
- **pipelines:** Use top-level imports in tasks.py and fix tests
- **webapp:** Replace Streamlit frontend with React
- **webapp:** Simplify listArtifacts url construction
- **structure:** Restructure project into ml and tests directories
- **deploy:** Use uv for python installation in dockerfiles
- Resolve SonarQube quality and security issues
- **frontend:** Simplify Models.jsx component
- **metrics:** Reduce cognitive complexity of confusion matrix

### 💄 Styling

- Auto-fix by pre-commit hooks

### 🧪 Testing

- Get all unit test to run
- **cli:** Add functional tests for training, inference, and preprocessing
- Add 62 unit tests to improve code coverage
- Add comprehensive system API tests and update config format
- **backend:** Improve test coverage for tasks and system api
- **ml:** Increase coverage and fix lint errors in test suite
- **coverage:** Boost aggregate coverage to 84% with comprehensive unit tests
- **inference:** Fix Predictor score type and precision issues
- **models:** Fix floating point comparison in WBF ensemble

### 📦 Build

- Pyproject.toml to use dynamic versioning for single source of truth
- **deps:** Replace detect-secrets with gitleaks and update ruff

### 👷 CI/CD

- Remove YAML formatting in pre-commit
- Migrate from GitLab CI to GitHub Actions
- Use 'group' flag for docs dependencies in uv sync command
- Add sonarqube workflow and disable local hook

### 🧹 Miscellaneous

- **container version:** Update base docker container image
- Update configuration and documentation
- **webapp:** Disable prop-types rule and add prop-types dependency
- Update backend dockerfile and lockfiles
- **deploy:** Refactor docker configuration and optimize builds
- **config:** Update sonar-project.properties
- **config:** Update project key
- Update GEMINI.md with dev tips
- Update changelog

### Refactor

- Reduce complexity in gradient_monitor and fix SonarQube issues

<!-- generated by git-cliff -->
