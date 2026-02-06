# Tests

This directory contains unified tests for the project.

## Structure
- `ml/`: Tests for the ML component (Unit and Integration).
- `backend/`: Tests for the Backend API.
- `e2e/`: End-to-End tests for the Webapp (Playwright).

## Running Tests
Use the Makefile from the root directory:
- `make test-all`: Run all tests.
- `make test-ml`: Run ML tests.
- `make test-backend`: Run backend tests.
- `make test-e2e`: Run E2E tests.
