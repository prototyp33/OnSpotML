# Tests

This directory contains all tests for the project's source code, helping to ensure reliability and prevent regressions.

**Types of Tests:**
- **Unit Tests:** Test individual functions or classes in isolation.
- **Integration Tests:** Test interactions between different components or modules.
- **Data Tests:** Validate properties or consistency of data (though some of this might also be in `src/data_validation/` if it's part of a pipeline).

Tests should be organized into subdirectories that mirror the structure of the `src/` directory where possible (e.g., `tests/features/` for tests related to `src/features/`).

Use a testing framework like `pytest` for writing and running tests. 