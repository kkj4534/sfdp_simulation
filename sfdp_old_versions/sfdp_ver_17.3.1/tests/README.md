# SFDP v17.3.1 Test Suite

## Overview

This directory contains test cases for the SFDP v17.3.1 simulation system. The test suite verifies the functionality and performance of all major components.

## Test Structure

```
tests/
├── test_initialization.py      # System initialization tests
├── test_data_loading.py        # Data loading and quality tests
├── test_validation.py          # 5-level validation tests
├── test_tuning.py              # Tuning system tests
├── test_integrity.py           # Integrity verification tests
└── test_integration.py         # End-to-end integration tests
```

## Running Tests

### All Tests
```bash
python -m pytest tests/
```

### Specific Test Module
```bash
python -m pytest tests/test_validation.py
```

### With Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Unit Tests
- Individual function validation
- Component isolation testing
- Parameter boundary testing

### Integration Tests
- Multi-component interaction
- End-to-end workflow validation
- System performance verification

### Validation Tests
- 5-level validation framework
- Score accuracy verification
- Consistency checking

## Expected Results

### Baseline Performance
- Overall validation score: ~53.9%
- Standard deviation: <0.001
- Reproducibility: 100%

### Tuned Performance
- Target validation score: ≥83%
- Level-wise pass criteria: ≥60%
- Convergence within 10 iterations

## Test Data

Test cases use simplified datasets for:
- Faster execution
- Deterministic results
- Isolated component testing

Real validation requires full dataset in `../data/` directory.

## Contact

For test-related questions:
SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr)