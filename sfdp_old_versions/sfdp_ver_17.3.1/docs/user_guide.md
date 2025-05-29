# SFDP v17.3.1 User Guide

## Installation

### Prerequisites

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Matplotlib ≥ 3.3.0
- SciPy ≥ 1.7.0

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/sfdp-v17.3.1.git
cd sfdp-v17.3.1

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Simulation

```python
# Import core modules
from src.modules.sfdp_initialize_system import sfdp_initialize_system
from src.modules.sfdp_intelligent_data_loader import sfdp_intelligent_data_loader
from src.modules.sfdp_comprehensive_validation import sfdp_comprehensive_validation

# Initialize system
state = sfdp_initialize_system()

# Load data
extended_data, data_confidence, summary = sfdp_intelligent_data_loader(state)

# Generate simulation results (example)
import numpy as np
simulation_results = {
    'cutting_temperature': np.random.normal(350, 25, 10),
    'tool_wear_rate': np.random.normal(0.1, 0.015, 10),
    'surface_roughness': np.random.normal(1.2, 0.2, 10)
}

# Run validation
validation_results = sfdp_comprehensive_validation(state, simulation_results, extended_data)

# Display results
overall_score = validation_results['validation_summary']['overall_confidence']
print(f"Overall Validation Score: {overall_score:.1%}")
```

### Command Line Usage

```bash
# Run main simulation
python src/sfdp_v17_3_main.py

# Run ultra tuning (10 iterations)
python src/sfdp_ultra_tuning_system.py

# Run integrity verification (90 rounds)
python src/sfdp_integrity_verification_system.py

# Generate validation plots
python src/sfdp_validation_plotter.py

# View portfolio demo
jupyter notebook examples/SFDP_Portfolio_Demo.ipynb
```

## Configuration

### Key Parameters

#### Physics Parameters
```python
PHYSICS_CONFIDENCE = 0.95
KALMAN_GAIN_RANGE = (0.05, 0.35)  # 5%-35%
NOISE_VARIANCE = 0.01
TAYLOR_MODEL_TYPE = "ENHANCED_CLASSIC"
MAX_TAYLOR_ORDER = 3
```

#### Validation Parameters
```python
VALIDATION_THRESHOLD = 0.60  # 60% pass threshold
TARGET_OVERALL_SCORE = 0.83  # 83% target
DATA_QUALITY_THRESHOLD = 0.60
ANOMALY_DETECTION_THRESHOLD = 0.15  # 15%
```

#### Tuning Parameters
```python
MAX_TUNING_ITERATIONS = 25
EXPERIMENTAL_CORRELATION_WEIGHT = 0.4
ADAPTIVE_STEP_SIZE = 0.1
CONVERGENCE_THRESHOLD = 0.001
```

### Data Paths

Update data paths in configuration files:

```python
DATA_PATHS = {
    'materials': 'data/extended_materials_csv.txt',
    'experiments': 'data/extended_validation_experiments.txt',
    'machining': 'data/extended_machining_conditions.txt',
    'tools': 'data/extended_tool_specifications.txt'
}
```

## Understanding Results

### Validation Levels

1. **Level 1 (Physical Consistency)**: 
   - Target: ≥60%, Achieved: 92.3%
   - Checks physical law compliance

2. **Level 2 (Mathematical Validation)**:
   - Target: ≥60%, Achieved: 98.0%
   - Verifies numerical accuracy

3. **Level 3 (Statistical Validation)**:
   - Target: ≥60%, Achieved: 73.6%
   - Statistical significance testing

4. **Level 4 (Experimental Correlation)**:
   - Target: ≥60%, Achieved: 63.5%
   - Comparison with experimental data

5. **Level 5 (Cross-validation)**:
   - Target: ≥60%, Achieved: 98.0%
   - Generalization performance

### Interpreting Scores

- **80%+ (Excellent)**: High confidence results
- **60-80% (Good)**: Acceptable for most applications
- **<60% (Caution)**: Results require careful interpretation

### Data Quality Metrics

- **Overall Data Confidence**: 84.2%
- **Individual Source Reliability**: Varies by dataset
- **Quality Threshold**: 60% minimum

## Advanced Usage

### Custom Tuning

```python
from src.sfdp_ultra_tuning_system import UltraTuningSystem

# Initialize tuning system
tuner = UltraTuningSystem(
    max_iterations=15,
    target_score=0.85,
    experimental_weight=0.5
)

# Run custom tuning
results = tuner.run_ultra_tuning()
```

### Integrity Verification

```python
from src.sfdp_integrity_verification_system import IntegrityVerificationSystem

# Run integrity check
verifier = IntegrityVerificationSystem(verification_rounds=50)
integrity_results = verifier.run_integrity_verification()
```

### Custom Visualization

```python
from src.sfdp_validation_plotter import plot_validation_errors

# Generate custom plots
final_error, final_score = plot_validation_errors()
```

## Troubleshooting

### Common Issues

1. **Low Level 4 Scores**: 
   - Expected behavior (63.5% is normal)
   - Limited by experimental data availability

2. **Data Loading Errors**:
   - Check data file paths in configuration
   - Verify file format consistency

3. **Convergence Issues**:
   - Increase tuning iterations
   - Adjust learning parameters

### Performance Optimization

- **Memory Usage**: Monitor for large datasets
- **Computation Time**: Tuning can take 10+ minutes
- **Reproducibility**: Use fixed random seeds

## File Structure

```
sfdp_ver_17.3.1/
├── src/
│   ├── modules/           # Core simulation modules
│   ├── config/            # Configuration files
│   ├── helpers/           # Utility functions
│   └── sfdp_*.py         # Main execution scripts
├── data/                  # Input datasets
├── docs/                  # Documentation
├── examples/              # Demo notebooks
├── results/               # Output files
│   ├── plots/            # Generated plots
│   └── logs/             # Tuning histories
└── tests/                 # Test suite
```

## API Reference

### Core Functions

- `sfdp_initialize_system()`: Initialize simulation state
- `sfdp_intelligent_data_loader()`: Load and assess data quality
- `sfdp_comprehensive_validation()`: Run 5-level validation
- `sfdp_ultra_tuning_system()`: Advanced parameter optimization

### Configuration Modules

- `src/config/sfdp_user_config.py`: User settings
- `src/config/sfdp_constants_tables.py`: Physical constants

### Helper Functions

- `src/helpers/sfdp_physics_suite.py`: Physics calculations
- `src/helpers/sfdp_empirical_ml_suite.py`: ML utilities

## Support

For questions, issues, or contributions:

**Contact**: SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr)

## License

Academic Research Use Only

---
**Version**: v17.3.1  
**Last Updated**: May 29, 2025  
**Validation Status**: 83.3% Verified