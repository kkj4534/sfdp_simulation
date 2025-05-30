# SFDP v17.3 User Guide

**Complete Guide for Smart Fusion-based Dynamic Prediction Framework**

---

## 📖 Table of Contents

1. [What is SFDP?](#what-is-sfdp)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [Input/Output Reference](#inputoutput-reference)
5. [Configuration Guide](#configuration-guide)
6. [Usage Examples](#usage-examples)
7. [Extending the System](#extending-the-system)
8. [Current System Status](#current-system-status)
9. [Troubleshooting](#troubleshooting)

---

## 🤖 What is SFDP?

**SFDP (Smart Fusion-based Dynamic Prediction)** is a comprehensive multi-physics simulation framework specifically designed for **Ti-6Al-4V machining prediction**. It combines advanced physics models, empirical data, and adaptive machine learning to predict machining outcomes with high accuracy.

### Key Capabilities
- **Cutting Temperature Prediction**: Thermal analysis using Carslaw & Jaeger models
- **Tool Wear Estimation**: 6-mechanism coupled wear analysis
- **Surface Roughness Calculation**: Multi-scale fractal and Whitehouse models
- **Cutting Force Analysis**: Merchant and Shaw stress analysis
- **Real-time Adaptation**: Kalman filtering for physics-empirical fusion

### Who Should Use SFDP?
- **Manufacturing Engineers**: Optimizing machining parameters
- **Researchers**: Studying Ti-6Al-4V machining physics
- **Process Planners**: Predicting machining outcomes before production
- **Quality Engineers**: Ensuring consistent machining results

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.12+ required
python --version

# Install dependencies
cd validated_17/code/
pip install -r requirements.txt
```

### Your First Simulation
```bash
# Navigate to code directory
cd validated_17/code/

# Run basic simulation
python sfdp_v17_3_main.py
```

**Expected Output:**
```
================================================================
🏗️  SFDP Framework v17.3 - 6-LAYER HIERARCHICAL ARCHITECTURE 🏗️
================================================================
✅ 6-Layer hierarchical architecture initialized
✅ Data loading complete: 84.2% confidence
✅ Physics foundation established: 98.2% confidence
✅ Tool selection complete: WC_AlCrN_002
✅ 6-layer calculations: 6/6 layers successful
✅ Final validation: 10.6% error ✅
================================================================
```

### Quick Validation Check
```bash
# Run 10-iteration validation (quick test)
python sfdp_fixed_validation_140.py
# Edit the file to set max_iterations = 10 for quick testing
```

---

## 🏗️ System Architecture

### 6-Layer Hierarchical Structure

```
Input → L1 → L2 → L3 → L4 → L5 → L6 → Output
```

| Layer | Name | Function | Confidence |
|-------|------|----------|------------|
| **L1** | Advanced Physics | 3D FEM-level calculations | 70.6% |
| **L2** | Simplified Physics | Classical analytical solutions | 76.0% |
| **L3** | Empirical Assessment | Data-driven analysis | 75.0% |
| **L4** | Data Correction | Experimental adjustment | 80.0% |
| **L5** | Adaptive Kalman | Physics↔Empirical fusion | 50.4% |
| **L6** | Final Validation | Quality assurance | 90.0% |

### Data Flow
```
Machining Conditions → Physics Foundation → Tool Selection 
→ Taylor Coefficients → 6-Layer Calculations → Validation → Results
```

---

## 📊 Input/Output Reference

### Input Parameters

#### Required Machining Conditions
```python
{
    "cutting_speed": 80.0,      # m/min (40-120 recommended)
    "feed_rate": 0.25,          # mm/rev (0.15-0.45 recommended)
    "depth_of_cut": 0.5,        # mm (0.3-0.8 recommended)
    "coolant_flow": 5.0         # L/min (optional, default: 5.0)
}
```

#### Optional Layer Weights
```python
{
    "L1": 0.506,    # Advanced Physics weight (0.1-0.7)
    "L2": 0.100,    # Simplified Physics weight (0.1-0.7)
    "L5": 0.398     # Adaptive Kalman weight (0.1-0.7)
}
```

### Output Results

#### Primary Outputs
```python
{
    "cutting_temperature": 334.2,    # °C
    "tool_wear_rate": 0.0001,        # mm/min
    "surface_roughness": 1.2,        # μm Ra
    "cutting_force": 485.0,          # N
    "validation_error": 10.634,      # %
    "system_confidence": 0.697       # 0-1 scale
}
```

#### Detailed Layer Results
```python
{
    "layer_status": [True, True, True, True, True, True],  # 6 layers
    "layer_confidence": [0.706, 0.760, 0.750, 0.800, 0.504, 0.900],
    "primary_source": "Layer 6: Final Validation",
    "execution_time": 0.06,          # seconds
    "successful_layers": 6           # out of 6
}
```

---

## ⚙️ Configuration Guide

### File Locations

#### Core Configuration
```
configs/
├── sfdp_constants_tables.py    # Physics constants and material properties
└── sfdp_user_config.py         # User-configurable parameters
```

#### Data Paths
```
data/
├── additional_experimental_data.csv     # Extended experimental data
└── integrated_experimental_data.csv     # Main experimental dataset

reference/data_set/
├── extended_materials_csv.txt           # Material properties database
├── extended_tool_specifications.txt     # Tool database
└── extended_validation_experiments.txt  # Validation experiments
```

### Key Configuration Changes

#### 1. Change Data Paths
**File:** `code/modules/sfdp_intelligent_data_loader.py`
```python
# Line ~120-130: Modify data file paths
data_file_paths = {
    'experiments': 'data/your_experimental_data.csv',
    'materials': 'reference/data_set/your_materials.txt',
    'tools': 'reference/data_set/your_tools.txt'
}
```

#### 2. Adjust Physics Constants
**File:** `configs/sfdp_constants_tables.py`
```python
# Modify material properties for different alloys
MATERIAL_PROPERTIES = {
    'Ti6Al4V': {
        'density': 4430,                    # kg/m³
        'thermal_conductivity': 7.3,        # W/m·K
        'specific_heat': 560,               # J/kg·K
        'correction_factor': 1.0            # No manipulation!
    }
}
```

#### 3. Set Validation Targets
**File:** `configs/sfdp_user_config.py`
```python
VALIDATION_CONFIG = {
    'target_error': 15.0,          # % (≤15% target)
    'max_iterations': 140,         # validation iterations
    'confidence_threshold': 0.6    # minimum confidence
}
```

#### 4. Customize Layer Weights
**File:** `code/sfdp_continuous_tuning_150.py`
```python
# Line ~45-55: Initial layer weights
self.current_weights = {
    'L1': 0.5,    # Increase for more physics-based results
    'L2': 0.3,    # Increase for classical model emphasis
    'L5': 0.2     # Increase for adaptive learning
}
```

---

## 💡 Usage Examples

### Example 1: Basic Simulation
```python
#!/usr/bin/env python3
"""Basic SFDP simulation example"""

import sys
sys.path.append('code/')

from sfdp_v17_3_main import main

# Run with default settings
results = main()
print(f"Temperature: {results.get('temperature', 0):.1f}°C")
print(f"Validation: {results.get('validation_score', 0):.3f}")
```

### Example 2: Custom Conditions
```python
#!/usr/bin/env python3
"""Custom machining conditions example"""

import sys
sys.path.append('code/')

from modules.sfdp_initialize_system import sfdp_initialize_system
from modules.sfdp_intelligent_data_loader import sfdp_intelligent_data_loader
from modules.sfdp_execute_6layer_calculations import sfdp_execute_6layer_calculations

# Initialize system
state = sfdp_initialize_system()
extended_data, _, _ = sfdp_intelligent_data_loader(state)

# Set custom conditions
conditions = {
    'cutting_speed': 100.0,    # High-speed machining
    'feed_rate': 0.2,          # Fine feed
    'depth_of_cut': 0.3        # Light cuts
}

# Mock required parameters (replace with real data)
physics_foundation = {'material_properties': {'density': 4430}}
selected_tools = {'primary_tool': {'nose_radius': 0.8e-3}}
taylor_results = {'coefficients': {'n': 0.25, 'C': 150.0}}

# Run calculation
layer_results, final_results = sfdp_execute_6layer_calculations(
    state, physics_foundation, selected_tools, taylor_results, conditions
)

print(f"Temperature: {final_results.cutting_temperature:.1f}°C")
print(f"Tool wear: {final_results.tool_wear_rate:.6f} mm/min")
print(f"Roughness: {final_results.surface_roughness:.2f} μm")
```

### Example 3: Batch Processing
```python
#!/usr/bin/env python3
"""Batch processing multiple conditions"""

import numpy as np
from itertools import product

# Define parameter ranges
speeds = [60, 80, 100]          # m/min
feeds = [0.2, 0.25, 0.3]        # mm/rev
depths = [0.4, 0.5, 0.6]        # mm

results = []

for speed, feed, depth in product(speeds, feeds, depths):
    conditions = {
        'cutting_speed': speed,
        'feed_rate': feed,
        'depth_of_cut': depth
    }
    
    # Run simulation (implement your simulation call here)
    # result = run_sfdp_simulation(conditions)
    # results.append(result)
    
    print(f"Conditions: {speed} m/min, {feed} mm/rev, {depth} mm")

print(f"Processed {len(list(product(speeds, feeds, depths)))} combinations")
```

### Example 4: Continuous Tuning
```python
#!/usr/bin/env python3
"""Run continuous auto-tuning"""

import sys
sys.path.append('code/')

from sfdp_continuous_tuning_150 import ContinuousTuningSystem

# Create tuning system
tuner = ContinuousTuningSystem()

# Run smaller batch for testing
tuner.max_iterations = 30  # Reduce for quick test
tuner.report_interval = 10  # Report every 10 iterations

# Execute tuning
tuner.run_continuous_tuning()

print(f"Best error: {tuner.best_error:.3f}%")
print(f"Final weights: {tuner.current_weights}")
```

---

## 🔧 Extending the System

### Adding New Materials

#### 1. Update Material Database
**File:** `reference/data_set/extended_materials_csv.txt`
```csv
Material,Density,Thermal_Conductivity,Specific_Heat,Hardness
Your_Material,7800,45.0,460,250
```

#### 2. Add Physics Constants
**File:** `configs/sfdp_constants_tables.py`
```python
MATERIAL_PROPERTIES['Your_Material'] = {
    'density': 7800,
    'thermal_conductivity': 45.0,
    'specific_heat': 460,
    'hardness': 250,
    'correction_factor': 1.0
}
```

### Adding New Tools

#### 1. Update Tool Database
**File:** `reference/data_set/extended_tool_specifications.txt`
```
Tool_ID: Your_Tool_001
Material: Carbide
Coating: TiAlN
Nose_Radius: 0.8e-3
Edge_Count: 4
Cost_Per_Edge: 30.0
```

#### 2. Extend Tool Selection
**File:** `code/modules/sfdp_enhanced_tool_selection.py`
```python
# Add tool selection logic for new tool types
def evaluate_tool_performance(tool_spec, conditions):
    # Your custom tool evaluation logic
    score = base_score * custom_factor
    return score
```

### Adding New Physics Models

#### 1. Create New Physics Module
**File:** `code/modules/your_physics_module.py`
```python
def your_physics_calculation(conditions, material_props):
    """
    Implement your physics model here
    """
    # Your physics calculations
    temperature = calculate_temperature(conditions)
    wear_rate = calculate_wear(conditions)
    
    return {
        'temperature': temperature,
        'wear_rate': wear_rate,
        'confidence': 0.85
    }
```

#### 2. Integrate into Layer System
**File:** `code/modules/sfdp_execute_6layer_calculations.py`
```python
# Add to appropriate layer (e.g., Layer 1)
def execute_layer_1_advanced_physics(...):
    # Existing calculations
    existing_results = current_calculations(...)
    
    # Add your module
    your_results = your_physics_calculation(conditions, materials)
    
    # Combine results
    combined_results = combine_physics_results(existing_results, your_results)
    return combined_results
```

### Adding New Validation Methods

#### 1. Create Validation Module
**File:** `code/modules/your_validation_module.py`
```python
def custom_validation_metric(predicted, experimental):
    """
    Implement custom validation metric
    """
    error = calculate_custom_error(predicted, experimental)
    confidence = assess_confidence(error)
    
    return {
        'error': error,
        'confidence': confidence,
        'method': 'Your_Custom_Method'
    }
```

#### 2. Integrate into Validation Framework
**File:** `code/sfdp_fixed_validation_140.py`
```python
# Add to validation pipeline
def enhanced_validation(simulation_results, experimental_data):
    # Existing validation
    standard_validation = current_validation(...)
    
    # Add custom validation
    custom_validation = custom_validation_metric(...)
    
    # Combine validations
    final_validation = combine_validations(standard_validation, custom_validation)
    return final_validation
```

---

## 📋 Current System Status

### Operational Components ✅

| Component | Status | Confidence | Notes |
|-----------|--------|------------|-------|
| **Layer 1**: Advanced Physics | ✅ Working | 70.6% | Using analytical methods (FEM unavailable) |
| **Layer 2**: Simplified Physics | ✅ Working | 76.0% | Classical Jaeger, Taylor models |
| **Layer 3**: Empirical Assessment | ✅ Working | 75.0% | ML-based predictions |
| **Layer 4**: Data Correction | ✅ Working | 80.0% | Experimental data fusion |
| **Layer 5**: Adaptive Kalman | ✅ Working | 50.4% | Physics-empirical fusion |
| **Layer 6**: Final Validation | ✅ Working | 90.0% | Quality assurance |

### Data Sources ✅

| Data Type | Records | Confidence | Source |
|-----------|---------|------------|--------|
| **Experimental Data** | 70 | 73.1% | Ti-6Al-4V machining experiments |
| **Taylor Coefficients** | 49 | 88.5% | Tool life database |
| **Material Properties** | 154 | 89.5% | Multi-material database |
| **Tool Specifications** | 25 | 83.0% | Tool manufacturer data |

### Performance Metrics ✅

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Convergence** | 10.634% ± 1.820% | ≤15% | ✅ Achieved |
| **Success Rate** | 97.3% | >90% | ✅ Exceeded |
| **Best Performance** | 5.342% | <10% | ✅ Achieved |
| **System Reliability** | 6/6 layers working | 100% | ✅ Perfect |

### Known Limitations ⚠️

1. **FEM Module**: Full 3D FEM not available, using analytical approximations
2. **Helper Suites**: Some helper functions use temporary stubs
3. **Material Coverage**: Optimized for Ti-6Al-4V, other materials have lower confidence
4. **Tool Database**: Limited to 25 tool configurations
5. **Computational Speed**: Real-time applications may require optimization

### Future Enhancement Opportunities 🚀

1. **FEM Integration**: Implement full 3D finite element analysis
2. **Extended Materials**: Add more aerospace alloys (Inconel 718, Ti-64 variants)
3. **Advanced ML**: Implement deep learning for empirical layers
4. **Real-time Processing**: Optimize for real-time machining control
5. **Uncertainty Quantification**: Enhanced confidence interval calculations

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'modules'
# Solution: Ensure you're in the code/ directory
cd validated_17/code/
python sfdp_v17_3_main.py
```

#### 2. Data File Not Found
```bash
# Error: FileNotFoundError: data file not found
# Solution: Check data paths in sfdp_intelligent_data_loader.py
# Verify files exist in data/ and reference/data_set/ directories
```

#### 3. Performance Issues
```python
# Issue: Low confidence or high validation error
# Solution: Check input parameter ranges
conditions = {
    'cutting_speed': 80.0,    # Keep in 40-120 range
    'feed_rate': 0.25,        # Keep in 0.15-0.45 range
    'depth_of_cut': 0.5       # Keep in 0.3-0.8 range
}
```

#### 4. Layer Failures
```bash
# Issue: Layer X failed
# Check logs for specific error messages
# Most common: numerical instability in extreme conditions
```

### Debug Mode

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run simulation with detailed output
python sfdp_v17_3_main.py
```

#### Manual Layer Testing
```python
# Test individual layers
from modules.sfdp_execute_6layer_calculations import execute_layer_1_advanced_physics

# Test Layer 1 only
result = execute_layer_1_advanced_physics(conditions, materials, tools)
print(f"Layer 1 result: {result}")
```

### Performance Optimization

#### 1. Reduce Iterations for Testing
```python
# In sfdp_fixed_validation_140.py
class FixedValidation140:
    def __init__(self):
        self.max_iterations = 10  # Reduce from 140 for testing
```

#### 2. Disable Expensive Calculations
```python
# In sfdp_execute_6layer_calculations.py
USE_FULL_FEM = False  # Use analytical approximations
ENABLE_ML_PROCESSING = True  # Keep ML enabled for accuracy
```

#### 3. Parallel Processing (Future Enhancement)
```python
# Template for parallel processing
from multiprocessing import Pool

def parallel_validation(conditions_list):
    with Pool() as pool:
        results = pool.map(run_single_simulation, conditions_list)
    return results
```

### Getting Help

#### 1. Check Logs
```bash
# System logs
tail -f tuning_logs/continuous_tuning_*.log
tail -f validation_logs/sfdp_v17_3.log
```

#### 2. Validate Installation
```python
# Run system integrity check
python -c "
from modules.sfdp_initialize_system import sfdp_initialize_system
state = sfdp_initialize_system()
print('✅ System initialization successful')
"
```

#### 3. Contact Information
- **Repository Issues**: Create GitHub issue with detailed error description
- **Research Inquiries**: memento1087@gmail.com
- **Performance Questions**: Include system logs and configuration details

---

## 📚 Additional Resources

### Documentation Files
- **README.md**: Project overview and quick start
- **FINAL_PERFORMANCE_SUMMARY.md**: Detailed performance metrics
- **VALIDATION_CHECKSUMS.md**: Data integrity verification

### Reference Materials
- **Ti-6Al-4V Machining**: ASM Handbook Volume 16
- **Physics Models**: Carslaw & Jaeger "Conduction of Heat in Solids"
- **Kalman Filtering**: Welch & Bishop "Introduction to Kalman Filter"

### Example Datasets
- **Validation Data**: `reference/data_set/extended_validation_experiments.txt`
- **Material Properties**: `reference/data_set/extended_materials_csv.txt`
- **Tool Specifications**: `reference/data_set/extended_tool_specifications.txt`

---

**🎯 You're now ready to use SFDP v17.3 effectively!**

*For additional questions or advanced usage scenarios, refer to the codebase documentation or contact the development team.*