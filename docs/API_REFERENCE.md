# SFDP v17.3 API Reference

**Complete API documentation for developers and advanced users**

---

## 📚 Core Modules

### 1. System Initialization

#### `sfdp_initialize_system()`
**File:** `modules/sfdp_initialize_system.py`

```python
def sfdp_initialize_system() -> SimulationState
```

**Description:** Initializes the complete 6-layer hierarchical simulation system.

**Returns:**
- `SimulationState`: Configured system state object

**Example:**
```python
from modules.sfdp_initialize_system import sfdp_initialize_system

state = sfdp_initialize_system()
print(f"Physics confidence: {state.physics.current_confidence}")
```

**State Object Structure:**
```python
class SimulationState:
    meta: dict              # System metadata
    layers: dict           # Layer configurations
    physics: PhysicsState  # Physics foundation
    learning: dict         # ML configurations
    kalman: dict          # Kalman filter settings
```

---

### 2. Data Loading

#### `sfdp_intelligent_data_loader()`
**File:** `modules/sfdp_intelligent_data_loader.py`

```python
def sfdp_intelligent_data_loader(simulation_state: SimulationState) -> Tuple[dict, dict, dict]
```

**Description:** Loads and validates experimental data with quality assessment.

**Parameters:**
- `simulation_state`: Initialized system state

**Returns:**
- `extended_data`: Loaded experimental datasets
- `data_confidence`: Confidence scores for each dataset
- `data_availability`: Data availability flags

**Example:**
```python
extended_data, confidence, availability = sfdp_intelligent_data_loader(state)
print(f"Data confidence: {confidence}")
```

**Data Structure:**
```python
extended_data = {
    'experimental_data': pandas.DataFrame,     # 70 Ti-6Al-4V experiments
    'taylor_coefficients': dict,               # Tool life coefficients
    'material_properties': dict,               # Material database
    'machining_conditions': list,              # Validated conditions
    'tool_specifications': list                # Tool database
}
```

---

### 3. 6-Layer Calculations

#### `sfdp_execute_6layer_calculations()`
**File:** `modules/sfdp_execute_6layer_calculations.py`

```python
def sfdp_execute_6layer_calculations(
    simulation_state: SimulationState,
    physics_foundation: dict,
    selected_tools: dict,
    taylor_results: dict,
    optimized_conditions: dict
) -> Tuple[LayerResults, FinalResults]
```

**Description:** Executes the complete 6-layer hierarchical calculation system.

**Parameters:**
- `simulation_state`: System state object
- `physics_foundation`: Material physics foundation
- `selected_tools`: Tool selection results
- `taylor_results`: Taylor coefficient data
- `optimized_conditions`: Machining conditions

**Returns:**
- `LayerResults`: Individual layer results and status
- `FinalResults`: Final validated simulation outputs

**Example:**
```python
layer_results, final_results = sfdp_execute_6layer_calculations(
    state, physics_foundation, tools, taylor_results, conditions
)

print(f"Temperature: {final_results.cutting_temperature:.1f}°C")
print(f"Layer success: {layer_results.layer_status}")
```

**Results Structure:**
```python
class FinalResults:
    cutting_temperature: float      # °C
    tool_wear_rate: float          # mm/min
    surface_roughness: float       # μm Ra
    cutting_force: float           # N
    system_confidence: float       # 0-1
    primary_source: str           # Best performing layer
    execution_time: float         # seconds
```

---

### 4. Validation Framework

#### `FixedValidation140`
**File:** `sfdp_fixed_validation_140.py`

```python
class FixedValidation140:
    def __init__(self):
        self.max_iterations = 140
        
    def run_validation(self) -> dict
```

**Description:** Runs 140-iteration validation with dynamic conditions.

**Methods:**
- `run_validation()`: Execute complete validation
- `generate_conditions(iteration)`: Create dynamic test conditions
- `calculate_validation_error(predicted, experimental)`: Compute error metrics

**Example:**
```python
validator = FixedValidation140()
results = validator.run_validation()
print(f"Min error: {results['min_error']:.3f}%")
```

---

### 5. Continuous Tuning

#### `ContinuousTuningSystem`
**File:** `sfdp_continuous_tuning_150.py`

```python
class ContinuousTuningSystem:
    def __init__(self):
        self.max_iterations = 150
        
    def run_continuous_tuning(self) -> None
```

**Description:** Executes 150-iteration continuous auto-tuning with adaptive learning.

**Key Methods:**
- `run_continuous_tuning()`: Execute full tuning sequence
- `generate_adaptive_conditions(iteration)`: Create adaptive test conditions
- `adapt_layer_weights()`: Dynamically adjust layer weights
- `calculate_recent_success_rate()`: Assess recent performance

**Example:**
```python
tuner = ContinuousTuningSystem()
tuner.max_iterations = 50  # Reduce for testing
tuner.run_continuous_tuning()
print(f"Best error: {tuner.best_error:.3f}%")
```

---

## 🔧 Helper Modules

### Physics Suite
**File:** `helpers/sfdp_physics_suite.py`

#### Key Functions:
```python
def calculate_cutting_temperature(conditions, material_props) -> float
def estimate_tool_wear_rate(conditions, tool_specs) -> float
def predict_surface_roughness(conditions, tool_geometry) -> float
def compute_cutting_forces(conditions, material_props) -> dict
```

### Kalman Fusion
**File:** `helpers/sfdp_kalman_fusion_suite.py`

#### Key Functions:
```python
def adaptive_kalman_filter(physics_pred, empirical_pred, uncertainty) -> dict
def update_kalman_gains(performance_history) -> dict
def fuse_multi_layer_predictions(layer_results, weights) -> dict
```

### Empirical ML Suite
**File:** `helpers/sfdp_empirical_ml_suite.py`

#### Key Functions:
```python
def apply_random_forest_prediction(features, simulation_state) -> dict
def apply_support_vector_regression(features, simulation_state) -> dict
def apply_neural_network_prediction(features, simulation_state) -> dict
def apply_gaussian_process_regression(features, simulation_state) -> dict
```

---

## 📊 Configuration API

### Constants Table
**File:** `configs/sfdp_constants_tables.py`

#### Material Properties
```python
MATERIAL_PROPERTIES = {
    'Ti6Al4V': {
        'density': 4430,                    # kg/m³
        'thermal_conductivity': 7.3,        # W/m·K
        'specific_heat': 560,               # J/kg·K
        'hardness': 334,                    # HV
        'correction_factor': 1.0            # No manipulation
    }
}
```

#### Physics Constants
```python
PHYSICS_CONSTANTS = {
    'stefan_boltzmann': 5.67e-8,           # W/m²·K⁴
    'gas_constant': 8.314,                 # J/mol·K
    'avogadro_number': 6.022e23           # 1/mol
}
```

### User Configuration
**File:** `configs/sfdp_user_config.py`

#### Validation Settings
```python
VALIDATION_CONFIG = {
    'target_error': 15.0,                  # % maximum acceptable error
    'confidence_threshold': 0.6,           # minimum confidence
    'max_iterations': 140                  # validation iterations
}
```

#### System Settings
```python
SYSTEM_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',
    'parallel_processing': False,
    'cache_results': True
}
```

---

## 🎯 Advanced Usage Patterns

### Custom Physics Models

#### Adding New Physics Calculation
```python
def custom_thermal_model(conditions, material_props):
    """Custom thermal calculation implementation"""
    
    # Extract parameters
    speed = conditions['cutting_speed']
    feed = conditions['feed_rate']
    depth = conditions['depth_of_cut']
    
    # Your physics model here
    temperature = your_thermal_calculation(speed, feed, depth, material_props)
    
    # Return standardized format
    return {
        'temperature': temperature,
        'confidence': 0.85,
        'method': 'Custom_Thermal_Model',
        'valid': True
    }

# Integration into Layer 1
def execute_layer_1_advanced_physics(...):
    # Standard calculations
    standard_results = current_thermal_analysis(...)
    
    # Add custom model
    custom_results = custom_thermal_model(conditions, materials)
    
    # Weighted combination
    if custom_results['valid']:
        combined_temp = (
            standard_results['temperature'] * 0.7 + 
            custom_results['temperature'] * 0.3
        )
        confidence = min(standard_results['confidence'], custom_results['confidence'])
    
    return combined_results
```

### Custom Validation Metrics

#### Implementing New Error Calculation
```python
def weighted_absolute_percentage_error(predicted, experimental, weights=None):
    """Custom WAPE calculation with optional weights"""
    
    if weights is None:
        weights = np.ones(len(predicted))
    
    # Calculate weighted APE
    ape = np.abs((predicted - experimental) / experimental) * 100
    wape = np.average(ape, weights=weights)
    
    return wape

def custom_validation_framework(simulation_results, experimental_data):
    """Enhanced validation with multiple metrics"""
    
    # Standard metrics
    mae = mean_absolute_error(simulation_results, experimental_data)
    rmse = np.sqrt(mean_squared_error(simulation_results, experimental_data))
    
    # Custom metric
    wape = weighted_absolute_percentage_error(simulation_results, experimental_data)
    
    # Confidence assessment
    confidence = 1.0 - (wape / 100.0)
    
    return {
        'error_percentage': wape,
        'mae': mae,
        'rmse': rmse,
        'confidence': confidence,
        'validation_method': 'Enhanced_Validation'
    }
```

### Parallel Processing Integration

#### Batch Processing Template
```python
from multiprocessing import Pool, cpu_count
import numpy as np

def parallel_simulation_batch(conditions_list):
    """Run multiple simulations in parallel"""
    
    def single_simulation(conditions):
        # Your simulation call here
        return run_sfdp_simulation(conditions)
    
    # Use all available cores
    num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(single_simulation, conditions_list)
    
    return results

# Usage example
speed_range = np.linspace(60, 100, 10)
feed_range = np.linspace(0.2, 0.3, 5)

conditions_list = [
    {'cutting_speed': s, 'feed_rate': f, 'depth_of_cut': 0.5}
    for s in speed_range for f in feed_range
]

# Run parallel batch
results = parallel_simulation_batch(conditions_list)
```

---

## 🔍 Debugging and Diagnostics

### System Health Check

```python
def system_health_check():
    """Comprehensive system diagnostic"""
    
    health_report = {
        'modules': {},
        'data': {},
        'performance': {},
        'errors': []
    }
    
    try:
        # Module imports
        from modules.sfdp_initialize_system import sfdp_initialize_system
        health_report['modules']['initialization'] = True
        
        from modules.sfdp_execute_6layer_calculations import sfdp_execute_6layer_calculations
        health_report['modules']['calculations'] = True
        
        # System initialization
        state = sfdp_initialize_system()
        health_report['performance']['init_time'] = time.time()
        
        # Data loading test
        extended_data, confidence, availability = sfdp_intelligent_data_loader(state)
        health_report['data']['loading'] = True
        health_report['data']['confidence'] = confidence
        
        # Layer test
        test_conditions = {'cutting_speed': 80, 'feed_rate': 0.25, 'depth_of_cut': 0.5}
        # Mock other parameters for test
        physics_foundation = {'material_properties': {'density': 4430}}
        selected_tools = {'primary_tool': {'nose_radius': 0.8e-3}}
        taylor_results = {'coefficients': {'n': 0.25, 'C': 150.0}}
        
        layer_results, final_results = sfdp_execute_6layer_calculations(
            state, physics_foundation, selected_tools, taylor_results, test_conditions
        )
        
        health_report['performance']['layer_success'] = sum(layer_results.layer_status)
        health_report['performance']['execution_time'] = layer_results.layer_execution_times
        
    except Exception as e:
        health_report['errors'].append(str(e))
    
    return health_report

# Usage
health = system_health_check()
print(f"System Health: {health}")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_simulation(conditions):
    """Profile simulation performance"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run simulation
    results = run_complete_simulation(conditions)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 time-consuming functions
    
    return results, stats

# Usage
conditions = {'cutting_speed': 80, 'feed_rate': 0.25, 'depth_of_cut': 0.5}
results, performance_stats = profile_simulation(conditions)
```

---

## 📈 Extension Guidelines

### Adding New Materials

1. **Update Material Database**
2. **Add Physics Constants**
3. **Validate Against Experimental Data**
4. **Update Tool Selection Logic**

### Adding New Tools

1. **Extend Tool Database**
2. **Update Tool Selection Algorithm**
3. **Add Tool-Specific Physics Models**
4. **Validate Tool Performance**

### Adding New Physics Models

1. **Implement Standardized Interface**
2. **Integrate into Appropriate Layer**
3. **Add Confidence Assessment**
4. **Validate Against Experimental Data**

---

This API reference provides comprehensive documentation for developers working with SFDP v17.3. For additional details, refer to the inline code documentation and USER_GUIDE.md.