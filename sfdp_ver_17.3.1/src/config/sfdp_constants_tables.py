"""
SFDP_CONSTANTS_TABLES - Centralized Constants and Adjustment Tables
=====================================================================
COMPREHENSIVE CONSTANTS MANAGEMENT FOR MULTI-PHYSICS SIMULATION

Python port of the MATLAB constants management system.

PURPOSE:
Centralized repository for all empirical constants, adjustment factors,
physics parameters, and validation thresholds used throughout the
6-layer hierarchical simulation framework.

DESIGN PHILOSOPHY:
- Single source of truth for all numerical constants
- Material-specific parameter organization  
- Physics-based hierarchical structure
- Easy maintenance and calibration
- Complete documentation with physical meaning

Author: SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr)
Date: January 2025
License: Academic Research Use Only
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class Ti6Al4VProperties(BaseModel):
    """Ti-6Al-4V material properties."""
    # Base properties at room temperature
    density: float = 4420.0  # kg/m³, Boyer (1996)
    melting_point: float = 1933.0  # K, NIST database
    thermal_conductivity_base: float = 7.0  # W/m·K at 298K, Mills (2002)
    specific_heat_base: float = 560.0  # J/kg·K at 298K, NIST
    elastic_modulus: float = 114e9  # Pa, Boyer (1996)
    poisson_ratio: float = 0.34  # dimensionless, Boyer (1996)
    
    # Johnson-Cook plasticity model parameters
    jc_A: float = 1098e6  # Pa, yield strength coefficient
    jc_B: float = 1092e6  # Pa, hardening modulus
    jc_n: float = 0.93  # dimensionless, hardening exponent
    jc_C: float = 0.014  # dimensionless, strain rate coefficient
    jc_m: float = 1.1  # dimensionless, thermal softening exponent
    jc_T_melt: float = 1933.0  # K, melting temperature
    jc_T_ref: float = 298.0  # K, reference temperature
    
    # Temperature-dependent property coefficients
    k_coeffs: List[float] = Field(default_factory=lambda: [7.0, 0.012, -1.2e-6])  # Thermal conductivity
    cp_coeffs: List[float] = Field(default_factory=lambda: [560, 0.15, -8.5e-5])  # Specific heat
    yield_activation_energy: float = 350000.0  # J/mol
    gas_constant: float = 8.314  # J/mol·K


class EmpiricalAdjustments(BaseModel):
    """Empirical adjustment factors for Layer 3."""
    physics_weight: float
    empirical_weight: float
    correction_factor: float
    bias_correction: float
    uncertainty_factor: float


class Layer3Adjustments(BaseModel):
    """Layer 3 empirical assessment adjustments."""
    temperature: EmpiricalAdjustments = Field(
        default_factory=lambda: EmpiricalAdjustments(
            physics_weight=0.75,
            empirical_weight=0.25,
            correction_factor=1.12,
            bias_correction=-15.3,
            uncertainty_factor=0.08
        )
    )
    tool_wear: EmpiricalAdjustments = Field(
        default_factory=lambda: EmpiricalAdjustments(
            physics_weight=0.60,
            empirical_weight=0.40,
            correction_factor=0.89,
            bias_correction=0.005,
            uncertainty_factor=0.15
        )
    )
    surface_roughness: EmpiricalAdjustments = Field(
        default_factory=lambda: EmpiricalAdjustments(
            physics_weight=0.45,
            empirical_weight=0.55,
            correction_factor=1.23,
            bias_correction=-0.12,
            uncertainty_factor=0.20
        )
    )


class Layer4Corrections(BaseModel):
    """Layer 4 empirical data correction constants."""
    # Source weights
    physics_advanced_weight: float = 0.40
    physics_simplified_weight: float = 0.30
    empirical_ml_weight: float = 0.20
    experimental_data_weight: float = 0.10
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.65
    low_confidence_threshold: float = 0.45
    
    # Correction intensity
    temperature_correction: float = 0.08
    tool_wear_correction: float = 0.12
    surface_roughness_correction: float = 0.15


class KalmanVariableConfig(BaseModel):
    """Kalman filter configuration for a specific variable."""
    correction_range: Tuple[float, float]
    adaptation_rate: float
    stability_threshold: float
    innovation_weight: float
    process_noise_base: float
    measurement_noise_base: float


class KalmanConstants(BaseModel):
    """Adaptive Kalman filter constants."""
    # Variable-specific dynamics
    temperature: KalmanVariableConfig = Field(
        default_factory=lambda: KalmanVariableConfig(
            correction_range=(0.10, 0.15),  # ±10-15%
            adaptation_rate=0.05,
            stability_threshold=0.02,
            innovation_weight=0.6,
            process_noise_base=0.01,
            measurement_noise_base=0.02
        )
    )
    tool_wear: KalmanVariableConfig = Field(
        default_factory=lambda: KalmanVariableConfig(
            correction_range=(0.08, 0.12),  # ±8-12%
            adaptation_rate=0.04,
            stability_threshold=0.015,
            innovation_weight=0.7,
            process_noise_base=0.015,
            measurement_noise_base=0.03
        )
    )
    surface_roughness: KalmanVariableConfig = Field(
        default_factory=lambda: KalmanVariableConfig(
            correction_range=(0.12, 0.18),  # ±12-18%
            adaptation_rate=0.06,
            stability_threshold=0.025,
            innovation_weight=0.8,
            process_noise_base=0.02,
            measurement_noise_base=0.04
        )
    )
    
    # Gain calculation constants
    min_gain: float = 0.01
    max_gain: float = 0.5
    learning_rate: float = 0.1
    forgetting_factor: float = 0.95


class PhysicalBounds(BaseModel):
    """Physical bounds for validation."""
    absolute_min: float
    absolute_max: float
    typical_min: float
    typical_max: float
    safety_factor: float


class ValidationConstants(BaseModel):
    """Validation and quality assurance thresholds."""
    # Physical bounds
    temperature_bounds: PhysicalBounds = Field(
        default_factory=lambda: PhysicalBounds(
            absolute_min=20.0,
            absolute_max=1200.0,
            typical_min=100.0,
            typical_max=800.0,
            safety_factor=1.2
        )
    )
    tool_wear_bounds: PhysicalBounds = Field(
        default_factory=lambda: PhysicalBounds(
            absolute_min=0.0,
            absolute_max=2.0,
            typical_min=0.01,
            typical_max=0.5,
            safety_factor=1.5
        )
    )
    surface_roughness_bounds: PhysicalBounds = Field(
        default_factory=lambda: PhysicalBounds(
            absolute_min=0.05,
            absolute_max=50.0,
            typical_min=0.5,
            typical_max=10.0,
            safety_factor=1.3
        )
    )
    
    # Statistical thresholds
    normality_p_threshold: float = 0.05
    outlier_z_threshold: float = 3.5
    variance_ratio_max: float = 4.0
    correlation_significance: float = 0.05
    confidence_level: float = 0.95
    
    # Quality thresholds
    excellent_threshold: float = 0.90
    good_threshold: float = 0.75
    acceptable_threshold: float = 0.60
    poor_threshold: float = 0.45


class ComputationalConstants(BaseModel):
    """Computational and system configuration constants."""
    # Parallel processing
    parallel_min_data_size_mb: int = 10
    parallel_min_computation_time_sec: int = 5
    parallel_min_worker_count: int = 2
    parallel_overhead_factor: float = 0.2
    parallel_memory_per_worker_mb: int = 512
    parallel_task_granularity_threshold: int = 100
    
    # Memory management
    max_allocation_mb: int = 2048
    garbage_collection_threshold: float = 0.8
    cache_size_mb: int = 256
    streaming_threshold_mb: int = 100
    
    # Logging configuration
    default_log_level: str = "INFO"
    max_log_file_size_mb: int = 50
    log_rotation_count: int = 5
    console_output: bool = True
    file_output: bool = True
    structured_format: bool = True
    performance_logging: bool = True
    
    # Convergence
    max_iterations: int = 1000
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-8
    stagnation_threshold: int = 10


class TaylorExtendedConstants(BaseModel):
    """Taylor tool life extended model constants."""
    # Ti-6Al-4V specific (carbide tools)
    C_base: float = 120.0  # Base Taylor constant (m/min)
    n: float = 0.25  # Tool life exponent
    a: float = 0.75  # Feed rate exponent
    b: float = 0.15  # Depth of cut exponent
    c: float = 0.50  # Hardness exponent
    
    # Temperature dependence
    activation_energy: float = 45000.0  # J/mol
    reference_temp: float = 298.0  # K
    temp_coefficient: float = 0.003
    
    # Wear mechanisms
    abrasive_k_coefficient: float = 1.2e-9
    abrasive_hardness_factor: float = 2.1
    diffusion_d_coefficient: float = 2.5e-12
    diffusion_temp_sensitivity: float = 0.08
    oxidation_k_ox: float = 1.8e-10
    oxidation_oxygen_factor: float = 1.5


class MachineLearningConstants(BaseModel):
    """Machine learning and AI model constants."""
    # Neural network
    nn_hidden_layers: List[int] = Field(default_factory=lambda: [64, 32, 16])
    nn_learning_rate: float = 0.001
    nn_dropout_rate: float = 0.2
    nn_batch_size: int = 32
    nn_max_epochs: int = 500
    nn_early_stopping_patience: int = 50
    
    # SVR
    svr_kernel: str = "rbf"
    svr_C_parameter: float = 1.0
    svr_gamma: str = "scale"
    svr_epsilon: float = 0.1
    
    # GPR
    gpr_kernel_type: str = "matern52"
    gpr_length_scale: float = 1.0
    gpr_noise_level: float = 0.01
    gpr_alpha: float = 1e-10
    
    # Ensemble weights
    ensemble_nn_weight: float = 0.3
    ensemble_svr_weight: float = 0.25
    ensemble_gpr_weight: float = 0.25
    ensemble_physics_weight: float = 0.2


class EmpiricalModelConstants(BaseModel):
    """Empirical model configuration constants."""
    standard_reliability: float = 0.7
    
    # Operating ranges
    cutting_speed_range: Tuple[float, float] = (50.0, 300.0)  # m/min
    feed_rate_range: Tuple[float, float] = (0.05, 0.5)  # mm/rev
    depth_of_cut_range: Tuple[float, float] = (0.2, 3.0)  # mm
    
    # Coverage scores
    within_range_score: float = 0.9
    outside_range_score: float = 0.6
    
    # Material specificity
    ti6al4v_specificity: float = 0.8
    aluminum_specificity: float = 0.85
    steel_specificity: float = 0.75
    inconel_specificity: float = 0.65


class ErrorHandlingConstants(BaseModel):
    """Error handling and recovery constants."""
    # Severity thresholds
    critical_error_threshold: float = 0.5
    major_error_threshold: float = 0.2
    minor_error_threshold: float = 0.05
    
    # Recovery strategy
    max_retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay_ms: int = 100
    fallback_method_timeout_sec: int = 30
    
    # Physics checks
    energy_conservation_tolerance: float = 0.05
    mass_conservation_tolerance: float = 0.01
    momentum_conservation_tolerance: float = 0.03


@dataclass
class SFDPConstants:
    """Main SFDP Constants class."""
    # Material properties
    ti6al4v: Ti6Al4VProperties = field(default_factory=Ti6Al4VProperties)
    
    # Empirical adjustments
    layer3_adjustments: Layer3Adjustments = field(default_factory=Layer3Adjustments)
    layer4_corrections: Layer4Corrections = field(default_factory=Layer4Corrections)
    
    # Kalman filter
    kalman: KalmanConstants = field(default_factory=KalmanConstants)
    
    # Validation
    validation: ValidationConstants = field(default_factory=ValidationConstants)
    
    # Computational
    computational: ComputationalConstants = field(default_factory=ComputationalConstants)
    
    # Taylor extended model
    taylor_extended: TaylorExtendedConstants = field(default_factory=TaylorExtendedConstants)
    
    # Machine learning
    machine_learning: MachineLearningConstants = field(default_factory=MachineLearningConstants)
    
    # Empirical models
    empirical_models: EmpiricalModelConstants = field(default_factory=EmpiricalModelConstants)
    
    # Error handling
    error_handling: ErrorHandlingConstants = field(default_factory=ErrorHandlingConstants)
    
    # Metadata
    version: str = "17.3"
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize metadata after creation."""
        self.total_constants = self._count_constants()
        self.checksum = self._generate_checksum()
    
    def _count_constants(self) -> int:
        """Recursively count total number of constants."""
        count = 0
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, BaseModel):
                    # Count fields in Pydantic models
                    count += len(attr.dict())
                elif isinstance(attr, (int, float, str, list, tuple)):
                    count += 1
        return count
    
    def _generate_checksum(self) -> str:
        """Generate checksum for constants validation."""
        # Convert all constants to a JSON string for hashing
        constants_dict = self.to_dict()
        constants_str = json.dumps(constants_dict, sort_keys=True)
        return hashlib.md5(constants_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """Convert constants to dictionary."""
        result = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                if isinstance(attr, BaseModel):
                    result[attr_name] = attr.dict()
                elif not callable(attr):
                    result[attr_name] = attr
        return result
    
    def print_summary(self):
        """Print constants loading summary."""
        print("📊 Loading SFDP centralized constants and adjustment tables...")
        print("  🔬 Loading material properties and physics constants...")
        print("  ⚙️ Loading empirical adjustment constants...")
        print("  🧠 Loading adaptive Kalman filter constants...")
        print("  ✅ Loading validation and QA thresholds...")
        print("  🖥️ Loading computational configuration constants...")
        print("  🔧 Loading Taylor tool life extended model constants...")
        print("  🤖 Loading ML and AI model constants...")
        print("  🛡️ Loading error handling and recovery constants...")
        print("  ✅ Constants loading completed successfully")
        print(f"  📊 Total constants loaded: {self.total_constants}")
        print(f"  🔐 Constants checksum: {self.checksum}")


def load_constants() -> SFDPConstants:
    """Load and return SFDP constants."""
    constants = SFDPConstants()
    constants.print_summary()
    return constants


if __name__ == "__main__":
    # Test constants loading
    constants = load_constants()
    print("\nConstants loaded successfully!")
    print(f"Version: {constants.version}")
    print(f"Ti-6Al-4V density: {constants.ti6al4v.density} kg/m³")