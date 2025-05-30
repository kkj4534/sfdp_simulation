"""
SFDP_USER_CONFIG - User Configuration File for SFDP v17.3 Framework
=====================================================================
Python port of the MATLAB user configuration file.

CONFIGURATION CATEGORIES:
1. MATLAB and Toolbox Paths → Python package configurations
2. External Toolbox Configurations → External library configurations
3. Data File Locations
4. Simulation Parameters
5. Performance Settings
6. Output Preferences
7. Advanced Settings

Author: SFDP Research Team (memento1087@gmail.com)
Date: January 2025
License: Academic Research Use Only
"""

import os
import platform
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator


class MatlabConfig(BaseModel):
    """MATLAB compatibility configuration (for reference)."""
    version: str = "Python3.9+"
    installation_path: str = ""
    minimum_required_version: str = "3.9"
    required_packages: List[str] = Field(default_factory=lambda: [
        "numpy",
        "scipy",
        "sympy",
        "scikit-learn",
        "pandas"
    ])
    optional_packages: List[str] = Field(default_factory=lambda: [
        "joblib",
        "numba",
        "cupy"
    ])


class ExternalToolboxConfig(BaseModel):
    """Configuration for external toolboxes/libraries."""
    enabled: bool = True
    auto_detect: bool = True
    search_paths: List[str] = Field(default_factory=list)
    manual_path: str = ""
    
    def get_valid_path(self) -> Optional[Path]:
        """Find and return the first valid path for the toolbox."""
        if self.manual_path and Path(self.manual_path).exists():
            return Path(self.manual_path)
        
        for path_str in self.search_paths:
            path = Path(os.path.expanduser(path_str))
            if path.exists():
                return path
        
        return None


class DataLocationsConfig(BaseModel):
    """Data file locations configuration."""
    base_directory: str = "./SFDP_6Layer_v17_3"
    
    # Extended dataset locations
    extended_experiments: str = "extended_data/extended_validation_experiments.csv"
    extended_taylor_coefficients: str = "extended_data/taylor_coefficients_csv.csv"
    extended_materials: str = "extended_data/extended_materials_csv.csv"
    extended_tools: str = "extended_data/extended_tool_specifications.csv"
    extended_conditions: str = "extended_data/extended_machining_conditions.csv"
    extended_targets: str = "extended_data/extended_validation_targets.csv"
    
    # Cache and temporary file locations
    cache_directory: str = "physics_cache"
    temp_directory: str = "temp"
    logs_directory: str = "adaptive_logs"
    
    # Output file locations
    output_directory: str = "output"
    figures_directory: str = "figures"
    reports_directory: str = "reports"
    validation_directory: str = "validation"
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.base_directory,
            self.cache_directory,
            self.temp_directory,
            self.logs_directory,
            self.output_directory,
            self.figures_directory,
            self.reports_directory,
            self.validation_directory
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class SimulationConfig(BaseModel):
    """Simulation parameters configuration."""
    default_material: str = "Ti6Al4V"
    
    # Default cutting conditions
    default_cutting_speed: float = 120.0  # m/min
    default_feed_rate: float = 0.1  # mm/rev
    default_depth_of_cut: float = 1.0  # mm
    default_coolant_type: str = "FLOOD"  # FLOOD, MQL, DRY
    
    # Simulation accuracy and convergence settings
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    cfl_safety_factor: float = 0.35
    mesh_refinement_threshold: float = 0.1
    
    # Physical bounds for validation
    temperature_range: Tuple[float, float] = (25.0, 800.0)  # °C
    wear_range: Tuple[float, float] = (0.001, 1.0)  # mm
    roughness_range: Tuple[float, float] = (0.1, 10.0)  # μm
    force_range: Tuple[float, float] = (10.0, 5000.0)  # N
    stress_range: Tuple[float, float] = (1e6, 2e9)  # Pa


class KalmanConfig(BaseModel):
    """Adaptive Kalman filter settings."""
    enabled: bool = True
    adaptation_mode: str = "VALIDATION_DRIVEN"  # FIXED, ADAPTIVE, VALIDATION_DRIVEN
    
    # Kalman gain parameters
    gain_bounds: Tuple[float, float] = (0.05, 0.35)  # 5-35% correction range
    base_gain: float = 0.15  # Base Kalman gain
    adaptation_rate: float = 0.1  # Rate of gain adaptation
    
    # Innovation and validation parameters
    innovation_threshold: float = 0.1
    validation_weight: float = 0.3
    physics_weight: float = 0.7
    history_length: int = 20
    convergence_tolerance: float = 1e-4
    
    @validator('gain_bounds')
    def validate_gain_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Lower gain bound must be less than upper gain bound")
        return v


class PerformanceConfig(BaseModel):
    """Performance settings configuration."""
    memory_limit: int = 8 * 1024**3  # 8GB in bytes
    auto_garbage_collection: bool = True
    memory_monitoring: bool = True
    
    # Parallel computing settings
    parallel_enabled: bool = False
    auto_detect_workers: bool = True
    max_workers: int = 4
    prefer_local: bool = True
    
    # Execution time limits (seconds)
    total_simulation_limit: int = 3600
    layer_calculation_limit: int = 600
    single_calculation_limit: int = 120
    
    # Performance monitoring
    monitoring_enabled: bool = True
    report_interval: int = 10  # seconds
    detailed_timing: bool = False
    memory_snapshots: bool = False


class OutputConfig(BaseModel):
    """Output and reporting preferences."""
    # Console output settings
    verbosity_level: str = "NORMAL"  # QUIET, NORMAL, VERBOSE, DEBUG
    progress_updates: bool = True
    layer_details: bool = True
    timing_information: bool = True
    confidence_scores: bool = True
    
    # File output settings
    save_results: bool = True
    save_intermediate: bool = False
    save_logs: bool = True
    compression: bool = True
    
    # Figure and plot settings
    generate_plots: bool = True
    save_figures: bool = True
    figure_format: str = "png"  # png, jpg, svg, pdf
    figure_resolution: int = 300  # DPI
    show_interactive: bool = False
    
    # Report generation settings
    generate_summary: bool = True
    generate_detailed: bool = False
    include_validation: bool = True
    include_genealogy: bool = False
    report_format: str = "txt"  # txt, html, pdf, md


class AdvancedConfig(BaseModel):
    """Advanced settings configuration."""
    # Error handling and recovery
    auto_recovery: bool = True
    max_retries: int = 3
    fallback_enabled: bool = True
    emergency_mode: bool = True
    
    # Quality assurance settings
    strict_validation: bool = False
    bounds_checking: bool = True
    consistency_checks: bool = True
    anomaly_detection: bool = True
    
    # Data quality settings
    minimum_confidence: float = 0.3
    outlier_detection: bool = True
    statistical_validation: bool = True
    source_verification: bool = False
    
    # Experimental features
    adaptive_mesh: bool = False
    ml_enhancement: bool = False
    gpu_acceleration: bool = False
    distributed_computing: bool = False


class UserConfig(BaseModel):
    """User-specific customizations."""
    name: str = Field(default_factory=lambda: os.environ.get('USER', 'unknown'))
    institution: str = "SFDP Research Team"
    email: str = ""
    research_group: str = ""
    
    # User preferences
    units: str = "METRIC"  # METRIC, IMPERIAL
    decimal_places: int = 3
    scientific_notation: bool = False
    temperature_unit: str = "CELSIUS"  # CELSIUS, FAHRENHEIT, KELVIN
    
    # Custom databases
    custom_materials_enabled: bool = False
    custom_materials_path: str = ""
    custom_tools_enabled: bool = False
    custom_tools_path: str = ""


@dataclass
class SFDPUserConfig:
    """Main SFDP User Configuration class."""
    version: str = "v17.3_UserConfig"
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Sub-configurations
    matlab: MatlabConfig = field(default_factory=MatlabConfig)
    data_locations: DataLocationsConfig = field(default_factory=DataLocationsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    user: UserConfig = field(default_factory=UserConfig)
    
    # External toolboxes/libraries
    external_toolboxes: Dict[str, ExternalToolboxConfig] = field(default_factory=dict)
    
    # Validation results
    validation_results: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize external toolboxes and validate configuration."""
        self._setup_external_toolboxes()
        self.validate()
    
    def _setup_external_toolboxes(self):
        """Setup external toolbox configurations."""
        # Python equivalents of MATLAB toolboxes
        self.external_toolboxes = {
            'fenics': ExternalToolboxConfig(
                search_paths=[
                    './toolboxes/fenics',
                    '~/fenics',
                    '/usr/local/fenics',
                    '/opt/fenics'
                ]
            ),
            'trimesh': ExternalToolboxConfig(
                search_paths=[
                    './toolboxes/trimesh',
                    '~/trimesh'
                ]
            ),
            'pygmo': ExternalToolboxConfig(  # Grey Wolf Optimizer equivalent
                search_paths=[
                    './toolboxes/pygmo',
                    '~/pygmo'
                ]
            )
        }
    
    def validate(self) -> Dict:
        """Validate configuration settings."""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'warnings': [],
            'errors': [],
            'overall_status': 'VALID'
        }
        
        # Validate Python version
        import sys
        if sys.version_info < (3, 9):
            validation_results['warnings'].append(
                f"Python version {sys.version} is older than recommended 3.9+"
            )
        
        # Validate data directories
        base_dir = Path(self.data_locations.base_directory)
        if not base_dir.exists():
            validation_results['warnings'].append(
                f"Base directory does not exist: {base_dir}"
            )
        
        # Validate memory settings
        if self.performance.memory_limit > 16 * 1024**3:  # 16GB
            validation_results['warnings'].append(
                "Memory limit set very high - ensure your system has adequate RAM"
            )
        
        # Validate Kalman filter settings (already validated by pydantic)
        
        # Set overall validation status
        if validation_results['errors']:
            validation_results['overall_status'] = 'INVALID'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'VALID_WITH_WARNINGS'
        
        self.validation_results = validation_results
        return validation_results
    
    def print_config_status(self):
        """Print configuration status to console."""
        print(f"📋 Loading SFDP v17.3 User Configuration...")
        print(f"  📊 Configuration Summary:")
        print(f"    Python Version: {platform.python_version()}")
        print(f"    Base Directory: {self.data_locations.base_directory}")
        print(f"    Kalman Filter: {'ENABLED' if self.kalman.enabled else 'DISABLED'}")
        print(f"    Memory Limit: {self.performance.memory_limit / 1e9:.1f} GB")
        print(f"    Verbosity: {self.output.verbosity_level}")
        
        # Check validation status
        if self.validation_results:
            status = self.validation_results['overall_status']
            if status == 'INVALID':
                print("  ❌ Configuration validation failed - please check settings")
                for error in self.validation_results['errors']:
                    print(f"    ERROR: {error}")
            elif status == 'VALID_WITH_WARNINGS':
                print("  ⚠️  Configuration valid with warnings")
                for warning in self.validation_results['warnings']:
                    print(f"    WARNING: {warning}")
            else:
                print("  ✅ Configuration validation passed")
        
        print("✅ SFDP v17.3 User Configuration Loaded Successfully")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'version': self.version,
            'creation_date': self.creation_date,
            'last_modified': self.last_modified,
            'matlab': self.matlab.dict(),
            'data_locations': self.data_locations.dict(),
            'simulation': self.simulation.dict(),
            'kalman': self.kalman.dict(),
            'performance': self.performance.dict(),
            'output': self.output.dict(),
            'advanced': self.advanced.dict(),
            'user': self.user.dict(),
            'external_toolboxes': {k: v.dict() for k, v in self.external_toolboxes.items()},
            'validation_results': self.validation_results
        }


def load_user_config(config_path: Optional[str] = None) -> SFDPUserConfig:
    """
    Load user configuration from file or create default.
    
    Args:
        config_path: Optional path to configuration file (JSON/YAML)
    
    Returns:
        SFDPUserConfig instance
    """
    if config_path and Path(config_path).exists():
        # TODO: Implement loading from JSON/YAML file
        warnings.warn("Configuration file loading not yet implemented, using defaults")
    
    config = SFDPUserConfig()
    config.print_config_status()
    return config


if __name__ == "__main__":
    # Test configuration loading
    config = load_user_config()
    print("\nConfiguration loaded successfully!")