"""
SFDP v17.3 - Modules Package
============================

Core modules for 6-layer hierarchical multi-physics simulation framework.

Modules:
- sfdp_initialize_system: System initialization and state management
- sfdp_intelligent_data_loader: Intelligent data loading with quality assessment
- sfdp_setup_physics_foundation: Physics foundation and parameter setup
- sfdp_execute_6layer_calculations: 6-layer hierarchical calculation execution
- sfdp_taylor_coefficient_processor: Extended Taylor coefficient processing
- sfdp_enhanced_tool_selection: Enhanced tool selection algorithms
- sfdp_comprehensive_validation: Comprehensive validation framework
"""

from .sfdp_initialize_system import sfdp_initialize_system, SimulationState
from .sfdp_intelligent_data_loader import sfdp_intelligent_data_loader, DataQualityMetrics, IntelligentLoader
from .sfdp_setup_physics_foundation import sfdp_setup_physics_foundation, MaterialPhysics, JohnsonCookParameters
from .sfdp_taylor_coefficient_processor import sfdp_taylor_coefficient_processor, TaylorResults, TaylorCoefficients

# These imports will be added as modules are created
# from .sfdp_execute_6layer_calculations import sfdp_execute_6layer_calculations
# from .sfdp_enhanced_tool_selection import sfdp_enhanced_tool_selection
# from .sfdp_comprehensive_validation import sfdp_comprehensive_validation

__all__ = [
    'sfdp_initialize_system',
    'SimulationState',
    'sfdp_intelligent_data_loader',
    'DataQualityMetrics',
    'IntelligentLoader',
    'sfdp_setup_physics_foundation',
    'MaterialPhysics',
    'JohnsonCookParameters',
    'sfdp_taylor_coefficient_processor',
    'TaylorResults',
    'TaylorCoefficients',
    # Future exports will be added here
]

__version__ = 'v17.3'