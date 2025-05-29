"""
SFDP_TAYLOR_COEFFICIENT_PROCESSOR - Data-Based Taylor Coefficient Processing
=========================================================================
INTELLIGENT TAYLOR COEFFICIENT SELECTION FROM EXPERIMENTAL DATABASE

THEORETICAL FOUNDATION:
Enhanced Taylor tool life equation with multi-variable dependencies:
V × T^n × f^a × d^b × Q^c = C_extended
where: V=cutting speed, T=tool life, f=feed rate, d=depth, Q=material hardness

DATABASE SELECTION CRITERIA:
1. Material match (Ti-6Al-4V priority)
2. Tool material compatibility
3. Cutting condition similarity
4. Data reliability rating (⭐⭐⭐⭐⭐ = highest)
5. Experimental validation coverage

COEFFICIENT ADAPTATION:
- Temperature dependence: C_eff = C_base × exp(-E_a/(RT))
- Tool material correction: C_tool = C_base × k_material
- Workpiece condition factor: C_condition = C_base × k_condition

REFERENCE: Taylor (1907) "On the art of cutting metals" Trans. ASME 28
REFERENCE: Kronenberg (1966) "Machining Science and Application" Pergamon
REFERENCE: ASM Handbook Vol. 16 (1989) "Machining" Ch. 5
REFERENCE: Trent & Wright (2000) "Metal Cutting" 4th Ed. Ch. 10

Author: SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr)
Date: May 2025
=========================================================================
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TaylorCoefficients:
    """Taylor equation coefficients"""
    C: float = 180.0      # Taylor constant (m/min)
    n: float = 0.25       # Taylor exponent
    a: float = 0.75       # Feed rate exponent
    b: float = 0.15       # Depth of cut exponent
    c: float = 0.5        # Material hardness exponent


@dataclass
class ReferenceConditions:
    """Reference conditions for Taylor coefficients"""
    material: str = 'Ti-6Al-4V'
    tool: str = 'Carbide_Uncoated'
    cutting_speed_range: Tuple[float, float] = (50.0, 300.0)  # m/min
    feed_range: Tuple[float, float] = (0.1, 0.5)  # mm/rev
    depth_range: Tuple[float, float] = (0.5, 5.0)  # mm


@dataclass
class TaylorResults:
    """Complete Taylor coefficient results"""
    coefficients: TaylorCoefficients = field(default_factory=TaylorCoefficients)
    selection_method: str = 'DATABASE_MATCH'
    material_match: str = ''
    tool_match: str = ''
    confidence: float = 0.0
    data_source: str = 'extended_materials_database'
    source: str = ''
    reliability: str = '⭐⭐⭐☆☆'
    adaptation_applied: bool = False
    base_source: str = ''
    reference_conditions: ReferenceConditions = field(default_factory=ReferenceConditions)
    validation_applied: bool = False


def calculate_match_confidence(material_data: Dict[str, Any], tool_data: Dict[str, Any]) -> float:
    """Calculate confidence score for material-tool match"""
    confidence = 0.5  # Base confidence
    
    # Check data quality indicators
    if 'reliability_rating' in material_data:
        rating = material_data['reliability_rating']
        if rating == '⭐⭐⭐⭐⭐':
            confidence += 0.4
        elif rating == '⭐⭐⭐⭐☆':
            confidence += 0.3
        elif rating == '⭐⭐⭐☆☆':
            confidence += 0.2
        else:
            confidence += 0.1
    
    # Check if experimental validation is available
    if tool_data.get('experimental_validation', False):
        confidence += 0.1
    
    # Cap at 1.0
    return min(confidence, 1.0)


def prepare_match_data(material_data: Dict[str, Any], tool_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare matched data structure"""
    match_data = {
        'source': f"{material_data.get('name', 'Unknown')} + {tool_data.get('material', 'Unknown')}",
        'coefficients': tool_data.get('coefficients', {}),
        'material_properties': material_data,
        'tool_properties': tool_data,
        'reliability': material_data.get('reliability_rating', '⭐⭐⭐☆☆')
    }
    return match_data


def find_best_taylor_match(
    extended_data: Dict[str, Any], 
    material_name: str, 
    tool_material: str
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Find best matching Taylor coefficients from database"""
    best_match = None
    confidence = 0.0
    
    # Check if materials data exists
    if 'materials_data' not in extended_data:
        return None, 0.0
    
    materials = extended_data['materials_data']
    
    # Search for exact material match first
    for material_entry in materials:
        if isinstance(material_entry, dict) and 'name' in material_entry and 'taylor_data' in material_entry:
            # Check material name match
            material_match = (
                material_name.lower() in material_entry['name'].lower() or
                material_entry['name'].lower() in material_name.lower()
            )
            
            if material_match and 'tool_materials' in material_entry['taylor_data']:
                # Check tool material compatibility
                tool_data_list = material_entry['taylor_data']['tool_materials']
                
                for tool_data in tool_data_list:
                    if isinstance(tool_data, dict) and 'material' in tool_data and 'coefficients' in tool_data:
                        tool_match = (
                            tool_material.lower() in tool_data['material'].lower() or
                            tool_data['material'].lower() in tool_material.lower()
                        )
                        
                        if tool_match:
                            current_confidence = calculate_match_confidence(material_entry, tool_data)
                            
                            if current_confidence > confidence:
                                best_match = prepare_match_data(material_entry, tool_data)
                                confidence = current_confidence
    
    # If no exact match, try partial matches for titanium alloys
    if best_match is None and 'ti' in material_name.lower():
        for material_entry in materials:
            if (isinstance(material_entry, dict) and 
                'name' in material_entry and 
                'ti-6al-4v' in material_entry['name'].lower() and
                'taylor_data' in material_entry):
                
                tool_data_list = material_entry['taylor_data'].get('tool_materials', [])
                
                for tool_data in tool_data_list:
                    if isinstance(tool_data, dict) and 'coefficients' in tool_data:
                        current_confidence = 0.7  # Reduced confidence for fallback
                        
                        if current_confidence > confidence:
                            best_match = prepare_match_data(material_entry, tool_data)
                            confidence = current_confidence
    
    return best_match, confidence


def adapt_taylor_coefficients(best_match: Dict[str, Any], simulation_state: Any) -> TaylorResults:
    """Adapt Taylor coefficients based on operating conditions"""
    taylor_results = TaylorResults()
    
    # Extract base coefficients
    base_coeffs = best_match.get('coefficients', {})
    
    # Basic Taylor coefficients
    taylor_results.coefficients.C = base_coeffs.get('C', 180.0)
    taylor_results.coefficients.n = base_coeffs.get('n', 0.25)
    
    # Extended coefficients (if available)
    taylor_results.coefficients.a = base_coeffs.get('a', 0.75)  # Feed rate exponent
    taylor_results.coefficients.b = base_coeffs.get('b', 0.15)  # Depth exponent
    taylor_results.coefficients.c = base_coeffs.get('c', 0.5)   # Hardness exponent
    
    # Temperature adaptation (Arrhenius-type)
    if hasattr(simulation_state, 'operating_temperature'):
        T_ref = 298  # K (reference temperature)
        T_op = simulation_state.operating_temperature + 273.15  # K
        activation_energy = 50000  # J/mol (typical for tool wear)
        R = 8.314  # J/mol⋅K
        
        temp_factor = np.exp(activation_energy/R * (1/T_ref - 1/T_op))
        taylor_results.coefficients.C *= temp_factor
    
    # Add metadata
    taylor_results.adaptation_applied = True
    taylor_results.base_source = best_match['source']
    taylor_results.reliability = best_match['reliability']
    
    return taylor_results


def get_default_taylor_coefficients() -> TaylorResults:
    """Get default Taylor coefficients for Ti-6Al-4V with carbide tools"""
    taylor_results = TaylorResults()
    
    # Default coefficients
    taylor_results.coefficients = TaylorCoefficients(
        C=180.0,  # Taylor constant (m/min)
        n=0.25,   # Taylor exponent
        a=0.75,   # Feed rate exponent
        b=0.15,   # Depth of cut exponent
        c=0.5     # Material hardness exponent
    )
    
    taylor_results.source = 'DEFAULT_TI6AL4V_CARBIDE'
    taylor_results.reliability = '⭐⭐⭐☆☆'
    taylor_results.adaptation_applied = False
    
    # Reference conditions
    taylor_results.reference_conditions = ReferenceConditions()
    
    return taylor_results


def validate_taylor_ranges(taylor_results: TaylorResults) -> TaylorResults:
    """Validate Taylor coefficients are within reasonable ranges"""
    
    # Taylor constant C validation
    if taylor_results.coefficients.C < 50 or taylor_results.coefficients.C > 1000:
        print(f"    ⚠️  Taylor constant C={taylor_results.coefficients.C:.1f} "
              f"outside normal range [50-1000], adjusting")
        taylor_results.coefficients.C = max(50, min(1000, taylor_results.coefficients.C))
    
    # Taylor exponent n validation
    if taylor_results.coefficients.n < 0.1 or taylor_results.coefficients.n > 0.8:
        print(f"    ⚠️  Taylor exponent n={taylor_results.coefficients.n:.3f} "
              f"outside normal range [0.1-0.8], adjusting")
        taylor_results.coefficients.n = max(0.1, min(0.8, taylor_results.coefficients.n))
    
    # Feed rate exponent a validation
    if taylor_results.coefficients.a < 0.0 or taylor_results.coefficients.a > 1.5:
        taylor_results.coefficients.a = max(0.0, min(1.5, taylor_results.coefficients.a))
    
    # Depth exponent b validation
    if taylor_results.coefficients.b < 0.0 or taylor_results.coefficients.b > 0.5:
        taylor_results.coefficients.b = max(0.0, min(0.5, taylor_results.coefficients.b))
    
    taylor_results.validation_applied = True
    return taylor_results


def sfdp_taylor_coefficient_processor(
    simulation_state: Any, 
    extended_data: Dict[str, Any], 
    data_confidence: Dict[str, Any]
) -> Tuple[TaylorResults, float]:
    """
    Intelligent Taylor coefficient selection from experimental database
    
    Args:
        simulation_state: Global simulation configuration
        extended_data: Complete experimental database
        data_confidence: Data quality and reliability metrics
        
    Returns:
        taylor_results: Selected Taylor coefficients and metadata
        taylor_confidence: Confidence in coefficient selection [0-1]
    """
    print("  📊 Data-based Taylor coefficient selection...")
    
    # Check data availability and quality
    if 'materials' not in extended_data or extended_data['materials'].empty:
        print("    ⚠️  No materials data available, using default coefficients")
        taylor_results = get_default_taylor_coefficients()
        taylor_confidence = 0.6
        return taylor_results, taylor_confidence
    
    # Extract material and tool information
    # Using default values if not available in simulation_state
    material_name = getattr(simulation_state, 'material', {}).get('name', 'Ti6Al4V')
    tool_material = getattr(simulation_state, 'tool', {}).get('material', 'Carbide')
    
    print(f"    🔍 Searching coefficients for: {material_name} with {tool_material} tool")
    
    # Convert pandas DataFrame to dict format if needed
    materials_data = extended_data.get('materials', {})
    if hasattr(materials_data, 'to_dict'):
        materials_data = {'materials_data': materials_data.to_dict('records')}
    else:
        materials_data = {'materials_data': []}
    
    # Search for best matching data
    best_match, match_confidence = find_best_taylor_match(materials_data, material_name, tool_material)
    
    if best_match is not None:
        # Use database coefficients with adaptations
        taylor_results = adapt_taylor_coefficients(best_match, simulation_state)
        
        # Calculate overall confidence
        reliability_factor = data_confidence.get('materials', 0.8) if isinstance(data_confidence, dict) else 0.8
        taylor_confidence = match_confidence * reliability_factor
        
        print(f"    ✅ Found matching data: {best_match['source']}")
        print(f"       Taylor constant C = {taylor_results.coefficients.C:.1f}, "
              f"exponent n = {taylor_results.coefficients.n:.3f}")
    else:
        # Use default coefficients
        print("    ⚠️  No matching data found, using default Ti-6Al-4V coefficients")
        taylor_results = get_default_taylor_coefficients()
        taylor_confidence = 0.6
    
    # Update metadata
    taylor_results.material_match = material_name
    taylor_results.tool_match = tool_material
    taylor_results.confidence = taylor_confidence
    
    # Validate coefficient ranges
    taylor_results = validate_taylor_ranges(taylor_results)
    
    print(f"    📈 Final Taylor equation: V × T^{taylor_results.coefficients.n:.3f} = "
          f"{taylor_results.coefficients.C:.1f}")
    
    return taylor_results, taylor_confidence


# Export main function and classes
__all__ = ['sfdp_taylor_coefficient_processor', 'TaylorResults', 'TaylorCoefficients']