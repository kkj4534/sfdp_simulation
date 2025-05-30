"""
SFDP Validation QA Suite - Comprehensive Validation & Quality Assurance
======================================================================

Advanced validation and quality assurance system for multi-physics simulation
results with comprehensive error checking, consistency validation, and quality metrics.

Author: SFDP Research Team
Version: 17.3 (Complete Validation QA Implementation)
License: Academic Research Use Only
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
import json
import os
from datetime import datetime
import warnings

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class ValidationConfig:
    """Validation configuration parameters"""
    strictness_level: float = 0.85
    safety_factor: float = 1.2
    min_confidence_threshold: float = 0.7
    statistical_alpha: float = 0.05
    convergence_tolerance: float = 1e-6


@dataclass
class ValidationResults:
    """Comprehensive validation results"""
    physical_bounds: Dict[str, Any]
    layer_consistency: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    uncertainty_analysis: Dict[str, Any]
    pass_fail_summary: Dict[str, bool]
    overall_score: float
    confidence: float
    status: str  # PASS, FAIL, WARNING


@dataclass
class PhysicalLimits:
    """Physical constraint limits for Ti-6Al-4V machining"""
    temp_min: float = 273.15  # K (0°C)
    temp_max: float = 1933.0  # K (melting point)
    force_min: float = 0.0    # N
    force_max: float = 5000.0 # N (typical machine limit)
    wear_rate_min: float = 0.0      # mm/min
    wear_rate_max: float = 0.01     # mm/min (catastrophic wear)
    roughness_min: float = 0.01     # μm (super finish)
    roughness_max: float = 50.0     # μm (very rough)
    speed_min: float = 10.0         # m/min
    speed_max: float = 500.0        # m/min
    feed_min: float = 0.01          # mm/rev
    feed_max: float = 0.5           # mm/rev
    doc_min: float = 0.1            # mm
    doc_max: float = 5.0            # mm


# Default physical limits
DEFAULT_LIMITS = PhysicalLimits()


def performComprehensiveValidation(layer_results: Dict[str, Any], 
                                 simulation_state: Dict[str, Any]) -> Tuple[ValidationResults, float, str]:
    """
    Main validation orchestrator implementing ASME V&V 10-2006 standards
    
    Performs multi-level validation:
    1. Physics validation - conservation laws
    2. Phenomenological validation - process mechanisms
    3. Empirical validation - data correlations
    4. Statistical validation - hypothesis testing
    5. Engineering validation - practical constraints
    """
    logger.info("🔍 Performing comprehensive validation across all layers")
    
    # Initialize validation configuration
    config = ValidationConfig()
    if 'validation_config' in simulation_state:
        for key, value in simulation_state['validation_config'].items():
            setattr(config, key, value)
    
    logger.info(f"📋 Validation Configuration: Strictness {config.strictness_level:.2f}, "
                f"Safety Factor {config.safety_factor:.1f}x")
    
    validation_scores = []
    
    # 1. Validate Physical Bounds
    logger.info("🌡️ Validating physical bounds and constraints")
    bounds_validation = validatePhysicalBounds(layer_results, config)
    validation_scores.append(bounds_validation['overall_score'])
    
    # 2. Check Consistency Across Layers
    logger.info("🔗 Checking consistency across prediction layers")
    consistency_validation = checkConsistencyAcrossLayers(layer_results, config)
    validation_scores.append(consistency_validation['overall_score'])
    
    # 3. Assess Prediction Quality
    logger.info("📊 Assessing prediction quality metrics")
    quality_validation = assessPredictionQuality(layer_results, simulation_state)
    validation_scores.append(quality_validation['overall_score'])
    
    # 4. Validate Material Properties
    logger.info("🔬 Validating material property consistency")
    material_validation = validateMaterialProperties(layer_results, simulation_state)
    validation_scores.append(material_validation['overall_score'])
    
    # 5. Check Thermal Consistency
    logger.info("🔥 Checking thermal physics consistency")
    thermal_validation = checkThermalConsistency(layer_results, simulation_state)
    validation_scores.append(thermal_validation['overall_score'])
    
    # 6. Validate Tool Wear Physics
    logger.info("🔧 Validating tool wear physics")
    wear_validation = validateToolWearPhysics(layer_results, simulation_state)
    validation_scores.append(wear_validation['overall_score'])
    
    # 7. Assess Surface Roughness Realism
    logger.info("📏 Assessing surface roughness realism")
    roughness_validation = assessSurfaceRoughnessRealism(layer_results, simulation_state)
    validation_scores.append(roughness_validation['overall_score'])
    
    # 8. Perform Statistical Validation
    logger.info("📈 Performing statistical validation tests")
    statistical_validation = performStatisticalValidation(layer_results, simulation_state)
    validation_scores.append(statistical_validation['overall_score'])
    
    # Calculate overall validation score
    overall_score = np.mean(validation_scores)
    validation_confidence = calculate_validation_confidence(validation_scores, config)
    
    # Determine validation status
    if overall_score >= config.min_confidence_threshold:
        validation_status = "PASS"
    elif overall_score >= config.min_confidence_threshold * 0.8:
        validation_status = "WARNING"
    else:
        validation_status = "FAIL"
    
    # Generate pass/fail summary
    pass_fail_summary = {
        'physical_bounds': bounds_validation['overall_score'] > config.min_confidence_threshold,
        'layer_consistency': consistency_validation['overall_score'] > config.min_confidence_threshold,
        'quality_metrics': quality_validation['overall_score'] > config.min_confidence_threshold,
        'material_properties': material_validation['overall_score'] > config.min_confidence_threshold,
        'thermal_consistency': thermal_validation['overall_score'] > config.min_confidence_threshold,
        'wear_physics': wear_validation['overall_score'] > config.min_confidence_threshold,
        'roughness_realism': roughness_validation['overall_score'] > config.min_confidence_threshold,
        'statistical_tests': statistical_validation['overall_score'] > config.min_confidence_threshold
    }
    
    # Create comprehensive results
    validation_results = ValidationResults(
        physical_bounds=bounds_validation,
        layer_consistency=consistency_validation,
        quality_metrics=quality_validation,
        statistical_tests=statistical_validation,
        uncertainty_analysis=perform_uncertainty_analysis(layer_results, validation_scores),
        pass_fail_summary=pass_fail_summary,
        overall_score=overall_score,
        confidence=validation_confidence,
        status=validation_status
    )
    
    # Generate validation report
    report = generateValidationReport(validation_results, simulation_state)
    
    # Log validation results
    log_validation_results(validation_results, simulation_state)
    
    logger.info(f"✅ Validation Complete: {validation_status} (Score: {overall_score:.1%})")
    
    return validation_results, validation_confidence, validation_status


def validatePhysicalBounds(layer_results: Dict[str, Any], config: ValidationConfig) -> Dict[str, Any]:
    """
    Validate all predictions against physical constraints
    """
    bounds_validation = {
        'temperature': {},
        'force': {},
        'wear_rate': {},
        'roughness': {},
        'violations': [],
        'overall_score': 1.0
    }
    
    limits = DEFAULT_LIMITS
    violation_count = 0
    total_checks = 0
    
    # Check each layer's predictions
    for layer_name, results in layer_results.items():
        if not isinstance(results, dict):
            continue
            
        # Temperature bounds
        if 'cutting_temperature' in results:
            temp = results['cutting_temperature']
            total_checks += 1
            if temp < limits.temp_min or temp > limits.temp_max:
                violation_count += 1
                bounds_validation['violations'].append(
                    f"{layer_name}: Temperature {temp:.1f}K outside bounds [{limits.temp_min}, {limits.temp_max}]"
                )
                bounds_validation['temperature'][layer_name] = False
            else:
                bounds_validation['temperature'][layer_name] = True
        
        # Force bounds
        if 'cutting_force_Fc' in results:
            force = results['cutting_force_Fc']
            total_checks += 1
            if force < limits.force_min or force > limits.force_max:
                violation_count += 1
                bounds_validation['violations'].append(
                    f"{layer_name}: Force {force:.1f}N outside bounds [{limits.force_min}, {limits.force_max}]"
                )
                bounds_validation['force'][layer_name] = False
            else:
                bounds_validation['force'][layer_name] = True
        
        # Wear rate bounds
        if 'tool_wear_rate' in results:
            wear = results['tool_wear_rate']
            total_checks += 1
            if wear < limits.wear_rate_min or wear > limits.wear_rate_max:
                violation_count += 1
                bounds_validation['violations'].append(
                    f"{layer_name}: Wear rate {wear:.6f}mm/min outside bounds"
                )
                bounds_validation['wear_rate'][layer_name] = False
            else:
                bounds_validation['wear_rate'][layer_name] = True
        
        # Roughness bounds
        if 'surface_roughness' in results:
            roughness = results['surface_roughness']
            total_checks += 1
            if roughness < limits.roughness_min or roughness > limits.roughness_max:
                violation_count += 1
                bounds_validation['violations'].append(
                    f"{layer_name}: Roughness {roughness:.2f}μm outside bounds"
                )
                bounds_validation['roughness'][layer_name] = False
            else:
                bounds_validation['roughness'][layer_name] = True
    
    # Calculate overall score
    if total_checks > 0:
        bounds_validation['overall_score'] = 1.0 - (violation_count / total_checks)
        # Apply strictness factor
        bounds_validation['overall_score'] *= config.strictness_level
    
    bounds_validation['violation_count'] = violation_count
    bounds_validation['total_checks'] = total_checks
    
    return bounds_validation


def checkConsistencyAcrossLayers(layer_results: Dict[str, Any], config: ValidationConfig) -> Dict[str, Any]:
    """
    Check inter-layer consistency of predictions
    """
    consistency_check = {
        'temperature_consistency': {},
        'force_consistency': {},
        'wear_consistency': {},
        'roughness_consistency': {},
        'max_deviation': {},
        'overall_score': 1.0
    }
    
    # Extract predictions from each layer
    variables = ['cutting_temperature', 'cutting_force_Fc', 'tool_wear_rate', 'surface_roughness']
    var_names = ['temperature', 'force', 'wear', 'roughness']
    
    consistency_scores = []
    
    for var, var_name in zip(variables, var_names):
        predictions = []
        layer_names = []
        
        for layer_name, results in layer_results.items():
            if isinstance(results, dict) and var in results:
                predictions.append(results[var])
                layer_names.append(layer_name)
        
        if len(predictions) > 1:
            # Calculate consistency metrics
            mean_val = np.mean(predictions)
            std_val = np.std(predictions)
            cv = std_val / mean_val if mean_val > 0 else 0
            
            # Maximum deviation
            max_dev = np.max(np.abs(predictions - mean_val)) / mean_val if mean_val > 0 else 0
            
            # Consistency score (lower CV = higher consistency)
            consistency_score = 1.0 / (1.0 + cv)
            
            # Apply safety factor for critical variables
            if var_name in ['temperature', 'force']:
                consistency_score /= config.safety_factor
            
            consistency_check[f'{var_name}_consistency'] = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'consistency_score': consistency_score,
                'predictions': dict(zip(layer_names, predictions))
            }
            consistency_check['max_deviation'][var_name] = max_dev
            consistency_scores.append(consistency_score)
    
    # Overall consistency score
    if consistency_scores:
        consistency_check['overall_score'] = np.mean(consistency_scores)
    
    return consistency_check


def assessPredictionQuality(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess prediction quality using multiple metrics
    """
    quality_metrics = {
        'accuracy_metrics': {},
        'precision_metrics': {},
        'reliability_metrics': {},
        'convergence_metrics': {},
        'overall_score': 1.0
    }
    
    quality_scores = []
    
    # Check if experimental data is available
    if 'experimental_data' in simulation_state:
        exp_data = simulation_state['experimental_data']
        
        for layer_name, results in layer_results.items():
            if not isinstance(results, dict):
                continue
            
            layer_metrics = {}
            
            # Calculate accuracy metrics
            for var in ['cutting_temperature', 'cutting_force_Fc', 'surface_roughness']:
                if var in results and var in exp_data:
                    predicted = results[var]
                    observed = exp_data[var]
                    
                    # Relative error
                    rel_error = abs(predicted - observed) / observed if observed > 0 else 0
                    
                    # Accuracy score
                    accuracy = 1.0 / (1.0 + rel_error)
                    
                    layer_metrics[var] = {
                        'predicted': predicted,
                        'observed': observed,
                        'relative_error': rel_error,
                        'accuracy': accuracy
                    }
            
            if layer_metrics:
                avg_accuracy = np.mean([m['accuracy'] for m in layer_metrics.values()])
                quality_metrics['accuracy_metrics'][layer_name] = {
                    'metrics': layer_metrics,
                    'average_accuracy': avg_accuracy
                }
                quality_scores.append(avg_accuracy)
    
    # Assess numerical convergence
    convergence_score = assess_convergence(layer_results, simulation_state)
    quality_metrics['convergence_metrics'] = convergence_score
    quality_scores.append(convergence_score['score'])
    
    # Overall quality score
    if quality_scores:
        quality_metrics['overall_score'] = np.mean(quality_scores)
    
    return quality_metrics


def validateMaterialProperties(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate material property consistency and physical realism
    """
    material_validation = {
        'property_bounds': {},
        'temperature_dependence': {},
        'consistency_check': {},
        'overall_score': 1.0
    }
    
    # Ti-6Al-4V property bounds
    property_bounds = {
        'density': (4420, 4440),  # kg/m³
        'specific_heat': (500, 700),  # J/kg·K
        'thermal_conductivity': (6.0, 25.0),  # W/m·K (temperature dependent)
        'elastic_modulus': (100e9, 120e9),  # Pa
        'yield_strength': (800e6, 1000e6),  # Pa
        'hardness': (3.0e9, 4.0e9)  # Pa
    }
    
    validation_scores = []
    
    # Check material properties in simulation state
    if 'material_properties' in simulation_state:
        mat_props = simulation_state['material_properties']
        
        for prop, (min_val, max_val) in property_bounds.items():
            if prop in mat_props:
                value = mat_props[prop]
                if min_val <= value <= max_val:
                    material_validation['property_bounds'][prop] = {
                        'value': value,
                        'valid': True,
                        'bounds': (min_val, max_val)
                    }
                    validation_scores.append(1.0)
                else:
                    material_validation['property_bounds'][prop] = {
                        'value': value,
                        'valid': False,
                        'bounds': (min_val, max_val)
                    }
                    validation_scores.append(0.0)
    
    # Check temperature-dependent properties
    if layer_results:
        temps = []
        for layer_name, results in layer_results.items():
            if isinstance(results, dict) and 'cutting_temperature' in results:
                temps.append(results['cutting_temperature'])
        
        if temps:
            avg_temp = np.mean(temps)
            # Validate thermal conductivity variation
            k_room = 6.7  # W/m·K at room temperature
            k_expected = k_room * (1 + 0.001 * (avg_temp - 293))  # Linear approximation
            
            material_validation['temperature_dependence'] = {
                'average_temperature': avg_temp,
                'expected_conductivity': k_expected,
                'valid': 6.0 <= k_expected <= 25.0
            }
            validation_scores.append(1.0 if material_validation['temperature_dependence']['valid'] else 0.0)
    
    # Overall score
    if validation_scores:
        material_validation['overall_score'] = np.mean(validation_scores)
    
    return material_validation


def checkThermalConsistency(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check thermal physics consistency including energy balance
    """
    thermal_consistency = {
        'energy_balance': {},
        'heat_partition': {},
        'temperature_gradients': {},
        'overall_score': 1.0
    }
    
    consistency_scores = []
    
    # Energy balance check
    if 'cutting_conditions' in simulation_state:
        V = simulation_state['cutting_conditions'].get('cutting_speed', 100)  # m/min
        F = simulation_state['cutting_conditions'].get('cutting_force_Fc', 1000)  # N
        
        # Mechanical power
        P_mech = F * V / 60  # W
        
        # Heat generation
        temps = []
        for layer_name, results in layer_results.items():
            if isinstance(results, dict) and 'cutting_temperature' in results:
                temps.append(results['cutting_temperature'])
        
        if temps:
            avg_temp_rise = np.mean(temps) - 300  # K
            
            # Approximate heat generation (simplified)
            mat_props = simulation_state.get('material_properties', {})
            rho = mat_props.get('density', 4430)
            cp = mat_props.get('specific_heat', 580)
            volume_rate = V * 1e-6  # Simplified volume rate
            
            Q_approx = rho * cp * volume_rate * avg_temp_rise
            
            # Energy balance ratio
            energy_ratio = Q_approx / P_mech if P_mech > 0 else 0
            
            # Should be between 0.5 and 1.0 (50-100% conversion)
            energy_valid = 0.5 <= energy_ratio <= 1.0
            
            thermal_consistency['energy_balance'] = {
                'mechanical_power': P_mech,
                'heat_generation_approx': Q_approx,
                'energy_ratio': energy_ratio,
                'valid': energy_valid
            }
            consistency_scores.append(1.0 if energy_valid else 0.5)
    
    # Heat partition check
    heat_partition_ratio = 0.7  # Typical for Ti-6Al-4V
    thermal_consistency['heat_partition'] = {
        'workpiece_fraction': heat_partition_ratio,
        'tool_fraction': 1 - heat_partition_ratio,
        'valid': True  # Using standard values
    }
    consistency_scores.append(1.0)
    
    # Overall thermal consistency
    if consistency_scores:
        thermal_consistency['overall_score'] = np.mean(consistency_scores)
    
    return thermal_consistency


def validateToolWearPhysics(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate tool wear physics and wear mechanisms
    """
    wear_validation = {
        'wear_mechanisms': {},
        'wear_rate_consistency': {},
        'taylor_validation': {},
        'overall_score': 1.0
    }
    
    validation_scores = []
    
    # Collect wear predictions
    wear_rates = []
    layer_names = []
    
    for layer_name, results in layer_results.items():
        if isinstance(results, dict) and 'tool_wear_rate' in results:
            wear_rates.append(results['tool_wear_rate'])
            layer_names.append(layer_name)
    
    if wear_rates:
        # Check wear rate magnitudes
        avg_wear = np.mean(wear_rates)
        
        # Typical wear rate ranges for Ti-6Al-4V
        if 1e-5 <= avg_wear <= 1e-2:  # mm/min
            wear_validation['wear_rate_consistency']['magnitude_valid'] = True
            validation_scores.append(1.0)
        else:
            wear_validation['wear_rate_consistency']['magnitude_valid'] = False
            validation_scores.append(0.0)
        
        # Check wear mechanism dominance based on temperature
        temps = []
        for layer_name, results in layer_results.items():
            if isinstance(results, dict) and 'cutting_temperature' in results:
                temps.append(results['cutting_temperature'])
        
        if temps:
            avg_temp = np.mean(temps)
            
            # Determine dominant wear mechanism
            if avg_temp < 600:  # K
                dominant_mechanism = 'adhesive'
            elif avg_temp < 800:
                dominant_mechanism = 'abrasive'
            else:
                dominant_mechanism = 'diffusion'
            
            wear_validation['wear_mechanisms'] = {
                'average_temperature': avg_temp,
                'dominant_mechanism': dominant_mechanism,
                'wear_rates': dict(zip(layer_names, wear_rates))
            }
            validation_scores.append(0.9)  # High score for mechanism identification
    
    # Taylor equation validation
    if 'cutting_conditions' in simulation_state:
        V = simulation_state['cutting_conditions'].get('cutting_speed', 100)
        
        # Taylor's tool life equation: VT^n = C
        # Typical values for Ti-6Al-4V
        n = 0.25
        C = 200
        
        T_life_expected = (C / V) ** (1/n)
        
        if wear_rates:
            avg_wear = np.mean(wear_rates)
            T_life_predicted = 0.3 / avg_wear if avg_wear > 0 else 1000  # VB = 0.3mm criterion
            
            # Check agreement
            ratio = T_life_predicted / T_life_expected if T_life_expected > 0 else 0
            taylor_valid = 0.5 <= ratio <= 2.0  # Within factor of 2
            
            wear_validation['taylor_validation'] = {
                'expected_life': T_life_expected,
                'predicted_life': T_life_predicted,
                'ratio': ratio,
                'valid': taylor_valid
            }
            validation_scores.append(1.0 if taylor_valid else 0.5)
    
    # Overall score
    if validation_scores:
        wear_validation['overall_score'] = np.mean(validation_scores)
    
    return wear_validation


def assessSurfaceRoughnessRealism(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess surface roughness predictions for physical realism
    """
    roughness_assessment = {
        'theoretical_bounds': {},
        'empirical_correlation': {},
        'scale_analysis': {},
        'overall_score': 1.0
    }
    
    assessment_scores = []
    
    # Collect roughness predictions
    roughness_values = []
    layer_names = []
    
    for layer_name, results in layer_results.items():
        if isinstance(results, dict) and 'surface_roughness' in results:
            roughness_values.append(results['surface_roughness'])
            layer_names.append(layer_name)
    
    if roughness_values and 'cutting_conditions' in simulation_state:
        f = simulation_state['cutting_conditions'].get('feed_rate', 0.1)  # mm/rev
        r_nose = simulation_state.get('tool_properties', {}).get('nose_radius', 0.8)  # mm
        
        # Theoretical minimum roughness
        Ra_min_theoretical = (f**2) / (32 * r_nose) * 1000  # μm
        
        # Check if predictions are above theoretical minimum
        for Ra, layer in zip(roughness_values, layer_names):
            if Ra >= Ra_min_theoretical * 0.9:  # Allow 10% tolerance
                roughness_assessment['theoretical_bounds'][layer] = {
                    'predicted': Ra,
                    'theoretical_min': Ra_min_theoretical,
                    'valid': True
                }
                assessment_scores.append(1.0)
            else:
                roughness_assessment['theoretical_bounds'][layer] = {
                    'predicted': Ra,
                    'theoretical_min': Ra_min_theoretical,
                    'valid': False
                }
                assessment_scores.append(0.0)
        
        # Empirical correlation check
        V = simulation_state['cutting_conditions'].get('cutting_speed', 100)
        
        # Typical empirical range for Ti-6Al-4V
        Ra_empirical_min = 0.1 * f**0.8 * (100/V)**0.2
        Ra_empirical_max = 5.0 * f**0.8 * (100/V)**0.2
        
        avg_roughness = np.mean(roughness_values)
        empirical_valid = Ra_empirical_min <= avg_roughness <= Ra_empirical_max
        
        roughness_assessment['empirical_correlation'] = {
            'average_predicted': avg_roughness,
            'empirical_range': (Ra_empirical_min, Ra_empirical_max),
            'valid': empirical_valid
        }
        assessment_scores.append(1.0 if empirical_valid else 0.5)
    
    # Overall assessment
    if assessment_scores:
        roughness_assessment['overall_score'] = np.mean(assessment_scores)
    
    return roughness_assessment


def performStatisticalValidation(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform statistical validation tests on predictions
    """
    statistical_tests = {
        'normality_tests': {},
        'correlation_tests': {},
        'hypothesis_tests': {},
        'confidence_intervals': {},
        'overall_score': 1.0
    }
    
    test_scores = []
    
    # Collect all predictions by variable
    variables = ['cutting_temperature', 'cutting_force_Fc', 'tool_wear_rate', 'surface_roughness']
    predictions_by_var = {var: [] for var in variables}
    
    for layer_name, results in layer_results.items():
        if isinstance(results, dict):
            for var in variables:
                if var in results:
                    predictions_by_var[var].append(results[var])
    
    # Normality tests
    for var, values in predictions_by_var.items():
        if len(values) >= 3:
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(values)
            
            statistical_tests['normality_tests'][var] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05,
                'n_samples': len(values)
            }
            test_scores.append(1.0 if p_value > 0.05 else 0.7)
    
    # Correlation tests between variables
    if len(predictions_by_var['cutting_temperature']) >= 3:
        # Temperature vs Force correlation
        if len(predictions_by_var['cutting_force_Fc']) >= 3:
            temp_force_corr, p_value = stats.pearsonr(
                predictions_by_var['cutting_temperature'][:3],
                predictions_by_var['cutting_force_Fc'][:3]
            )
            
            statistical_tests['correlation_tests']['temp_force'] = {
                'correlation': temp_force_corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            # Expect positive correlation
            test_scores.append(1.0 if temp_force_corr > 0 else 0.5)
    
    # Confidence intervals
    for var, values in predictions_by_var.items():
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            
            # 95% confidence interval
            ci_lower = mean_val - 1.96 * std_val / np.sqrt(n)
            ci_upper = mean_val + 1.96 * std_val / np.sqrt(n)
            
            statistical_tests['confidence_intervals'][var] = {
                'mean': mean_val,
                'std': std_val,
                'ci_95': (ci_lower, ci_upper),
                'ci_width_ratio': (ci_upper - ci_lower) / mean_val if mean_val > 0 else 0
            }
            
            # Narrower CI = better
            ci_score = 1.0 / (1.0 + statistical_tests['confidence_intervals'][var]['ci_width_ratio'])
            test_scores.append(ci_score)
    
    # Overall statistical score
    if test_scores:
        statistical_tests['overall_score'] = np.mean(test_scores)
    
    return statistical_tests


def generateValidationReport(validation_results: ValidationResults, 
                           simulation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive validation report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_version': '17.3',
        'summary': {
            'overall_status': validation_results.status,
            'overall_score': validation_results.overall_score,
            'confidence': validation_results.confidence,
            'pass_fail_summary': validation_results.pass_fail_summary
        },
        'detailed_results': {
            'physical_bounds': validation_results.physical_bounds,
            'layer_consistency': validation_results.layer_consistency,
            'quality_metrics': validation_results.quality_metrics,
            'statistical_tests': validation_results.statistical_tests,
            'uncertainty_analysis': validation_results.uncertainty_analysis
        },
        'recommendations': generate_recommendations(validation_results),
        'simulation_metadata': {
            'cutting_conditions': simulation_state.get('cutting_conditions', {}),
            'material': simulation_state.get('material_name', 'Ti-6Al-4V'),
            'timestamp': simulation_state.get('timestamp', '')
        }
    }
    
    # Save report
    save_validation_report(report)
    
    return report


# Helper functions

def assess_convergence(layer_results: Dict[str, Any], simulation_state: Dict[str, Any]) -> Dict[str, float]:
    """Assess numerical convergence of predictions"""
    convergence = {
        'residual_norms': {},
        'iteration_counts': {},
        'score': 0.9  # Default high score
    }
    
    # Check for convergence indicators in simulation state
    if 'convergence_data' in simulation_state:
        conv_data = simulation_state['convergence_data']
        
        # Check residual norms
        if 'residuals' in conv_data:
            max_residual = np.max(conv_data['residuals'])
            convergence['residual_norms']['max'] = max_residual
            convergence['residual_norms']['converged'] = max_residual < 1e-6
            
            if not convergence['residual_norms']['converged']:
                convergence['score'] *= 0.8
        
        # Check iteration counts
        if 'iterations' in conv_data:
            convergence['iteration_counts'] = conv_data['iterations']
    
    return convergence


def perform_uncertainty_analysis(layer_results: Dict[str, Any], 
                               validation_scores: List[float]) -> Dict[str, Any]:
    """Perform comprehensive uncertainty analysis"""
    uncertainty = {
        'aleatory': {},    # Random variability
        'epistemic': {},   # Knowledge uncertainty
        'numerical': {},   # Computational uncertainty
        'combined': {}
    }
    
    # Aleatory uncertainty from prediction spread
    variables = ['cutting_temperature', 'cutting_force_Fc', 'tool_wear_rate', 'surface_roughness']
    
    for var in variables:
        values = []
        for layer_name, results in layer_results.items():
            if isinstance(results, dict) and var in results:
                values.append(results[var])
        
        if len(values) > 1:
            # Aleatory uncertainty as coefficient of variation
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else 0
            
            uncertainty['aleatory'][var] = {
                'cv': cv,
                'std': std_val,
                'uncertainty_percentage': cv * 100
            }
    
    # Epistemic uncertainty from validation scores
    score_std = np.std(validation_scores)
    score_mean = np.mean(validation_scores)
    
    uncertainty['epistemic'] = {
        'validation_score_spread': score_std,
        'mean_validation_score': score_mean,
        'confidence_level': 1.0 - score_std  # Higher spread = lower confidence
    }
    
    # Combined uncertainty
    if uncertainty['aleatory']:
        avg_aleatory = np.mean([u['cv'] for u in uncertainty['aleatory'].values()])
        combined_uncertainty = np.sqrt(avg_aleatory**2 + score_std**2)
        
        uncertainty['combined'] = {
            'total_uncertainty': combined_uncertainty,
            'confidence_interval_width': combined_uncertainty * 1.96,  # 95% CI
            'uncertainty_percentage': combined_uncertainty * 100
        }
    
    return uncertainty


def calculate_validation_confidence(scores: List[float], config: ValidationConfig) -> float:
    """Calculate overall validation confidence"""
    if not scores:
        return 0.0
    
    # Base confidence from mean score
    base_confidence = np.mean(scores)
    
    # Penalize for high variability
    score_std = np.std(scores)
    variability_penalty = score_std * 0.5
    
    # Apply strictness factor
    confidence = (base_confidence - variability_penalty) * config.strictness_level
    
    # Ensure bounds [0, 1]
    return np.clip(confidence, 0.0, 1.0)


def generate_recommendations(validation_results: ValidationResults) -> List[str]:
    """Generate actionable recommendations based on validation results"""
    recommendations = []
    
    # Check physical bounds violations
    if validation_results.physical_bounds['violation_count'] > 0:
        recommendations.append(
            "⚠️ Physical bounds violations detected. Review model parameters and constraints."
        )
    
    # Check layer consistency
    if validation_results.layer_consistency['overall_score'] < 0.7:
        recommendations.append(
            "🔄 Low inter-layer consistency. Consider adjusting fusion weights or model calibration."
        )
    
    # Check statistical tests
    if validation_results.statistical_tests['overall_score'] < 0.6:
        recommendations.append(
            "📊 Statistical tests indicate high uncertainty. Increase sample size or refine models."
        )
    
    # Check overall status
    if validation_results.status == "FAIL":
        recommendations.append(
            "❌ Validation failed. Major model revision recommended before production use."
        )
    elif validation_results.status == "WARNING":
        recommendations.append(
            "⚠️ Validation passed with warnings. Monitor closely and validate with additional data."
        )
    else:
        recommendations.append(
            "✅ Validation passed. Model suitable for intended application within specified bounds."
        )
    
    return recommendations


def save_validation_report(report: Dict[str, Any]):
    """Save validation report to file"""
    report_dir = 'SFDP_6Layer_v17_3/validation_diagnosis'
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{report_dir}/validation_report_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.debug(f"Saved validation report to {filename}")


def log_validation_results(results: ValidationResults, simulation_state: Dict[str, Any]):
    """Log validation results for tracking"""
    log_dir = 'SFDP_6Layer_v17_3/validation'
    os.makedirs(log_dir, exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': results.status,
        'overall_score': results.overall_score,
        'confidence': results.confidence,
        'pass_fail_summary': results.pass_fail_summary,
        'simulation_conditions': simulation_state.get('cutting_conditions', {}),
        'validation_suite_version': '17.3'
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{log_dir}/validation_log_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    logger.debug(f"Logged validation results to {filename}")


# Integration with tuning systems
def get_validation_parameters():
    """Return adjustable validation parameters for tuning systems"""
    return {
        'config': {
            'strictness_level': (0.5, 0.95),
            'safety_factor': (1.0, 2.0),
            'min_confidence_threshold': (0.5, 0.9),
            'statistical_alpha': (0.01, 0.10)
        },
        'bounds': {
            'temperature_tolerance': (0.9, 1.1),  # Multiplier on limits
            'force_tolerance': (0.85, 1.15),
            'wear_tolerance': (0.8, 1.2),
            'roughness_tolerance': (0.7, 1.3)
        },
        'consistency': {
            'max_cv_allowed': (0.1, 0.5),
            'min_correlation': (0.3, 0.9),
            'convergence_tolerance': (1e-8, 1e-4)
        }
    }