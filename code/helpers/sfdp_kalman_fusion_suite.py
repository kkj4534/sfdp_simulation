"""
SFDP Kalman Fusion Suite - Advanced State Estimation and Multi-Source Fusion
===========================================================================

Comprehensive Kalman filtering framework for intelligent fusion of multi-physics predictions.
Implements adaptive Kalman filters with variable-specific dynamics and uncertainty quantification.

Author: SFDP Research Team
Version: 17.3 (Complete Kalman Fusion Implementation)
License: Academic Research Use Only
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.linalg import inv, sqrtm
import json
import os
from datetime import datetime
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state representation"""
    estimate: float
    uncertainty: float
    prediction: float
    innovation: float
    gain: float
    timestamp: float


@dataclass
class KalmanResults:
    """Complete Kalman filtering results"""
    temperature: KalmanState
    wear_rate: KalmanState
    surface_roughness: KalmanState
    overall_confidence: float
    fusion_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    stability_status: Dict[str, bool]


@dataclass
class VariableDynamics:
    """Variable-specific dynamics parameters"""
    name: str
    process_noise: float
    measurement_noise: float
    correction_range: Tuple[float, float]
    time_constant: float
    adaptation_rate: float


# Variable-specific dynamics configurations
VARIABLE_DYNAMICS = {
    'temperature': VariableDynamics(
        name='temperature',
        process_noise=5.0,      # K
        measurement_noise=10.0, # K
        correction_range=(0.85, 1.15),  # ±15%
        time_constant=0.1,      # Thermal response time
        adaptation_rate=0.05    # Learning rate
    ),
    'wear_rate': VariableDynamics(
        name='wear_rate',
        process_noise=1e-5,     # mm/min
        measurement_noise=2e-5, # mm/min
        correction_range=(0.88, 1.12),  # ±12%
        time_constant=1.0,      # Wear evolution time
        adaptation_rate=0.03    # Slower adaptation
    ),
    'surface_roughness': VariableDynamics(
        name='surface_roughness',
        process_noise=0.1,      # μm
        measurement_noise=0.2,  # μm
        correction_range=(0.82, 1.18),  # ±18%
        time_constant=0.05,     # Fast surface changes
        adaptation_rate=0.08    # Faster adaptation
    )
}


def applyEnhancedAdaptiveKalman(layer_results: Dict[str, Any], 
                               simulation_state: Dict[str, Any]) -> Tuple[KalmanResults, float, Dict[str, float]]:
    """
    Apply enhanced adaptive Kalman filtering to fuse multi-layer predictions
    
    Based on optimal state estimation theory with adaptive gain calculation
    """
    logger.info("Applying enhanced adaptive Kalman filtering")
    
    # Initialize Kalman states
    kalman_states = {}
    kalman_gains = {}
    
    # Process each variable
    for var_name in ['temperature', 'wear_rate', 'surface_roughness']:
        # Extract predictions from different layers
        predictions = extract_layer_predictions(layer_results, var_name)
        
        # Get variable-specific dynamics
        var_dynamics = VARIABLE_DYNAMICS[var_name]
        
        # Calculate innovation sequence
        innovation = calculateInnovationSequence(predictions, var_name, simulation_state)
        
        # Determine adaptive Kalman gain
        optimal_gain, gain_confidence = determineAdaptiveKalmanGain(
            predictions, var_name, var_dynamics, simulation_state
        )
        
        # Perform Bayesian update
        updated_state, updated_uncertainty = performBayesianUpdate(
            predictions, innovation, optimal_gain, var_name, var_dynamics
        )
        
        # Store results
        kalman_states[var_name] = KalmanState(
            estimate=updated_state,
            uncertainty=updated_uncertainty,
            prediction=np.mean(list(predictions.values())),
            innovation=innovation,
            gain=optimal_gain
        )
        kalman_gains[var_name] = optimal_gain
    
    # Calculate fusion weights
    fusion_weights = calculateFusionWeights(layer_results, kalman_gains)
    
    # Validate performance
    performance_metrics = {}
    for var_name, state in kalman_states.items():
        metrics = validateKalmanPerformance(state, predictions, var_name)
        performance_metrics[var_name] = metrics
    
    # Monitor stability
    stability_status = {}
    for var_name, state in kalman_states.items():
        var_dynamics = VARIABLE_DYNAMICS[var_name]
        stability = monitorKalmanStability(state, var_dynamics)
        stability_status[var_name] = stability
    
    # Overall confidence
    kalman_confidence = calculate_overall_confidence(
        kalman_states, performance_metrics, stability_status
    )
    
    # Create results
    kalman_results = KalmanResults(
        temperature=kalman_states['temperature'],
        wear_rate=kalman_states['wear_rate'],
        surface_roughness=kalman_states['surface_roughness'],
        overall_confidence=kalman_confidence,
        fusion_weights=fusion_weights,
        performance_metrics=performance_metrics,
        stability_status=stability_status
    )
    
    # Log Kalman evolution
    log_kalman_evolution(kalman_states, kalman_gains, performance_metrics)
    
    return kalman_results, kalman_confidence, kalman_gains


def determineAdaptiveKalmanGain(predictions: Dict[str, float], variable_name: str,
                               dynamics: VariableDynamics, simulation_state: Dict[str, Any]) -> Tuple[float, float]:
    """
    Determine optimal Kalman gain based on innovation and uncertainty
    
    K(k) = P(k|k-1)H^T[HP(k|k-1)H^T + R]^(-1)
    """
    # Extract uncertainties from predictions
    uncertainties = extract_uncertainties(predictions, variable_name)
    
    # Prior uncertainty (from previous step or initialization)
    P_prior = simulation_state.get(f'{variable_name}_prior_uncertainty', 
                                  dynamics.process_noise**2)
    
    # Measurement noise covariance
    R = dynamics.measurement_noise**2
    
    # Observation matrix (simple scalar case)
    H = 1.0
    
    # Innovation covariance
    S = H * P_prior * H + R
    
    # Kalman gain
    K = P_prior * H / S
    
    # Adapt gain based on innovation history
    innovation_history = simulation_state.get(f'{variable_name}_innovation_history', [])
    if len(innovation_history) > 5:
        # Check innovation consistency
        innovation_variance = np.var(innovation_history[-10:])
        expected_variance = S
        
        # Adaptation factor
        if innovation_variance > 2 * expected_variance:
            # Innovations too large - increase gain
            K *= 1.2
        elif innovation_variance < 0.5 * expected_variance:
            # Innovations too small - decrease gain
            K *= 0.8
    
    # Bound gain to reasonable range
    K = np.clip(K, 0.1, 0.9)
    
    # Gain confidence based on innovation consistency
    if len(innovation_history) > 5:
        consistency = 1.0 / (1.0 + innovation_variance / expected_variance)
        gain_confidence = 0.5 + 0.5 * consistency
    else:
        gain_confidence = 0.7  # Initial confidence
    
    return K, gain_confidence


def calculateInnovationSequence(predictions: Dict[str, float], variable_name: str,
                               simulation_state: Dict[str, Any]) -> float:
    """
    Calculate innovation (prediction error) sequence
    
    ν(k) = z(k) - H·x(k|k-1)
    """
    # Current measurement (best available prediction)
    measurements = list(predictions.values())
    z_k = np.median(measurements)  # Robust to outliers
    
    # Prior prediction
    x_prior = simulation_state.get(f'{variable_name}_prior_estimate', z_k)
    
    # Innovation
    innovation = z_k - x_prior
    
    # Store in history
    innovation_history = simulation_state.get(f'{variable_name}_innovation_history', deque(maxlen=50))
    innovation_history.append(innovation)
    simulation_state[f'{variable_name}_innovation_history'] = innovation_history
    
    return innovation


def updateKalmanUncertainty(current_uncertainty: float, var_dynamics: VariableDynamics,
                           simulation_state: Dict[str, Any]) -> float:
    """
    Update uncertainty using Joseph form for numerical stability
    
    P(k|k) = (I - K·H)P(k|k-1)(I - K·H)^T + K·R·K^T
    """
    # Get Kalman gain
    K = simulation_state.get(f'{var_dynamics.name}_kalman_gain', 0.5)
    
    # Prior uncertainty
    P_prior = current_uncertainty**2
    
    # Measurement noise
    R = var_dynamics.measurement_noise**2
    
    # Joseph form update (scalar case)
    H = 1.0
    factor = (1 - K * H)
    P_posterior = factor**2 * P_prior + K**2 * R
    
    # Add process noise for next step
    Q = var_dynamics.process_noise**2
    P_next = P_posterior + Q
    
    # Return standard deviation
    return np.sqrt(P_next)


def performBayesianUpdate(available_predictions: Dict[str, float], innovation: float,
                         adaptive_gain: float, var_name: str, 
                         var_dynamics: VariableDynamics) -> Tuple[float, float]:
    """
    Perform Bayesian state update with adaptive gain
    
    x(k|k) = x(k|k-1) + K(k)·ν(k)
    """
    # Prior estimate (weighted average of predictions)
    weights = calculate_prediction_weights(available_predictions, var_dynamics)
    x_prior = sum(w * p for w, p in zip(weights, available_predictions.values()))
    
    # State update
    x_posterior = x_prior + adaptive_gain * innovation
    
    # Apply correction limits
    min_factor, max_factor = var_dynamics.correction_range
    x_posterior = np.clip(x_posterior, min_factor * x_prior, max_factor * x_prior)
    
    # Update uncertainty
    prior_uncertainty = np.sqrt(sum(w**2 * var_dynamics.measurement_noise**2 for w in weights))
    posterior_uncertainty = (1 - adaptive_gain) * prior_uncertainty
    
    return x_posterior, posterior_uncertainty


def calculateFusionWeights(available_predictions: Dict[str, Dict[str, float]], 
                          kalman_gains: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate fusion weights based on Kalman gains and prediction quality
    """
    # Layer names
    layers = ['layer1_advanced_physics', 'layer2_simplified_physics', 
              'layer3_empirical_ml', 'layer4_corrected']
    
    # Initialize weights
    weights = {layer: 0.0 for layer in layers}
    
    # Calculate weights based on inverse variance
    for layer in layers:
        if layer in available_predictions:
            # Get average uncertainty across variables
            uncertainties = []
            for var_name in ['temperature', 'wear_rate', 'surface_roughness']:
                if var_name in available_predictions[layer]:
                    var_dynamics = VARIABLE_DYNAMICS[var_name]
                    uncertainties.append(var_dynamics.measurement_noise)
            
            if uncertainties:
                avg_uncertainty = np.mean(uncertainties)
                weights[layer] = 1.0 / (avg_uncertainty**2)
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # No artificial weight boosting - keep physics-based weights as is
    
    # Renormalize
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    return weights


def validateKalmanPerformance(kalman_state: KalmanState, predictions: Dict[str, float],
                             var_name: str) -> Dict[str, float]:
    """
    Validate Kalman filter performance using innovation statistics
    """
    performance_metrics = {}
    
    # Innovation magnitude (should be small)
    performance_metrics['innovation_magnitude'] = abs(kalman_state.innovation)
    
    # Prediction spread (uncertainty in measurements)
    pred_values = list(predictions.values())
    performance_metrics['prediction_spread'] = np.std(pred_values) if len(pred_values) > 1 else 0
    
    # Normalized innovation (should be ~N(0,1))
    if kalman_state.uncertainty > 0:
        normalized_innovation = kalman_state.innovation / kalman_state.uncertainty
        performance_metrics['normalized_innovation'] = abs(normalized_innovation)
    else:
        performance_metrics['normalized_innovation'] = 0
    
    # Gain stability (should not vary too much)
    performance_metrics['gain_stability'] = 1.0 - abs(kalman_state.gain - 0.5) * 2
    
    # Overall performance score
    performance_metrics['overall_score'] = np.mean([
        1.0 / (1.0 + performance_metrics['innovation_magnitude'] / 10),
        1.0 / (1.0 + performance_metrics['prediction_spread'] / 5),
        1.0 / (1.0 + performance_metrics['normalized_innovation']),
        performance_metrics['gain_stability']
    ])
    
    return performance_metrics


def adaptKalmanParameters(var_dynamics: VariableDynamics, performance_metrics: Dict[str, float],
                         simulation_state: Dict[str, Any]) -> VariableDynamics:
    """
    Adapt Kalman parameters based on performance
    """
    # Copy dynamics to avoid modifying original
    adapted = VariableDynamics(
        name=var_dynamics.name,
        process_noise=var_dynamics.process_noise,
        measurement_noise=var_dynamics.measurement_noise,
        correction_range=var_dynamics.correction_range,
        time_constant=var_dynamics.time_constant,
        adaptation_rate=var_dynamics.adaptation_rate
    )
    
    # Get performance score
    score = performance_metrics.get('overall_score', 0.5)
    
    # Adapt process noise based on innovation magnitude
    innovation_mag = performance_metrics.get('innovation_magnitude', 0)
    if innovation_mag > adapted.process_noise * 3:
        # Large innovations - increase process noise
        adapted.process_noise *= (1 + adapted.adaptation_rate)
    elif innovation_mag < adapted.process_noise * 0.5:
        # Small innovations - decrease process noise
        adapted.process_noise *= (1 - adapted.adaptation_rate * 0.5)
    
    # Adapt measurement noise based on prediction spread
    pred_spread = performance_metrics.get('prediction_spread', 0)
    if pred_spread > adapted.measurement_noise * 2:
        # High spread - increase measurement noise
        adapted.measurement_noise *= (1 + adapted.adaptation_rate * 0.5)
    elif pred_spread < adapted.measurement_noise * 0.3:
        # Low spread - decrease measurement noise
        adapted.measurement_noise *= (1 - adapted.adaptation_rate * 0.3)
    
    # Store adapted parameters
    simulation_state[f'{var_dynamics.name}_adapted_dynamics'] = adapted
    
    return adapted


def monitorKalmanStability(kalman_state: KalmanState, var_dynamics: VariableDynamics) -> bool:
    """
    Monitor Kalman filter stability
    """
    # Check for divergence
    if abs(kalman_state.innovation) > var_dynamics.measurement_noise * 10:
        logger.warning(f"Large innovation detected for {var_dynamics.name}: {kalman_state.innovation}")
        return False
    
    # Check uncertainty growth
    if kalman_state.uncertainty > var_dynamics.measurement_noise * 5:
        logger.warning(f"High uncertainty for {var_dynamics.name}: {kalman_state.uncertainty}")
        return False
    
    # Check gain bounds
    if not 0.01 <= kalman_state.gain <= 0.99:
        logger.warning(f"Kalman gain out of bounds for {var_dynamics.name}: {kalman_state.gain}")
        return False
    
    return True


def logKalmanEvolution(kalman_states: Dict[str, KalmanState], kalman_gains: Dict[str, float],
                      performance_metrics: Dict[str, Dict[str, float]]):
    """
    Log Kalman filter evolution for analysis
    """
    log_dir = 'SFDP_6Layer_v17_3/kalman_corrections'
    os.makedirs(log_dir, exist_ok=True)
    
    evolution_log = {
        'timestamp': datetime.now().isoformat(),
        'kalman_states': {},
        'kalman_gains': kalman_gains,
        'performance_metrics': performance_metrics
    }
    
    # Convert KalmanState objects to dict
    for var_name, state in kalman_states.items():
        evolution_log['kalman_states'][var_name] = {
            'estimate': state.estimate,
            'uncertainty': state.uncertainty,
            'prediction': state.prediction,
            'innovation': state.innovation,
            'gain': state.gain
        }
    
    # Write log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{log_dir}/kalman_evolution_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(evolution_log, f, indent=2)
    
    logger.debug(f"Logged Kalman evolution to {filename}")


# Helper functions

def extract_layer_predictions(layer_results: Dict[str, Any], var_name: str) -> Dict[str, float]:
    """Extract predictions for a specific variable from all layers"""
    predictions = {}
    
    # Map variable names to result keys
    var_key_map = {
        'temperature': 'cutting_temperature',
        'wear_rate': 'tool_wear_rate',
        'surface_roughness': 'surface_roughness'
    }
    
    result_key = var_key_map.get(var_name, var_name)
    
    # Extract from each layer
    layer_names = ['layer1_advanced_physics', 'layer2_simplified_physics',
                   'layer3_empirical_ml', 'layer4_corrected']
    
    for layer in layer_names:
        if layer in layer_results and result_key in layer_results[layer]:
            predictions[layer] = layer_results[layer][result_key]
    
    return predictions


def extract_uncertainties(predictions: Dict[str, float], var_name: str) -> Dict[str, float]:
    """Extract uncertainty estimates for each prediction source"""
    uncertainties = {}
    var_dynamics = VARIABLE_DYNAMICS[var_name]
    
    # Assign uncertainties based on source
    uncertainty_factors = {
        'layer1_advanced_physics': 0.8,    # Most accurate
        'layer2_simplified_physics': 1.0,   # Baseline
        'layer3_empirical_ml': 1.2,        # More uncertain
        'layer4_corrected': 0.9            # Corrected, fairly accurate
    }
    
    for source, prediction in predictions.items():
        factor = uncertainty_factors.get(source, 1.0)
        uncertainties[source] = var_dynamics.measurement_noise * factor
    
    return uncertainties


def calculate_prediction_weights(predictions: Dict[str, float], 
                               var_dynamics: VariableDynamics) -> List[float]:
    """Calculate weights for predictions based on inverse variance"""
    uncertainties = extract_uncertainties(predictions, var_dynamics.name)
    
    # Inverse variance weighting
    weights = []
    for source in predictions:
        uncertainty = uncertainties.get(source, var_dynamics.measurement_noise)
        weight = 1.0 / (uncertainty**2)
        weights.append(weight)
    
    # Normalize
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w/total_weight for w in weights]
    else:
        weights = [1.0/len(predictions)] * len(predictions)
    
    return weights


def calculate_overall_confidence(kalman_states: Dict[str, KalmanState],
                               performance_metrics: Dict[str, Dict[str, float]],
                               stability_status: Dict[str, bool]) -> float:
    """Calculate overall Kalman filtering confidence"""
    confidences = []
    
    # State-based confidence
    for var_name, state in kalman_states.items():
        var_dynamics = VARIABLE_DYNAMICS[var_name]
        # Lower uncertainty = higher confidence
        uncertainty_ratio = state.uncertainty / var_dynamics.measurement_noise
        state_confidence = 1.0 / (1.0 + uncertainty_ratio)
        confidences.append(state_confidence)
    
    # Performance-based confidence
    for var_name, metrics in performance_metrics.items():
        perf_confidence = metrics.get('overall_score', 0.5)
        confidences.append(perf_confidence)
    
    # Stability-based confidence
    stability_confidence = sum(stability_status.values()) / len(stability_status)
    confidences.append(stability_confidence)
    
    # Overall confidence
    overall_confidence = np.mean(confidences)
    
    # Apply penalties
    if not all(stability_status.values()):
        overall_confidence *= 0.8  # Stability penalty
    
    return overall_confidence


# Advanced Kalman techniques

def applyExtendedKalmanFilter(nonlinear_model: callable, state: np.ndarray, 
                             measurement: np.ndarray, simulation_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Kalman Filter for nonlinear systems
    """
    # Linearize around current state
    # This is a placeholder - would need actual Jacobian calculation
    F = np.eye(len(state))  # State transition Jacobian
    H = np.eye(len(measurement))  # Measurement Jacobian
    
    # Standard Kalman update with linearized matrices
    # ... (implementation details)
    
    return state, np.eye(len(state))


def applyUnscentedKalmanFilter(state: np.ndarray, covariance: np.ndarray,
                               measurement: np.ndarray, simulation_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unscented Kalman Filter for highly nonlinear systems
    """
    # Generate sigma points
    n = len(state)
    alpha = 1e-3
    beta = 2
    kappa = 0
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Sigma points
    sigma_points = generate_sigma_points(state, covariance, lambda_)
    
    # Transform sigma points
    # ... (implementation details)
    
    return state, covariance


def generate_sigma_points(mean: np.ndarray, covariance: np.ndarray, 
                         lambda_: float) -> np.ndarray:
    """Generate sigma points for unscented transform"""
    n = len(mean)
    sigma_points = np.zeros((2*n + 1, n))
    
    # Mean point
    sigma_points[0] = mean
    
    # Square root of covariance
    try:
        sqrt_cov = sqrtm((n + lambda_) * covariance)
    except:
        sqrt_cov = np.eye(n) * np.sqrt(n + lambda_)
    
    # Positive and negative sigma points
    for i in range(n):
        sigma_points[i+1] = mean + sqrt_cov[i]
        sigma_points[n+i+1] = mean - sqrt_cov[i]
    
    return sigma_points


# Integration with tuning systems
def get_kalman_parameters():
    """Return adjustable Kalman parameters for tuning systems"""
    return {
        'temperature': {
            'process_noise': (1.0, 20.0),
            'measurement_noise': (5.0, 50.0),
            'correction_range': [(0.8, 1.2), (0.85, 1.15), (0.9, 1.1)],
            'adaptation_rate': (0.01, 0.1)
        },
        'wear_rate': {
            'process_noise': (1e-6, 1e-4),
            'measurement_noise': (1e-5, 1e-3),
            'correction_range': [(0.85, 1.15), (0.88, 1.12), (0.92, 1.08)],
            'adaptation_rate': (0.01, 0.05)
        },
        'surface_roughness': {
            'process_noise': (0.05, 0.5),
            'measurement_noise': (0.1, 1.0),
            'correction_range': [(0.75, 1.25), (0.82, 1.18), (0.9, 1.1)],
            'adaptation_rate': (0.02, 0.15)
        },
        'fusion': {
            'innovation_weight': (0.1, 0.9),
            'stability_threshold': (0.1, 0.5),
            'adaptation_memory': (5, 50)
        }
    }