"""
SFDP Proper Kalman Filter Implementation for Layer 5
===================================================

Dynamic Kalman filter that properly fuses physics and empirical results
based on their uncertainties and historical performance.

Author: SFDP Research Team
Date: May 2025
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state for each variable"""
    x: np.ndarray        # State estimate [value, rate_of_change]
    P: np.ndarray        # Error covariance matrix (2x2)
    innovation: float    # Latest innovation (measurement residual)
    gain: float         # Kalman gain


class ProperKalmanLayer5:
    """Proper implementation of adaptive Kalman filtering for Layer 5"""
    
    def __init__(self, simulation_state: Dict[str, Any]):
        self.simulation_state = simulation_state
        
        # Initialize Kalman states for each variable
        self.kalman_states = {
            'temperature': KalmanState(
                x=np.array([300.0, 0.0]),  # [temperature, rate]
                P=np.eye(2) * 100,         # Initial uncertainty
                innovation=0.0,
                gain=0.5
            ),
            'wear_rate': KalmanState(
                x=np.array([0.01, 0.0]),   # [wear_rate, acceleration]
                P=np.eye(2) * 0.01,
                innovation=0.0,
                gain=0.5
            ),
            'roughness': KalmanState(
                x=np.array([1.0, 0.0]),    # [roughness, rate]
                P=np.eye(2) * 1.0,
                innovation=0.0,
                gain=0.5
            ),
            'force': KalmanState(
                x=np.array([500.0, 0.0]),  # [force, rate]
                P=np.eye(2) * 1000,
                innovation=0.0,
                gain=0.5
            )
        }
        
        # Process noise (Q) and measurement noise (R) matrices
        self.Q_base = {
            'temperature': np.diag([25.0, 1.0]),      # Temperature variance, rate variance
            'wear_rate': np.diag([0.0001, 0.00001]), # Wear variance, acceleration variance
            'roughness': np.diag([0.01, 0.001]),     # Roughness variance, rate variance
            'force': np.diag([100.0, 10.0])          # Force variance, rate variance
        }
        
        self.R_base = {
            'temperature': 50.0,    # Measurement noise variance
            'wear_rate': 0.001,
            'roughness': 0.1,
            'force': 200.0
        }
        
        # Time step
        self.dt = 0.1  # seconds
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, self.dt],
                          [0, 1]])
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0]])
        
    def execute_adaptive_kalman_filter(
        self,
        layer_results: Dict[str, Any],
        cutting_speed: float,
        feed_rate: float,
        depth_of_cut: float
    ) -> Tuple[Dict[str, Any], float]:
        """
        Execute proper adaptive Kalman filtering
        """
        
        logger.info("🧠 Executing proper adaptive Kalman filter...")
        
        # Extract measurements from different layers
        measurements = self._extract_layer_measurements(layer_results)
        
        # Extract uncertainties from layer confidences
        uncertainties = self._extract_layer_uncertainties(layer_results)
        
        results = {}
        
        for variable in ['temperature', 'wear_rate', 'roughness', 'force']:
            # Get measurements and weights
            layer_values = measurements[variable]
            layer_weights = uncertainties[variable]
            
            # Weighted measurement (fusion of L1, L2, L3, L4)
            z = self._calculate_weighted_measurement(layer_values, layer_weights)
            
            # Adaptive process noise based on cutting conditions
            Q = self._adapt_process_noise(variable, cutting_speed, feed_rate, depth_of_cut)
            
            # Adaptive measurement noise based on layer agreement
            R = self._adapt_measurement_noise(variable, layer_values)
            
            # Kalman filter prediction step
            x_pred, P_pred = self._predict(variable, Q)
            
            # Kalman filter update step
            x_updated, P_updated, K, innovation = self._update(variable, x_pred, P_pred, z, R)
            
            # Store updated state
            self.kalman_states[variable].x = x_updated
            self.kalman_states[variable].P = P_updated
            self.kalman_states[variable].gain = K[0, 0]  # Main gain component
            self.kalman_states[variable].innovation = innovation
            
            # Extract filtered value
            results[f'kalman_{variable}'] = x_updated[0]
        
        # Calculate overall confidence based on innovation consistency
        confidence = self._calculate_filter_confidence()
        
        # Add diagnostic information
        results['kalman_diagnostics'] = {
            'gains': {var: state.gain for var, state in self.kalman_states.items()},
            'innovations': {var: state.innovation for var, state in self.kalman_states.items()},
            'uncertainties': {var: np.sqrt(state.P[0, 0]) for var, state in self.kalman_states.items()}
        }
        
        results['method'] = 'Proper Adaptive Kalman Filter'
        
        return results, confidence
    
    def _extract_layer_measurements(self, layer_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract measurements from each layer"""
        
        measurements = {
            'temperature': {},
            'wear_rate': {},
            'roughness': {},
            'force': {}
        }
        
        # Layer 1 (Advanced Physics)
        if 'L1_advanced_physics' in layer_results:
            L1 = layer_results['L1_advanced_physics']
            measurements['temperature']['L1'] = L1.get('thermal_analysis', {}).get('max_temperature', 500)
            measurements['wear_rate']['L1'] = L1.get('wear_analysis', {}).get('wear_rate', 0.01)
            measurements['roughness']['L1'] = L1.get('surface_analysis', {}).get('Ra', 1.0)
            measurements['force']['L1'] = L1.get('force_analysis', {}).get('cutting_forces', {}).get('Fc', 500)
        
        # Layer 2 (Simplified Physics)
        if 'L2_simplified_physics' in layer_results:
            L2 = layer_results['L2_simplified_physics']
            measurements['temperature']['L2'] = L2.get('thermal_analysis', {}).get('max_temperature', 500)
            measurements['wear_rate']['L2'] = L2.get('wear_rate', 0.01)
            measurements['roughness']['L2'] = L2.get('surface_roughness', {}).get('Ra', 1.0)
            measurements['force']['L2'] = L2.get('force_analysis', {}).get('cutting_forces', {}).get('Fc', 500)
        
        # Layer 3 (Empirical)
        if 'L3_empirical_assessment' in layer_results:
            L3 = layer_results['L3_empirical_assessment']
            measurements['temperature']['L3'] = L3.get('temperature_empirical', 500)
            measurements['wear_rate']['L3'] = L3.get('wear_rate_empirical', 0.01)
            measurements['roughness']['L3'] = L3.get('roughness_empirical', 1.0)
            measurements['force']['L3'] = L3.get('force_empirical', 500)
        
        # Layer 4 (Corrected)
        if 'L4_empirical_correction' in layer_results:
            L4 = layer_results['L4_empirical_correction']
            measurements['temperature']['L4'] = L4.get('corrected_temperature', 500)
            measurements['wear_rate']['L4'] = L4.get('corrected_wear_rate', 0.01)
            measurements['roughness']['L4'] = L4.get('corrected_roughness', 1.0)
            measurements['force']['L4'] = L4.get('corrected_force', 500)
        
        return measurements
    
    def _extract_layer_uncertainties(self, layer_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract uncertainties (weights) based on layer confidences"""
        
        # Get layer confidences
        layer_confidences = layer_results.get('layer_confidence', [0.9, 0.85, 0.75, 0.8, 0.85, 0.9])
        
        # Convert confidence to weight (inverse of variance)
        weights = {
            'temperature': {
                'L1': layer_confidences[0] ** 2,      # Physics layers more trusted for temperature
                'L2': layer_confidences[1] ** 2,
                'L3': layer_confidences[2] ** 1.5,
                'L4': layer_confidences[3] ** 1.8
            },
            'wear_rate': {
                'L1': layer_confidences[0] ** 1.5,    # Empirical layers more trusted for wear
                'L2': layer_confidences[1] ** 1.5,
                'L3': layer_confidences[2] ** 2,
                'L4': layer_confidences[3] ** 2
            },
            'roughness': {
                'L1': layer_confidences[0] ** 1.5,
                'L2': layer_confidences[1] ** 1.5,
                'L3': layer_confidences[2] ** 2,      # Empirical best for roughness
                'L4': layer_confidences[3] ** 2
            },
            'force': {
                'L1': layer_confidences[0] ** 2,      # Physics best for force
                'L2': layer_confidences[1] ** 1.8,
                'L3': layer_confidences[2] ** 1.5,
                'L4': layer_confidences[3] ** 1.8
            }
        }
        
        return weights
    
    def _calculate_weighted_measurement(self, layer_values: Dict[str, float], 
                                      layer_weights: Dict[str, float]) -> float:
        """Calculate weighted average of layer measurements"""
        
        total_weight = sum(layer_weights.values())
        if total_weight == 0:
            return np.mean(list(layer_values.values()))
        
        weighted_sum = sum(value * layer_weights[layer] 
                          for layer, value in layer_values.items())
        
        return weighted_sum / total_weight
    
    def _adapt_process_noise(self, variable: str, cutting_speed: float, 
                           feed_rate: float, depth_of_cut: float) -> np.ndarray:
        """Adapt process noise based on cutting conditions"""
        
        # Base process noise
        Q = self.Q_base[variable].copy()
        
        # Scale based on cutting aggressiveness
        aggressiveness = (cutting_speed / 100) * (feed_rate / 0.2) * (depth_of_cut / 1.0)
        
        # More aggressive cutting = more process noise
        Q *= (1 + 0.5 * aggressiveness)
        
        return Q
    
    def _adapt_measurement_noise(self, variable: str, 
                               layer_values: Dict[str, float]) -> float:
        """Adapt measurement noise based on layer agreement"""
        
        # Base measurement noise
        R = self.R_base[variable]
        
        # Calculate standard deviation of layer predictions
        values = list(layer_values.values())
        if len(values) > 1:
            std_dev = np.std(values)
            mean_val = np.mean(values)
            
            # Relative disagreement
            if mean_val > 0:
                relative_disagreement = std_dev / mean_val
                
                # More disagreement = more measurement noise
                R *= (1 + 2 * relative_disagreement)
        
        return R
    
    def _predict(self, variable: str, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter prediction step"""
        
        state = self.kalman_states[variable]
        
        # State prediction: x_pred = F * x
        x_pred = self.F @ state.x
        
        # Covariance prediction: P_pred = F * P * F' + Q
        P_pred = self.F @ state.P @ self.F.T + Q
        
        return x_pred, P_pred
    
    def _update(self, variable: str, x_pred: np.ndarray, P_pred: np.ndarray,
                z: float, R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Kalman filter update step"""
        
        # Innovation: y = z - H * x_pred
        y = z - self.H @ x_pred
        
        # Innovation covariance: S = H * P_pred * H' + R
        S = self.H @ P_pred @ self.H.T + R
        
        # Kalman gain: K = P_pred * H' * inv(S)
        K = P_pred @ self.H.T / S
        
        # State update: x = x_pred + K * y
        x_updated = x_pred + K * y
        
        # Covariance update: P = (I - K * H) * P_pred
        P_updated = (np.eye(2) - K @ self.H) @ P_pred
        
        return x_updated, P_updated, K, float(y)
    
    def _calculate_filter_confidence(self) -> float:
        """Calculate overall filter confidence based on performance metrics"""
        
        confidences = []
        
        for variable, state in self.kalman_states.items():
            # Confidence based on innovation size (smaller is better)
            innovation_confidence = 1.0 / (1.0 + abs(state.innovation) / self.R_base[variable])
            
            # Confidence based on uncertainty (smaller is better)
            uncertainty = np.sqrt(state.P[0, 0])
            uncertainty_confidence = 1.0 / (1.0 + uncertainty / (0.1 * abs(state.x[0]) + 1))
            
            # Confidence based on Kalman gain (moderate is best)
            gain_confidence = 1.0 - abs(state.gain - 0.5) * 2
            
            # Combined confidence
            variable_confidence = (innovation_confidence + uncertainty_confidence + gain_confidence) / 3
            confidences.append(variable_confidence)
        
        # Overall confidence
        return np.mean(confidences) * 0.9  # Scale to max 0.9