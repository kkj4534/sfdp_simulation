#!/usr/bin/env python3
"""
SFDP Auto-Tuning System
=======================

Automatic parameter tuning system for 6-layer hierarchical architecture.
Optimizes layer parameters based on validation feedback.

Author: SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr)
Date: May 2025
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# SFDP modules
from modules.sfdp_initialize_system import sfdp_initialize_system
from modules.sfdp_intelligent_data_loader import sfdp_intelligent_data_loader
from modules.sfdp_comprehensive_validation import sfdp_comprehensive_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TUNING] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TuningParameters:
    """Parameters for each layer that can be tuned"""
    # Layer 1: Advanced Physics
    layer1_confidence_weight: float = 0.8
    layer1_convergence_tolerance: float = 1e-6
    
    # Layer 2: Simplified Physics
    layer2_confidence_weight: float = 0.65
    layer2_simplification_factor: float = 0.8
    
    # Layer 3: Empirical Assessment
    layer3_ml_weight: float = 0.55
    layer3_historical_weight: float = 0.4
    
    # Layer 4: Data Correction
    layer4_correction_factor: float = 0.7
    layer4_bias_threshold: float = 0.15
    
    # Layer 5: Kalman Filter
    layer5_kalman_gain: float = 0.15
    layer5_innovation_threshold: float = 0.1
    
    # Layer 6: Final Validation
    layer6_validation_threshold: float = 0.631  # Based on our data confidence


@dataclass
class TuningIteration:
    """Results from a single tuning iteration"""
    iteration_id: int
    parameters: TuningParameters
    validation_score: float
    individual_scores: Dict[str, float]
    execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AutoTuningSystem:
    """Automatic parameter tuning system"""
    
    def __init__(self, max_iterations: int = 15, target_score: float = 0.631):
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.current_best_score = 0.0
        self.current_best_params = TuningParameters()
        self.tuning_history: List[TuningIteration] = []
        self.patience = 5  # Early stopping patience
        self.no_improvement_count = 0
        
    def generate_parameter_variation(self, base_params: TuningParameters, 
                                   iteration: int) -> TuningParameters:
        """Generate parameter variations for tuning"""
        
        # Progressive refinement: larger changes early, smaller changes later
        variation_scale = max(0.05, 0.3 * (1 - iteration / self.max_iterations))
        
        new_params = TuningParameters()
        
        # Layer 1 tuning
        new_params.layer1_confidence_weight = np.clip(
            base_params.layer1_confidence_weight + np.random.normal(0, variation_scale * 0.1),
            0.5, 0.95
        )
        
        # Layer 2 tuning  
        new_params.layer2_confidence_weight = np.clip(
            base_params.layer2_confidence_weight + np.random.normal(0, variation_scale * 0.1),
            0.4, 0.8
        )
        
        # Layer 3 tuning
        new_params.layer3_ml_weight = np.clip(
            base_params.layer3_ml_weight + np.random.normal(0, variation_scale * 0.15),
            0.3, 0.8
        )
        
        # Layer 4 tuning
        new_params.layer4_correction_factor = np.clip(
            base_params.layer4_correction_factor + np.random.normal(0, variation_scale * 0.1),
            0.5, 0.9
        )
        
        # Layer 5 Kalman tuning
        new_params.layer5_kalman_gain = np.clip(
            base_params.layer5_kalman_gain + np.random.normal(0, variation_scale * 0.05),
            0.05, 0.35
        )
        
        # Layer 6 validation threshold (less aggressive tuning)
        new_params.layer6_validation_threshold = np.clip(
            base_params.layer6_validation_threshold + np.random.normal(0, variation_scale * 0.02),
            0.55, 0.75
        )
        
        return new_params
    
    def run_simulation_with_params(self, params: TuningParameters) -> Tuple[float, Dict[str, float]]:
        """Run simulation with given parameters and return validation score"""
        
        try:
            # Initialize system
            state = sfdp_initialize_system()
            
            # Apply tuning parameters to state
            state.physics.base_confidence = params.layer1_confidence_weight
            state.kalman.base_gain = params.layer5_kalman_gain
            state.kalman.innovation_threshold = params.layer5_innovation_threshold
            
            # Load data
            extended_data, data_confidence, _ = sfdp_intelligent_data_loader(state)
            
            # Generate synthetic simulation results (representing 6-layer calculation)
            # In real implementation, this would call sfdp_execute_6layer_calculations
            simulation_results = self._generate_synthetic_results(params)
            
            # Run validation
            validation_results = sfdp_comprehensive_validation(state, simulation_results, extended_data)
            
            # Extract scores
            if isinstance(validation_results, dict) and 'validation_summary' in validation_results:
                overall_score = validation_results['validation_summary'].get('overall_confidence', 0.0)
                
                # Extract individual level scores
                individual_scores = {}
                if 'level_results' in validation_results:
                    for level_result in validation_results['level_results']:
                        level_name = f"Level_{level_result['level']}"
                        individual_scores[level_name] = level_result['confidence']
                
                return overall_score, individual_scores
            else:
                return 0.0, {}
                
        except Exception as e:
            logger.error(f"Simulation failed with params: {e}")
            return 0.0, {}
    
    def _generate_synthetic_results(self, params: TuningParameters) -> Dict[str, Any]:
        """Generate synthetic simulation results based on parameters"""
        
        # Simulate the effect of different layer parameters
        base_temp = 350
        base_wear = 0.1
        base_roughness = 1.2
        
        # Layer effects on results
        layer1_effect = params.layer1_confidence_weight
        layer3_effect = params.layer3_ml_weight
        layer4_effect = params.layer4_correction_factor
        
        # Generate results with some randomness but influenced by parameters
        n_points = 10
        results = {
            'cutting_temperature': np.random.normal(
                base_temp * (0.8 + 0.4 * layer1_effect), 
                20 + 10 * (1 - layer4_effect), 
                n_points
            ),
            'tool_wear_rate': np.random.normal(
                base_wear * (0.7 + 0.6 * layer3_effect),
                0.02 * (1 - layer4_effect),
                n_points
            ),
            'surface_roughness': np.random.normal(
                base_roughness * (0.6 + 0.8 * layer1_effect),
                0.3 * (1 - layer4_effect),
                n_points
            )
        }
        
        return results
    
    def run_auto_tuning(self) -> Tuple[TuningParameters, List[TuningIteration]]:
        """Run the complete auto-tuning process"""
        
        logger.info("🔄 Starting Auto-Tuning System...")
        logger.info(f"🎯 Target validation score: {self.target_score:.3f}")
        logger.info(f"🔢 Maximum iterations: {self.max_iterations}")
        
        # Start with default parameters
        current_params = TuningParameters()
        
        for iteration in range(self.max_iterations):
            start_time = time.time()
            
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Generate parameter variation
            if iteration == 0:
                test_params = current_params  # Use default for first iteration
            else:
                test_params = self.generate_parameter_variation(self.current_best_params, iteration)
            
            # Run simulation
            logger.info("🔬 Running simulation with tuned parameters...")
            validation_score, individual_scores = self.run_simulation_with_params(test_params)
            
            execution_time = time.time() - start_time
            
            # Record iteration
            iteration_result = TuningIteration(
                iteration_id=iteration + 1,
                parameters=test_params,
                validation_score=validation_score,
                individual_scores=individual_scores,
                execution_time=execution_time
            )
            self.tuning_history.append(iteration_result)
            
            # Check for improvement
            if validation_score > self.current_best_score:
                self.current_best_score = validation_score
                self.current_best_params = test_params
                self.no_improvement_count = 0
                logger.info(f"✅ NEW BEST: {validation_score:.3f} (improved by {validation_score - self.current_best_score:.3f})")
            else:
                self.no_improvement_count += 1
                logger.info(f"⏸️  No improvement: {validation_score:.3f} (best: {self.current_best_score:.3f})")
            
            # Check target achievement
            if validation_score >= self.target_score:
                logger.info(f"🎯 TARGET ACHIEVED! Score: {validation_score:.3f}")
                break
            
            # Check early stopping
            if self.no_improvement_count >= self.patience:
                logger.info(f"⏹️  Early stopping: No improvement for {self.patience} iterations")
                break
            
            logger.info(f"⏱️  Iteration time: {execution_time:.2f}s")
        
        # Final summary
        logger.info(f"\n🏁 AUTO-TUNING COMPLETE!")
        logger.info(f"🎯 Best validation score: {self.current_best_score:.3f}")
        logger.info(f"🔢 Total iterations: {len(self.tuning_history)}")
        
        return self.current_best_params, self.tuning_history


def main():
    """Main entry point for auto-tuning system"""
    
    print("=" * 60)
    print("🔄 SFDP Auto-Tuning System v1.0")
    print("=" * 60)
    
    # Create tuning system
    tuner = AutoTuningSystem(max_iterations=15, target_score=0.631)
    
    # Run tuning
    best_params, history = tuner.run_auto_tuning()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save tuning history
    history_file = f"tuning_history_{timestamp}.json"
    with open(history_file, 'w') as f:
        history_data = [
            {
                'iteration_id': h.iteration_id,
                'validation_score': h.validation_score,
                'individual_scores': h.individual_scores,
                'execution_time': h.execution_time,
                'timestamp': h.timestamp
            } for h in history
        ]
        json.dump(history_data, f, indent=2)
    
    print(f"\n📊 Results saved to: {history_file}")
    print(f"🎯 Best validation score achieved: {tuner.current_best_score:.3f}")
    
    return best_params, history


if __name__ == "__main__":
    main()