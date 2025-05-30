"""
SFDP_CONDITIONS_OPTIMIZER - Machining Conditions Optimization Module
====================================================================

MATLAB to Python 1:1 Migration
Grey Wolf Optimizer (GWO) based parameter optimization for machining conditions.
Multi-objective function considering productivity, quality, tool life.

Original MATLAB Reference: 
Mirjalili et al. (2014) "Grey Wolf Optimizer" Adv. Eng. Software 69:46-61

Author: SFDP Research Team (memento1087@gmail.com) (Python Migration)
Date: May 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class OptimizationResults:
    """Optimization results data structure - 1:1 MATLAB migration"""
    method: str
    best_score: float
    convergence_curve: List[float]
    iterations: int
    search_agents: int
    optimization_criteria: List[str]
    
def sfdp_conditions_optimizer(
    simulation_state: Dict[str, Any],
    selected_tools: Dict[str, Any],
    taylor_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    SFDP_CONDITIONS_OPTIMIZER - 1:1 MATLAB Migration
    
    Grey Wolf Optimizer based parameter optimization for machining conditions
    Multi-objective function considering productivity, quality, tool life
    Constraint handling for machine tool limitations and workpiece geometry
    
    Args:
        simulation_state: Global simulation configuration
        selected_tools: Selected tool specifications structure  
        taylor_results: Extended Taylor coefficient results
        
    Returns:
        Tuple[Dict, Dict]: optimized_conditions, optimization_results
    """
    
    print('\n=== SECTION 6: MACHINING CONDITIONS OPTIMIZATION ===')
    print('Grey Wolf Optimizer (GWO) based parameter optimization')
    print('Multi-objective function considering productivity, quality, tool life')
    
    # GWO optimization parameters (following MATLAB implementation)
    nvars = 3  # cutting_speed, feed_rate, depth_of_cut
    search_agents = 10
    max_iterations = 50
    
    # Define bounds for optimization variables
    # cutting_speed (m/min), feed_rate (mm/rev), depth_of_cut (mm)
    lb = np.array([50.0, 0.05, 0.5])    # Lower bounds
    ub = np.array([300.0, 0.3, 3.0])    # Upper bounds
    
    try:
        # Define objective function for multi-criteria optimization
        def objective_function(x):
            return evaluate_machining_performance(x, selected_tools, taylor_results, simulation_state)
        
        # Run GWO optimization (1:1 MATLAB migration)
        best_position, best_score, convergence_curve = gwo_optimizer(
            objective_function, nvars, search_agents, max_iterations, lb, ub
        )
        
        # Extract optimized conditions
        optimized_conditions = {
            'cutting_speed': best_position[0],     # m/min
            'feed_rate': best_position[1],         # mm/rev  
            'depth_of_cut': best_position[2],      # mm
            'coolant_flow': 5.0,                   # L/min (default)
            'spindle_speed': calculate_spindle_speed(best_position[0])  # rpm
        }
        
        # Generate optimization report (1:1 MATLAB structure)
        optimization_results = {
            'method': 'GWO',
            'best_score': best_score,
            'convergence_curve': convergence_curve,
            'iterations': max_iterations,
            'search_agents': search_agents,
            'optimization_criteria': [
                'Tool life maximization (40%)',
                'Surface quality optimization (25%)', 
                'Cost minimization (20%)',
                'Productivity maximization (15%)'
            ],
            'optimized_variables': {
                'cutting_speed': optimized_conditions['cutting_speed'],
                'feed_rate': optimized_conditions['feed_rate'], 
                'depth_of_cut': optimized_conditions['depth_of_cut']
            }
        }
        
        print(f'  ✓ GWO optimization completed successfully')
        print(f'  Best score: {best_score:.6f}')
        print(f'  Optimal cutting speed: {optimized_conditions["cutting_speed"]:.1f} m/min')
        print(f'  Optimal feed rate: {optimized_conditions["feed_rate"]:.3f} mm/rev')
        print(f'  Optimal depth of cut: {optimized_conditions["depth_of_cut"]:.1f} mm')
        
    except Exception as e:
        print(f'  ⚠️  GWO optimization failed: {str(e)}')
        print('  Falling back to default machining conditions.')
        
        # Fallback to default conditions
        optimized_conditions = {
            'cutting_speed': 150.0,
            'feed_rate': 0.15,
            'depth_of_cut': 1.5,
            'coolant_flow': 5.0,
            'spindle_speed': calculate_spindle_speed(150.0)
        }
        
        optimization_results = {
            'method': 'fallback',
            'error': str(e),
            'best_score': 1.0,
            'convergence_curve': [],
            'iterations': 0,
            'search_agents': 0,
            'optimization_criteria': []
        }
    
    return optimized_conditions, optimization_results


def evaluate_machining_performance(x, selected_tools, taylor_results, simulation_state):
    """
    Multi-criteria machining performance evaluation - 1:1 MATLAB migration
    Lower score = better performance (for minimization)
    
    Args:
        x: [cutting_speed, feed_rate, depth_of_cut]
        selected_tools: Selected tool specifications
        taylor_results: Taylor coefficient results
        simulation_state: Simulation state
        
    Returns:
        float: Performance score (lower is better)
    """
    
    cutting_speed, feed_rate, depth_of_cut = x
    
    # Validate input ranges
    if cutting_speed < 50 or cutting_speed > 300:
        return 1e6  # Penalty for out of bounds
    if feed_rate < 0.05 or feed_rate > 0.3:
        return 1e6
    if depth_of_cut < 0.5 or depth_of_cut > 3.0:
        return 1e6
    
    # Extract tool parameters
    primary_tool = selected_tools.get('primary_tool', {})
    nose_radius = primary_tool.get('nose_radius', 0.8e-3)  # m
    expected_life = primary_tool.get('expected_life', 30.0)  # minutes
    
    # Extract Taylor coefficients
    taylor_coeffs = getattr(taylor_results, 'coefficients', None)
    if taylor_coeffs:
        taylor_n = getattr(taylor_coeffs, 'n', 0.25)
        taylor_c = getattr(taylor_coeffs, 'C', 150.0)
    else:
        taylor_n = 0.25
        taylor_c = 150.0
    
    ## CRITERION 1: Tool Life (40% weight) - Maximize
    # Use Taylor equation: VT^n = C
    predicted_tool_life = (taylor_c / cutting_speed)**(1/taylor_n)
    
    # Normalize to 0-1 scale (longer life = better)
    max_expected_life = 120.0  # minutes
    tool_life_score = min(predicted_tool_life / max_expected_life, 1.0)
    tool_life_penalty = 1.0 - tool_life_score  # Convert to penalty
    
    ## CRITERION 2: Surface Quality (25% weight) - Maximize
    # Theoretical surface roughness: Ra = f²/(32*r)
    theoretical_ra = (feed_rate**2) / (32 * nose_radius * 1000)  # μm
    target_surface_finish = 1.6  # μm Ra target
    surface_quality_ratio = theoretical_ra / target_surface_finish
    
    if surface_quality_ratio <= 1.0:
        surface_quality_penalty = 0.1 * surface_quality_ratio
    else:
        surface_quality_penalty = 0.1 + 0.9 * (surface_quality_ratio - 1.0)
    
    ## CRITERION 3: Cost (20% weight) - Minimize
    # Cost per part estimation
    parts_per_edge = predicted_tool_life / (depth_of_cut / feed_rate)
    cost_per_edge = primary_tool.get('cost_per_edge', 25.0)  # USD
    cost_per_part = cost_per_edge / max(parts_per_edge, 1)
    
    # Normalize cost (typical range 0.5-20 USD per part)
    max_expected_cost = 20.0
    cost_penalty = min(cost_per_part / max_expected_cost, 1.0)
    
    ## CRITERION 4: Productivity (15% weight) - Maximize
    # Material Removal Rate (MRR)
    calculated_mrr = cutting_speed * feed_rate * depth_of_cut * 1000  # mm³/min
    max_mrr = 1000.0  # mm³/min reference
    productivity_ratio = min(calculated_mrr / max_mrr, 1.0)
    productivity_penalty = 1.0 - productivity_ratio
    
    ## WEIGHTED COMBINATION (1:1 MATLAB weights)
    weights = np.array([0.40, 0.25, 0.20, 0.15])  # [tool_life, surface_quality, cost, productivity]
    penalties = np.array([tool_life_penalty, surface_quality_penalty, cost_penalty, productivity_penalty])
    
    # Final score (lower is better for GWO minimization)
    score = np.sum(weights * penalties)
    
    # Add penalty for extreme operating conditions
    if cutting_speed > 200 or feed_rate > 0.4:
        score += 0.1  # Penalty for aggressive conditions
    
    return score


def gwo_optimizer(objective_function, nvars, search_agents, max_iterations, lb, ub):
    """
    Grey Wolf Optimizer implementation - 1:1 MATLAB migration
    
    Reference: Mirjalili et al. (2014) "Grey Wolf Optimizer" Adv. Eng. Software 69:46-61
    """
    
    # Initialize the positions of search agents
    positions = np.random.uniform(lb, ub, (search_agents, nvars))
    
    # Initialize alpha, beta, and delta positions
    alpha_pos = np.zeros(nvars)
    alpha_score = float('inf')
    
    beta_pos = np.zeros(nvars)
    beta_score = float('inf')
    
    delta_pos = np.zeros(nvars)
    delta_score = float('inf')
    
    convergence_curve = []
    
    # Main loop
    for iteration in range(max_iterations):
        
        # Update the position of search agents
        for i in range(search_agents):
            
            # Calculate objective function for each search agent
            fitness = objective_function(positions[i, :])
            
            # Update Alpha, Beta, and Delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            
            if fitness > alpha_score and fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            
            if fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()
        
        # Update the positions
        a = 2 - iteration * (2 / max_iterations)  # Linearly decreased from 2 to 0
        
        for i in range(search_agents):
            for j in range(nvars):
                
                # Position updating for alpha
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Position updating for beta
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Position updating for delta
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Update position
                positions[i, j] = (X1 + X2 + X3) / 3
                
                # Boundary checking
                positions[i, j] = max(positions[i, j], lb[j])
                positions[i, j] = min(positions[i, j], ub[j])
        
        convergence_curve.append(alpha_score)
    
    return alpha_pos, alpha_score, convergence_curve


def calculate_spindle_speed(cutting_speed):
    """
    Calculate spindle speed from cutting speed
    Assuming typical tool diameter of 50mm
    """
    tool_diameter = 50.0  # mm
    spindle_speed = (cutting_speed * 1000) / (np.pi * tool_diameter)  # rpm
    return spindle_speed


if __name__ == "__main__":
    # Test functionality
    test_simulation_state = {}
    test_selected_tools = {
        'primary_tool': {
            'nose_radius': 0.8e-3,
            'expected_life': 30.0,
            'cost_per_edge': 25.0
        }
    }
    test_taylor_results = {
        'coefficients': {
            'n_value': 0.25,
            'c_value': 150.0
        }
    }
    
    conditions, results = sfdp_conditions_optimizer(
        test_simulation_state, test_selected_tools, test_taylor_results
    )
    
    print(f"\nTest completed!")
    print(f"Optimized conditions: {conditions}")
    print(f"Optimization method: {results['method']}")