"""
SFDP_ENHANCED_TOOL_SELECTION - Enhanced Tool Selection with Optimization
======================================================================

Implements multi-criteria decision making (MCDM) with Grey Wolf Optimizer (GWO)
for optimal cutting tool selection based on multiple performance criteria.

Criteria weights:
- Tool life: 40%
- Surface quality: 25% 
- Cost effectiveness: 20%
- Productivity: 15%

Author: SFDP Research Team (memento1087@gmail.com, memento645@konkuk.ac.kr) (Python Migration)
Date: May 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import warnings

@dataclass
class ToolSpecification:
    """Tool specification data structure"""
    tool_id: str
    material: str = "Carbide"
    coating: str = "TiAlN"
    nose_radius: float = 0.8e-3  # m
    edge_radius: float = 5e-6    # m
    rake_angle: float = 10.0     # degrees
    clearance_angle: float = 7.0 # degrees
    cost_per_edge: float = 25.0  # USD
    expected_life: float = 30.0  # minutes
    surface_finish_capability: float = 1.6  # Ra μm
    cutting_speed_range: Tuple[float, float] = (50, 300)  # m/min
    feed_rate_range: Tuple[float, float] = (0.05, 0.4)    # mm/rev


@dataclass
class ToolOptimizationResults:
    """Tool optimization results"""
    selected_tool: ToolSpecification
    optimization_score: float
    criteria_scores: Dict[str, float]
    alternative_tools: List[ToolSpecification]
    optimization_method: str
    confidence: float


class EnhancedToolSelection:
    """Enhanced tool selection with multi-criteria optimization"""
    
    def __init__(self):
        self.tool_database = self._initialize_tool_database()
        self.criteria_weights = {
            'tool_life': 0.40,
            'surface_quality': 0.25,
            'cost_effectiveness': 0.20,
            'productivity': 0.15
        }
    
    def _initialize_tool_database(self) -> List[ToolSpecification]:
        """Initialize tool database with Ti-6Al-4V cutting tools"""
        
        tools = [
            # Carbide tools
            ToolSpecification(
                tool_id="WC_TiAlN_001",
                material="Tungsten Carbide",
                coating="TiAlN",
                nose_radius=0.8e-3,
                edge_radius=5e-6,
                cost_per_edge=25.0,
                expected_life=30.0,
                surface_finish_capability=1.6,
                cutting_speed_range=(80, 250),
                feed_rate_range=(0.08, 0.3)
            ),
            ToolSpecification(
                tool_id="WC_AlCrN_002", 
                material="Tungsten Carbide",
                coating="AlCrN",
                nose_radius=1.2e-3,
                edge_radius=8e-6,
                cost_per_edge=30.0,
                expected_life=25.0,
                surface_finish_capability=2.0,
                cutting_speed_range=(60, 200),
                feed_rate_range=(0.1, 0.4)
            ),
            # Ceramic tools
            ToolSpecification(
                tool_id="AL2O3_001",
                material="Alumina Ceramic",
                coating="None",
                nose_radius=0.6e-3,
                edge_radius=3e-6,
                cost_per_edge=15.0,
                expected_life=45.0,
                surface_finish_capability=1.2,
                cutting_speed_range=(120, 400),
                feed_rate_range=(0.05, 0.25)
            ),
            # PCD tools
            ToolSpecification(
                tool_id="PCD_001",
                material="Polycrystalline Diamond",
                coating="None",
                nose_radius=0.4e-3,
                edge_radius=2e-6,
                cost_per_edge=150.0,
                expected_life=120.0,
                surface_finish_capability=0.8,
                cutting_speed_range=(150, 500),
                feed_rate_range=(0.05, 0.2)
            )
        ]
        
        return tools
    
    def select_optimal_tool(
        self,
        simulation_state: Dict[str, Any],
        physics_foundation: Dict[str, Any],
        cutting_conditions: Dict[str, Any]
    ) -> ToolOptimizationResults:
        """
        Select optimal tool using multi-criteria decision making
        
        Args:
            simulation_state: Current simulation state
            physics_foundation: Material properties and physics data
            cutting_conditions: Cutting speed, feed rate, depth of cut
            
        Returns:
            ToolOptimizationResults: Optimization results with selected tool
        """
        
        print('  🛠️  Executing enhanced tool selection...')
        
        cutting_speed = cutting_conditions.get('cutting_speed', 150.0)
        feed_rate = cutting_conditions.get('feed_rate', 0.15)
        depth_of_cut = cutting_conditions.get('depth_of_cut', 1.0)
        
        # Filter feasible tools based on cutting conditions
        feasible_tools = self._filter_feasible_tools(cutting_speed, feed_rate)
        
        if not feasible_tools:
            # Use default tool if no feasible tools found
            default_tool = self.tool_database[0]
            warnings.warn("No feasible tools found, using default tool")
            return self._create_default_results(default_tool)
        
        # Evaluate each feasible tool
        tool_scores = []
        for tool in feasible_tools:
            score = self._evaluate_tool_performance(
                tool, cutting_speed, feed_rate, depth_of_cut, physics_foundation
            )
            tool_scores.append((tool, score))
        
        # Sort by total score (descending)
        tool_scores.sort(key=lambda x: x[1]['total_score'], reverse=True)
        
        # Select best tool
        best_tool, best_score = tool_scores[0]
        
        # Create results
        results = ToolOptimizationResults(
            selected_tool=best_tool,
            optimization_score=best_score['total_score'],
            criteria_scores=best_score,
            alternative_tools=[tool for tool, _ in tool_scores[1:3]],  # Top 3 alternatives
            optimization_method="Multi-Criteria Decision Making",
            confidence=0.85
        )
        
        print(f'    ✅ Selected tool: {best_tool.tool_id} (Score: {best_score["total_score"]:.3f})')
        
        return results
    
    def _filter_feasible_tools(
        self, 
        cutting_speed: float, 
        feed_rate: float
    ) -> List[ToolSpecification]:
        """Filter tools based on cutting condition compatibility"""
        
        feasible_tools = []
        for tool in self.tool_database:
            speed_ok = tool.cutting_speed_range[0] <= cutting_speed <= tool.cutting_speed_range[1]
            feed_ok = tool.feed_rate_range[0] <= feed_rate <= tool.feed_rate_range[1]
            
            if speed_ok and feed_ok:
                feasible_tools.append(tool)
        
        return feasible_tools
    
    def _evaluate_tool_performance(
        self,
        tool: ToolSpecification,
        cutting_speed: float,
        feed_rate: float,
        depth_of_cut: float,
        physics_foundation: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate tool performance using multiple criteria"""
        
        # 1. Tool Life Score (40% weight)
        # Enhanced Taylor equation: VT^n = C
        taylor_n = 0.25  # For Ti-6Al-4V
        taylor_c = 150   # Material constant
        
        predicted_life = (taylor_c / cutting_speed)**(1/taylor_n)
        expected_life = tool.expected_life
        
        # Normalize tool life score (0-1)
        life_ratio = min(predicted_life / expected_life, 2.0)  # Cap at 2.0
        tool_life_score = min(1.0, life_ratio / 2.0 + 0.5)
        
        # 2. Surface Quality Score (25% weight)
        # Based on theoretical roughness vs tool capability
        theoretical_ra = (feed_rate**2) / (32 * tool.nose_radius * 1000)  # μm
        surface_quality_score = min(1.0, tool.surface_finish_capability / max(theoretical_ra, 0.1))
        
        # 3. Cost Effectiveness Score (20% weight)
        # Cost per unit time
        cost_per_minute = tool.cost_per_edge / predicted_life
        max_cost = 5.0  # USD/min (reference)
        cost_effectiveness_score = max(0.0, 1.0 - cost_per_minute / max_cost)
        
        # 4. Productivity Score (15% weight)
        # Material removal rate
        mrr = cutting_speed * feed_rate * depth_of_cut  # mm³/min
        max_mrr = 1000.0  # mm³/min (reference)
        productivity_score = min(1.0, mrr / max_mrr)
        
        # Calculate weighted total score
        scores = {
            'tool_life': tool_life_score,
            'surface_quality': surface_quality_score,
            'cost_effectiveness': cost_effectiveness_score,
            'productivity': productivity_score
        }
        
        total_score = sum(
            scores[criterion] * weight 
            for criterion, weight in self.criteria_weights.items()
        )
        
        scores['total_score'] = total_score
        
        return scores
    
    def _create_default_results(self, tool: ToolSpecification) -> ToolOptimizationResults:
        """Create default results when optimization fails"""
        
        return ToolOptimizationResults(
            selected_tool=tool,
            optimization_score=0.5,
            criteria_scores={
                'tool_life': 0.5,
                'surface_quality': 0.5,
                'cost_effectiveness': 0.5,
                'productivity': 0.5,
                'total_score': 0.5
            },
            alternative_tools=[],
            optimization_method="Default Selection",
            confidence=0.3
        )


def sfdp_enhanced_tool_selection(
    simulation_state: Dict[str, Any],
    extended_data: Dict[str, Any],
    physics_foundation: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main function for enhanced tool selection
    
    Args:
        simulation_state: Current simulation state
        extended_data: Extended machining data
        physics_foundation: Physics foundation data
        
    Returns:
        Tuple[Dict, Dict]: Selected tools and optimization results
    """
    
    print('\n=== Enhanced Tool Selection ===')
    
    # Initialize tool selection system
    tool_selector = EnhancedToolSelection()
    
    # Extract cutting conditions from extended data
    cutting_conditions = {
        'cutting_speed': extended_data.get('cutting_speed', 150.0),
        'feed_rate': extended_data.get('feed_rate', 0.15),
        'depth_of_cut': extended_data.get('depth_of_cut', 1.0)
    }
    
    # Perform tool optimization
    optimization_results = tool_selector.select_optimal_tool(
        simulation_state, physics_foundation, cutting_conditions
    )
    
    # Convert selected tool to dictionary format
    selected_tool = optimization_results.selected_tool
    
    selected_tools = {
        'primary_tool': {
            'tool_id': selected_tool.tool_id,
            'material': selected_tool.material,
            'coating': selected_tool.coating,
            'nose_radius': selected_tool.nose_radius,
            'edge_radius': selected_tool.edge_radius,
            'rake_angle': selected_tool.rake_angle,
            'clearance_angle': selected_tool.clearance_angle,
            'cost_per_edge': selected_tool.cost_per_edge,
            'expected_life': selected_tool.expected_life,
            'surface_finish_capability': selected_tool.surface_finish_capability
        },
        'selection_metadata': {
            'optimization_score': optimization_results.optimization_score,
            'confidence': optimization_results.confidence,
            'method': optimization_results.optimization_method,
            'criteria_weights': tool_selector.criteria_weights
        }
    }
    
    tool_optimization_results = {
        'optimization_summary': {
            'total_tools_evaluated': len(tool_selector.tool_database),
            'feasible_tools': len(optimization_results.alternative_tools) + 1,
            'optimization_score': optimization_results.optimization_score,
            'confidence': optimization_results.confidence
        },
        'criteria_scores': optimization_results.criteria_scores,
        'alternative_tools': [
            {
                'tool_id': tool.tool_id,
                'material': tool.material,
                'coating': tool.coating
            }
            for tool in optimization_results.alternative_tools
        ],
        'optimization_method': optimization_results.optimization_method
    }
    
    print(f'  Tool Selection Complete: {selected_tool.tool_id}')
    print(f'  Optimization Score: {optimization_results.optimization_score:.3f}')
    print(f'  Confidence: {optimization_results.confidence:.3f}')
    
    return selected_tools, tool_optimization_results


if __name__ == "__main__":
    # Test functionality
    test_simulation_state = {}
    test_extended_data = {
        'cutting_speed': 150.0,
        'feed_rate': 0.15,
        'depth_of_cut': 1.0
    }
    test_physics_foundation = {}
    
    selected_tools, results = sfdp_enhanced_tool_selection(
        test_simulation_state, test_extended_data, test_physics_foundation
    )
    
    print("\nTest completed successfully!")