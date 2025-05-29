function [selected_tools, tool_optimization_results] = SFDP_enhanced_tool_selection(simulation_state, extended_data, physics_foundation)
%% SFDP_ENHANCED_TOOL_SELECTION - Multi-Criteria Tool Selection Module
% =========================================================================
% COMPREHENSIVE TOOL SELECTION WITH USER-FRIENDLY INTERFACE
%
% THEORETICAL FOUNDATION:
% Multi-criteria decision making (MCDM) for optimal tool selection considering:
% 1. TOOL LIFE: Taylor equation predictions and experimental data
% 2. SURFACE QUALITY: Ra requirements and achievable finish
% 3. PRODUCTIVITY: Material removal rate and machining time
% 4. ECONOMICS: Tool cost per part and operational efficiency
%
% SELECTION CRITERIA:
% - Material compatibility (Ti-6Al-4V specific recommendations)
% - Cutting speed range suitability
% - Feed rate and depth capabilities
% - Tool life expectation
% - Surface finish achievability
%
% REFERENCE: Saaty (1980) "Analytic Hierarchy Process" McGraw-Hill
% REFERENCE: Hwang & Yoon (1981) "Multiple Attribute Decision Making" Springer
% REFERENCE: ASM Handbook Vol. 16 (1989) "Machining" Tool Selection Guidelines
%
% INPUT PARAMETERS:
% simulation_state - Global simulation configuration
% extended_data - Material and cutting data database
% physics_foundation - Physical property foundation
%
% OUTPUT PARAMETERS:
% selected_tools - Selected tool specifications structure
% tool_optimization_results - Selection rationale and performance metrics
% =========================================================================

    fprintf('  🔧 Enhanced Multi-Criteria Tool Selection...\n');
    
    % Initialize tool database for Ti-6Al-4V
    tool_database = initialize_tool_database();
    
    % Display available tool options
    fprintf('  Available tool options for Ti-6Al-4V machining:\n');
    for i = 1:length(tool_database)
        fprintf('    %d. %s\n', i, tool_database(i).description);
    end
    
    % Check if GWO automatic optimization is enabled
    try
        user_config = SFDP_user_config();
        if user_config.external_toolboxes.gwo.enabled && exist('GWO', 'file')
            fprintf('  🤖 GWO automatic optimization enabled. Running tool optimization...\n');
            [tool_choice, optimization_results] = optimize_tool_selection_gwo(tool_database, simulation_state, physics_foundation);
            fprintf('  ✅ GWO optimization complete. Selected tool: %s\n', tool_database(tool_choice).description);
        else
            % Manual user selection
            fprintf('  Please select tool number (1-%d): ', length(tool_database));
            tool_choice = input('');
            optimization_results = struct('method', 'manual', 'reason', 'GWO not available or disabled');
        end
    catch
        % Fallback to manual selection
        fprintf('  Please select tool number (1-%d): ', length(tool_database));
        tool_choice = input('');
        optimization_results = struct('method', 'manual', 'reason', 'GWO configuration error');
    end
    
    if isempty(tool_choice) || tool_choice < 1 || tool_choice > length(tool_database)
        fprintf('  ⚠️  Invalid selection, using default tool (Carbide Insert)\n');
            tool_choice = 1;
        end
    catch
        fprintf('  ⚠️  Input error, using default tool (Carbide Insert)\n');
        tool_choice = 1;
    end
    
    % Select tool based on user choice
    selected_tools = tool_database(tool_choice);
    
    % Calculate tool performance metrics
    tool_optimization_results = struct();
    tool_optimization_results.selection_method = 'USER_INPUT_MCDM';
    tool_optimization_results.tool_index = tool_choice;
    tool_optimization_results.expected_life = selected_tools.taylor_constant / 150; % Assuming 150 m/min
    tool_optimization_results.surface_capability = selected_tools.surface_finish_capability;
    tool_optimization_results.cost_per_part = selected_tools.cost_per_edge / tool_optimization_results.expected_life;
    tool_optimization_results.productivity_index = selected_tools.max_mrr;
    
    fprintf('  ✅ Tool selected: %s\n', selected_tools.description);
    fprintf('      Expected life: %.1f minutes\n', tool_optimization_results.expected_life);
    fprintf('      Surface finish: Ra %.2f μm achievable\n', selected_tools.surface_finish_capability);
end

function tool_database = initialize_tool_database()
    % Initialize comprehensive tool database for Ti-6Al-4V
    
    tool_database = [];
    
    % Tool 1: Carbide Insert (Standard)
    tool_database(1).description = 'Carbide Insert (CNMG120408, Uncoated) - Standard Choice';
    tool_database(1).material = 'Carbide_Uncoated';
    tool_database(1).geometry = 'CNMG120408';
    tool_database(1).taylor_constant = 180;
    tool_database(1).taylor_exponent = 0.25;
    tool_database(1).nose_radius = 0.8; % mm
    tool_database(1).rake_angle = -6; % degrees
    tool_database(1).relief_angle = 6; % degrees
    tool_database(1).surface_finish_capability = 1.6; % Ra μm
    tool_database(1).max_mrr = 15; % cm³/min
    tool_database(1).cost_per_edge = 8.5; % USD
    tool_database(1).recommended_speed_range = [80, 250]; % m/min
    
    % Tool 2: Coated Carbide (TiAlN)
    tool_database(2).description = 'TiAlN Coated Carbide (CNMG120408, TiAlN) - High Performance';
    tool_database(2).material = 'Carbide_TiAlN';
    tool_database(2).geometry = 'CNMG120408';
    tool_database(2).taylor_constant = 280;
    tool_database(2).taylor_exponent = 0.22;
    tool_database(2).nose_radius = 0.8;
    tool_database(2).rake_angle = -3;
    tool_database(2).relief_angle = 7;
    tool_database(2).surface_finish_capability = 1.2;
    tool_database(2).max_mrr = 25;
    tool_database(2).cost_per_edge = 12.5;
    tool_database(2).recommended_speed_range = [120, 350];
    
    % Tool 3: CBN (Cubic Boron Nitride)
    tool_database(3).description = 'CBN Insert (CNGA120408, CBN) - Ultra High Speed';
    tool_database(3).material = 'CBN';
    tool_database(3).geometry = 'CNGA120408';
    tool_database(3).taylor_constant = 450;
    tool_database(3).taylor_exponent = 0.18;
    tool_database(3).nose_radius = 0.8;
    tool_database(3).rake_angle = -5;
    tool_database(3).relief_angle = 5;
    tool_database(3).surface_finish_capability = 0.8;
    tool_database(3).max_mrr = 40;
    tool_database(3).cost_per_edge = 45.0;
    tool_database(3).recommended_speed_range = [200, 600];
    
    % Tool 4: PCD (Polycrystalline Diamond)
    tool_database(4).description = 'PCD Insert (CCMG120408, PCD) - Ultra Precision';
    tool_database(4).material = 'PCD';
    tool_database(4).geometry = 'CCMG120408';
    tool_database(4).taylor_constant = 650;
    tool_database(4).taylor_exponent = 0.15;
    tool_database(4).nose_radius = 0.4;
    tool_database(4).rake_angle = 0;
    tool_database(4).relief_angle = 8;
    tool_database(4).surface_finish_capability = 0.4;
    tool_database(4).max_mrr = 50;
    tool_database(4).cost_per_edge = 125.0;

end

function [optimal_tool_index, optimization_results] = optimize_tool_selection_gwo(tool_database, simulation_state, physics_foundation)
%% GWO-BASED TOOL OPTIMIZATION
% Uses Grey Wolf Optimizer to find optimal tool selection based on:
% - Tool life maximization
% - Surface quality optimization  
% - Cost minimization
% - Productivity maximization

    % Optimization parameters
    nvars = 1;  % Single variable: tool index (discrete)
    lb = 1;     % Lower bound: first tool
    ub = length(tool_database);  % Upper bound: last tool
    
    % GWO parameters
    max_iterations = 50;
    search_agents = 10;
    
    try
        % Define objective function for multi-criteria optimization
        objective_function = @(x) evaluate_tool_performance(round(x), tool_database, simulation_state, physics_foundation);
        
        % Run GWO optimization
        [best_position, best_score, convergence_curve] = GWO(objective_function, nvars, search_agents, max_iterations, lb, ub);
        
        % Extract results
        optimal_tool_index = round(best_position);
        
        % Ensure valid tool index
        if optimal_tool_index < 1
            optimal_tool_index = 1;
        elseif optimal_tool_index > length(tool_database)
            optimal_tool_index = length(tool_database);
        end
        
        % Generate optimization report
        optimization_results = struct();
        optimization_results.method = 'GWO';
        optimization_results.best_score = best_score;
        optimization_results.convergence_curve = convergence_curve;
        optimization_results.iterations = max_iterations;
        optimization_results.search_agents = search_agents;
        optimization_results.optimization_criteria = {
            'Tool life maximization (40%)',
            'Surface quality optimization (25%)', 
            'Cost minimization (20%)',
            'Productivity maximization (15%)'
        };
        
    catch ME
        fprintf('  ⚠️  GWO optimization failed: %s\n', ME.message);
        fprintf('  Falling back to default tool selection.\n');
        
        % Fallback to simple scoring
        optimal_tool_index = 1;  % Default to first tool
        optimization_results = struct();
        optimization_results.method = 'fallback';
        optimization_results.error = ME.message;
    end

end

function score = evaluate_tool_performance(tool_index, tool_database, simulation_state, physics_foundation)
%% MULTI-CRITERIA TOOL PERFORMANCE EVALUATION
% Evaluates tool performance using weighted criteria
% Lower score = better performance (for minimization)

    % Validate tool index
    if tool_index < 1 || tool_index > length(tool_database)
        score = 1e6;  % Penalty for invalid index
        return;
    end
    
    tool = tool_database(tool_index);
    
    % Extract cutting conditions from simulation state
    cutting_speed = simulation_state.cutting_conditions.cutting_speed;
    feed_rate = simulation_state.cutting_conditions.feed_rate;
    depth_of_cut = simulation_state.cutting_conditions.depth_of_cut;
    
    %% CRITERION 1: Tool Life (40% weight) - Maximize
    % Use Taylor equation: VT^n = C
    predicted_tool_life = tool.taylor_constant / (cutting_speed^tool.taylor_exponent);
    
    % Normalize to 0-1 scale (longer life = better)
    max_expected_life = 120;  % minutes
    tool_life_score = min(predicted_tool_life / max_expected_life, 1.0);
    tool_life_penalty = 1.0 - tool_life_score;  % Convert to penalty (lower is better)
    
    %% CRITERION 2: Surface Quality (25% weight) - Maximize  
    % Better surface finish capability = lower Ra
    target_surface_finish = 1.6;  % μm Ra target
    surface_quality_ratio = tool.surface_finish_capability / target_surface_finish;
    
    if surface_quality_ratio <= 1.0
        surface_quality_penalty = 0.1 * surface_quality_ratio;  % Reward better than target
    else
        surface_quality_penalty = 0.1 + 0.9 * (surface_quality_ratio - 1.0);  % Penalize worse than target
    end
    
    %% CRITERION 3: Cost (20% weight) - Minimize
    % Cost per part estimation
    parts_per_edge = predicted_tool_life / (depth_of_cut / feed_rate);  % Rough estimation
    cost_per_part = tool.cost_per_edge / max(parts_per_edge, 1);
    
    % Normalize cost (typical range 0.5-20 USD per part)
    max_expected_cost = 20.0;
    cost_penalty = min(cost_per_part / max_expected_cost, 1.0);
    
    %% CRITERION 4: Productivity (15% weight) - Maximize
    % Material Removal Rate (MRR)
    calculated_mrr = cutting_speed * feed_rate * depth_of_cut * 1000;  % mm³/min
    productivity_ratio = min(calculated_mrr / tool.max_mrr, 1.0);
    productivity_penalty = 1.0 - productivity_ratio;
    
    %% WEIGHTED COMBINATION
    weights = [0.40, 0.25, 0.20, 0.15];  % [tool_life, surface_quality, cost, productivity]
    penalties = [tool_life_penalty, surface_quality_penalty, cost_penalty, productivity_penalty];
    
    % Final score (lower is better for GWO minimization)
    score = sum(weights .* penalties);
    
    % Add penalty for extreme operating conditions
    if cutting_speed > 200 || feed_rate > 0.4
        score = score + 0.1;  % Penalty for aggressive conditions
    end

end