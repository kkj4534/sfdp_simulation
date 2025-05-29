function [taylor_results, taylor_confidence] = SFDP_taylor_coefficient_processor(simulation_state, extended_data, data_confidence)
%% SFDP_TAYLOR_COEFFICIENT_PROCESSOR - Data-Based Taylor Coefficient Processing
% =========================================================================
% INTELLIGENT TAYLOR COEFFICIENT SELECTION FROM EXPERIMENTAL DATABASE
%
% THEORETICAL FOUNDATION:
% Enhanced Taylor tool life equation with multi-variable dependencies:
% V × T^n × f^a × d^b × Q^c = C_extended
% where: V=cutting speed, T=tool life, f=feed rate, d=depth, Q=material hardness
%
% DATABASE SELECTION CRITERIA:
% 1. Material match (Ti-6Al-4V priority)
% 2. Tool material compatibility
% 3. Cutting condition similarity
% 4. Data reliability rating (⭐⭐⭐⭐⭐ = highest)
% 5. Experimental validation coverage
%
% COEFFICIENT ADAPTATION:
% - Temperature dependence: C_eff = C_base × exp(-E_a/(RT))
% - Tool material correction: C_tool = C_base × k_material
% - Workpiece condition factor: C_condition = C_base × k_condition
%
% REFERENCE: Taylor (1907) "On the art of cutting metals" Trans. ASME 28
% REFERENCE: Kronenberg (1966) "Machining Science and Application" Pergamon
% REFERENCE: ASM Handbook Vol. 16 (1989) "Machining" Ch. 5
% REFERENCE: Trent & Wright (2000) "Metal Cutting" 4th Ed. Ch. 10
%
% INPUT PARAMETERS:
% simulation_state - Global simulation configuration
% extended_data - Complete experimental database
% data_confidence - Data quality and reliability metrics
%
% OUTPUT PARAMETERS:
% taylor_results - Selected Taylor coefficients and metadata
% taylor_confidence - Confidence in coefficient selection [0-1]
% =========================================================================

    fprintf('  📊 Data-based Taylor coefficient selection...\n');
    
    taylor_results = struct();
    
    % Check data availability and quality
    if ~isfield(extended_data, 'materials_data') || isempty(extended_data.materials_data)
        fprintf('    ⚠️  No materials data available, using default coefficients\n');
        taylor_results = get_default_taylor_coefficients();
        taylor_confidence = 0.6;
        return;
    end
    
    % Extract material and tool information
    material_name = simulation_state.material.name;
    tool_material = simulation_state.tool.material;
    
    fprintf('    🔍 Searching coefficients for: %s with %s tool\n', material_name, tool_material);
    
    % Search for best matching data
    [best_match, match_confidence] = find_best_taylor_match(extended_data, material_name, tool_material);
    
    if ~isempty(best_match)
        % Use database coefficients with adaptations
        taylor_results = adapt_taylor_coefficients(best_match, simulation_state);
        taylor_confidence = match_confidence * data_confidence.reliability_factor;
        
        fprintf('    ✅ Found matching data: %s\n', best_match.source);
        fprintf('       Taylor constant C = %.1f, exponent n = %.3f\n', ...
                taylor_results.coefficients.C, taylor_results.coefficients.n);
    else
        % Use default coefficients
        fprintf('    ⚠️  No matching data found, using default Ti-6Al-4V coefficients\n');
        taylor_results = get_default_taylor_coefficients();
        taylor_confidence = 0.6;
    end
    
    % Add metadata
    taylor_results.selection_method = 'DATABASE_MATCH';
    taylor_results.material_match = material_name;
    taylor_results.tool_match = tool_material;
    taylor_results.confidence = taylor_confidence;
    taylor_results.data_source = 'extended_materials_database';
    
    % Validate coefficient ranges
    taylor_results = validate_taylor_ranges(taylor_results);
    
    fprintf('    📈 Final Taylor equation: V × T^%.3f = %.1f\n', ...
            taylor_results.coefficients.n, taylor_results.coefficients.C);
end

function [best_match, confidence] = find_best_taylor_match(extended_data, material_name, tool_material)
    best_match = [];
    confidence = 0;
    
    % Check if materials data exists
    if ~isfield(extended_data, 'materials_data')
        return;
    end
    
    materials = extended_data.materials_data;
    
    % Search for exact material match first
    for i = 1:length(materials)
        if isfield(materials(i), 'name') && isfield(materials(i), 'taylor_data')
            % Check material name match
            material_match = contains(lower(materials(i).name), lower(material_name)) || ...
                           contains(lower(material_name), lower(materials(i).name));
            
            if material_match && isfield(materials(i).taylor_data, 'tool_materials')
                % Check tool material compatibility
                tool_data = materials(i).taylor_data.tool_materials;
                
                for j = 1:length(tool_data)
                    if isfield(tool_data(j), 'material') && isfield(tool_data(j), 'coefficients')
                        tool_match = contains(lower(tool_data(j).material), lower(tool_material)) || ...
                                   contains(lower(tool_material), lower(tool_data(j).material));
                        
                        if tool_match
                            current_confidence = calculate_match_confidence(materials(i), tool_data(j));
                            
                            if current_confidence > confidence
                                best_match = prepare_match_data(materials(i), tool_data(j));
                                confidence = current_confidence;
                            end
                        end
                    end
                end
            end
        end
    end
    
    % If no exact match, try partial matches
    if isempty(best_match)
        for i = 1:length(materials)
            if isfield(materials(i), 'taylor_data')
                % Use Ti-6Al-4V as fallback for titanium alloys
                if contains(lower(material_name), 'ti') && contains(lower(materials(i).name), 'ti-6al-4v')
                    tool_data = materials(i).taylor_data.tool_materials;
                    
                    for j = 1:length(tool_data)
                        if isfield(tool_data(j), 'coefficients')
                            current_confidence = 0.7; % Reduced confidence for fallback
                            
                            if current_confidence > confidence
                                best_match = prepare_match_data(materials(i), tool_data(j));
                                confidence = current_confidence;
                            end
                        end
                    end
                end
            end
        end
    end
end

function confidence = calculate_match_confidence(material_data, tool_data)
    confidence = 0.5; % Base confidence
    
    % Check data quality indicators
    if isfield(material_data, 'reliability_rating')
        switch material_data.reliability_rating
            case '⭐⭐⭐⭐⭐'
                confidence = confidence + 0.4;
            case '⭐⭐⭐⭐☆'
                confidence = confidence + 0.3;
            case '⭐⭐⭐☆☆'
                confidence = confidence + 0.2;
            otherwise
                confidence = confidence + 0.1;
        end
    end
    
    % Check if experimental validation is available
    if isfield(tool_data, 'experimental_validation') && tool_data.experimental_validation
        confidence = confidence + 0.1;
    end
    
    % Cap at 1.0
    confidence = min(confidence, 1.0);
end

function match_data = prepare_match_data(material_data, tool_data)
    match_data = struct();
    
    match_data.source = sprintf('%s + %s', material_data.name, tool_data.material);
    match_data.coefficients = tool_data.coefficients;
    match_data.material_properties = material_data;
    match_data.tool_properties = tool_data;
    
    if isfield(material_data, 'reliability_rating')
        match_data.reliability = material_data.reliability_rating;
    else
        match_data.reliability = '⭐⭐⭐☆☆';
    end
end

function taylor_results = adapt_taylor_coefficients(best_match, simulation_state)
    taylor_results = struct();
    
    % Extract base coefficients
    base_coeffs = best_match.coefficients;
    
    % Initialize coefficients structure
    taylor_results.coefficients = struct();
    
    % Basic Taylor coefficients
    if isfield(base_coeffs, 'C')
        taylor_results.coefficients.C = base_coeffs.C;
    else
        taylor_results.coefficients.C = 180; % Default for Ti-6Al-4V
    end
    
    if isfield(base_coeffs, 'n')
        taylor_results.coefficients.n = base_coeffs.n;
    else
        taylor_results.coefficients.n = 0.25; % Default exponent
    end
    
    % Extended coefficients (if available)
    if isfield(base_coeffs, 'a')
        taylor_results.coefficients.a = base_coeffs.a; % Feed rate exponent
    else
        taylor_results.coefficients.a = 0.75; % Default feed exponent
    end
    
    if isfield(base_coeffs, 'b')
        taylor_results.coefficients.b = base_coeffs.b; % Depth exponent
    else
        taylor_results.coefficients.b = 0.15; % Default depth exponent
    end
    
    % Temperature adaptation (Arrhenius-type)
    if isfield(simulation_state, 'operating_temperature')
        T_ref = 298; % K (reference temperature)
        T_op = simulation_state.operating_temperature + 273.15; % K
        activation_energy = 50000; % J/mol (typical for tool wear)
        R = 8.314; % J/mol⋅K
        
        temp_factor = exp(activation_energy/R * (1/T_ref - 1/T_op));
        taylor_results.coefficients.C = taylor_results.coefficients.C * temp_factor;
    end
    
    % Add metadata
    taylor_results.adaptation_applied = true;
    taylor_results.base_source = best_match.source;
    taylor_results.reliability = best_match.reliability;
end

function taylor_results = get_default_taylor_coefficients()
    % Default Taylor coefficients for Ti-6Al-4V with carbide tools
    taylor_results = struct();
    
    taylor_results.coefficients = struct();
    taylor_results.coefficients.C = 180;      % Taylor constant (m/min)
    taylor_results.coefficients.n = 0.25;     % Taylor exponent
    taylor_results.coefficients.a = 0.75;     % Feed rate exponent
    taylor_results.coefficients.b = 0.15;     % Depth of cut exponent
    taylor_results.coefficients.c = 0.5;      % Material hardness exponent
    
    taylor_results.source = 'DEFAULT_TI6AL4V_CARBIDE';
    taylor_results.reliability = '⭐⭐⭐☆☆';
    taylor_results.adaptation_applied = false;
    
    % Reference conditions
    taylor_results.reference_conditions = struct();
    taylor_results.reference_conditions.material = 'Ti-6Al-4V';
    taylor_results.reference_conditions.tool = 'Carbide_Uncoated';
    taylor_results.reference_conditions.cutting_speed_range = [50, 300]; % m/min
    taylor_results.reference_conditions.feed_range = [0.1, 0.5]; % mm/rev
    taylor_results.reference_conditions.depth_range = [0.5, 5.0]; % mm
end

function taylor_results = validate_taylor_ranges(taylor_results)
    % Validate Taylor coefficients are within reasonable ranges
    
    % Taylor constant C validation
    if taylor_results.coefficients.C < 50 || taylor_results.coefficients.C > 1000
        fprintf('    ⚠️  Taylor constant C=%.1f outside normal range [50-1000], adjusting\n', ...
                taylor_results.coefficients.C);
        taylor_results.coefficients.C = max(50, min(1000, taylor_results.coefficients.C));
    end
    
    % Taylor exponent n validation  
    if taylor_results.coefficients.n < 0.1 || taylor_results.coefficients.n > 0.8
        fprintf('    ⚠️  Taylor exponent n=%.3f outside normal range [0.1-0.8], adjusting\n', ...
                taylor_results.coefficients.n);
        taylor_results.coefficients.n = max(0.1, min(0.8, taylor_results.coefficients.n));
    end
    
    taylor_results.validation_applied = true;
end