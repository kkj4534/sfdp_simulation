function config = SFDP_user_config()
%% SFDP_USER_CONFIG - User Configuration File for SFDP v17.3 Framework
% =========================================================================
% FUNCTION PURPOSE:
% Centralized configuration file for user-specific settings including
% toolbox paths, MATLAB configurations, and simulation preferences
%
% CONFIGURATION CATEGORIES:
% 1. MATLAB and Toolbox Paths
% 2. External Toolbox Configurations
% 3. Data File Locations
% 4. Simulation Parameters
% 5. Performance Settings
% 6. Output Preferences
% 7. Advanced Settings
%
% USAGE:
% This file should be customized by each user according to their system
% configuration. Default values are provided for common setups.
%
% IMPORTANT NOTES:
% - All paths should use forward slashes (/) or MATLAB-compatible format
% - Ensure all specified paths exist before running simulations
% - This configuration will be automatically validated on startup
%
% Reference: MATLAB configuration management best practices
% Reference: Scientific computing software configuration standards
%
% Author: SFDP Research Team
% Date: May 2025
% License: Academic Research Use Only
% =========================================================================

    fprintf('📋 Loading SFDP v17.3 User Configuration...\n');
    
    config = struct();
    config.version = 'v17.3_UserConfig';
    config.creation_date = datestr(now);
    config.last_modified = datestr(now);
    
    %% ========================================================================
    %% SECTION 1: MATLAB AND TOOLBOX PATHS
    %% ========================================================================
    
    % MATLAB Installation Paths (automatically detected, but can be overridden)
    config.matlab = struct();
    config.matlab.version = version('-release');
    config.matlab.installation_path = matlabroot;
    config.matlab.minimum_required_version = 'R2018b';
    
    % Core MATLAB Toolbox Requirements
    config.matlab.required_toolboxes = {
        'Symbolic Math Toolbox';
        'Curve Fitting Toolbox';
        'Statistics and Machine Learning Toolbox';
        'Optimization Toolbox';
        'Partial Differential Equation Toolbox'
    };
    
    % Optional but recommended toolboxes
    config.matlab.optional_toolboxes = {
        'Parallel Computing Toolbox';
        'Image Processing Toolbox';
        'Signal Processing Toolbox'
    };
    
    %% ========================================================================
    %% SECTION 2: EXTERNAL TOOLBOX CONFIGURATIONS
    %% ========================================================================
    
    config.external_toolboxes = struct();
    
    % GIBBON FEA Toolbox Configuration
    % Download from: https://github.com/gibbonCode/GIBBON
    config.external_toolboxes.gibbon = struct();
    config.external_toolboxes.gibbon.enabled = true;
    config.external_toolboxes.gibbon.auto_detect = true;
    
    % Common installation paths (the system will try these automatically)
    config.external_toolboxes.gibbon.search_paths = {
        './toolboxes/GIBBON';               % Local project directory
        '~/GIBBON';                         % User home directory
        'C:/GIBBON';                        % Windows C drive
        'D:/GIBBON';                        % Windows D drive
        '/opt/GIBBON';                      % Linux/Mac system directory
        '~/Documents/MATLAB/GIBBON'         % MATLAB user directory
    };
    
    % Manual path specification (set this if auto-detection fails)
    config.external_toolboxes.gibbon.manual_path = '';  % e.g., 'C:/MyTools/GIBBON'
    
    % FEATool Multiphysics Configuration
    % Download from: https://www.featool.com/
    config.external_toolboxes.featool = struct();
    config.external_toolboxes.featool.enabled = true;
    config.external_toolboxes.featool.auto_detect = true;
    config.external_toolboxes.featool.search_paths = {
        './toolboxes/FEATool';
        '~/FEATool';
        'C:/FEATool';
        'D:/FEATool';
        '/opt/FEATool';
        '~/Documents/MATLAB/FEATool'
    };
    config.external_toolboxes.featool.manual_path = '';
    
    % CFDTool Configuration
    % Download from: https://www.featool.com/cfdtool
    config.external_toolboxes.cfdtool = struct();
    config.external_toolboxes.cfdtool.enabled = true;
    config.external_toolboxes.cfdtool.auto_detect = true;
    config.external_toolboxes.cfdtool.search_paths = {
        './toolboxes/CFDTool';
        '~/CFDTool';
        'C:/CFDTool';
        'D:/CFDTool';
        '/opt/CFDTool';
        '~/Documents/MATLAB/CFDTool'
    };
    config.external_toolboxes.cfdtool.manual_path = '';
    
    % Iso2Mesh Configuration
    % Download from: http://iso2mesh.sourceforge.net/
    config.external_toolboxes.iso2mesh = struct();
    config.external_toolboxes.iso2mesh.enabled = true;
    config.external_toolboxes.iso2mesh.auto_detect = true;
    config.external_toolboxes.iso2mesh.search_paths = {
        './toolboxes/iso2mesh';
        '~/iso2mesh';
        'C:/iso2mesh';
        'D:/iso2mesh';
        '/opt/iso2mesh';
        '~/Documents/MATLAB/iso2mesh'
    };
    config.external_toolboxes.iso2mesh.manual_path = '';
    
    % Grey Wolf Optimizer Configuration
    % Download from: https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer
    config.external_toolboxes.gwo = struct();
    config.external_toolboxes.gwo.enabled = true;
    config.external_toolboxes.gwo.auto_detect = true;
    config.external_toolboxes.gwo.search_paths = {
        './toolboxes/GWO';
        '~/GWO';
        'C:/GWO';
        'D:/GWO';
        '/opt/GWO';
        '~/Documents/MATLAB/GWO'
    };
    config.external_toolboxes.gwo.manual_path = '';
    
    %% ========================================================================
    %% SECTION 3: DATA FILE LOCATIONS
    %% ========================================================================
    
    config.data_locations = struct();
    
    % Base directory for all SFDP data
    config.data_locations.base_directory = './SFDP_6Layer_v17_3';
    
    % Extended dataset locations
    config.data_locations.extended_data = struct();
    config.data_locations.extended_data.experiments = 'extended_data/extended_validation_experiments.csv';
    config.data_locations.extended_data.taylor_coefficients = 'extended_data/taylor_coefficients_csv.csv';
    config.data_locations.extended_data.materials = 'extended_data/extended_materials_csv.csv';
    config.data_locations.extended_data.tools = 'extended_data/extended_tool_specifications.csv';
    config.data_locations.extended_data.conditions = 'extended_data/extended_machining_conditions.csv';
    config.data_locations.extended_data.targets = 'extended_data/extended_validation_targets.csv';
    
    % Cache and temporary file locations
    config.data_locations.cache_directory = 'physics_cache';
    config.data_locations.temp_directory = 'temp';
    config.data_locations.logs_directory = 'adaptive_logs';
    
    % Output file locations
    config.data_locations.output_directory = 'output';
    config.data_locations.figures_directory = 'figures';
    config.data_locations.reports_directory = 'reports';
    config.data_locations.validation_directory = 'validation';
    
    %% ========================================================================
    %% SECTION 4: SIMULATION PARAMETERS
    %% ========================================================================
    
    config.simulation = struct();
    
    % Default material for simulations
    config.simulation.default_material = 'Ti6Al4V';
    
    % Default cutting conditions (can be overridden at runtime)
    config.simulation.default_conditions = struct();
    config.simulation.default_conditions.cutting_speed = 120;     % m/min
    config.simulation.default_conditions.feed_rate = 0.1;        % mm/rev
    config.simulation.default_conditions.depth_of_cut = 1.0;     % mm
    config.simulation.default_conditions.coolant_type = 'FLOOD'; % FLOOD, MQL, DRY
    
    % Simulation accuracy and convergence settings
    config.simulation.accuracy = struct();
    config.simulation.accuracy.convergence_tolerance = 1e-6;
    config.simulation.accuracy.max_iterations = 1000;
    config.simulation.accuracy.cfl_safety_factor = 0.35;
    config.simulation.accuracy.mesh_refinement_threshold = 0.1;
    
    % Physical bounds for validation
    config.simulation.physical_bounds = struct();
    config.simulation.physical_bounds.temperature_range = [25, 800];    % °C
    config.simulation.physical_bounds.wear_range = [0.001, 1.0];        % mm
    config.simulation.physical_bounds.roughness_range = [0.1, 10];      % μm
    config.simulation.physical_bounds.force_range = [10, 5000];         % N
    config.simulation.physical_bounds.stress_range = [1e6, 2e9];        % Pa
    
    %% ========================================================================
    %% SECTION 5: ADAPTIVE KALMAN FILTER SETTINGS
    %% ========================================================================
    
    config.kalman = struct();
    
    % Kalman filter operation mode
    config.kalman.enabled = true;
    config.kalman.adaptation_mode = 'VALIDATION_DRIVEN'; % FIXED, ADAPTIVE, VALIDATION_DRIVEN
    
    % Kalman gain parameters
    config.kalman.gain_bounds = [0.05, 0.35];           % 5-35% correction range
    config.kalman.base_gain = 0.15;                     % Base Kalman gain
    config.kalman.adaptation_rate = 0.1;                % Rate of gain adaptation
    
    % Innovation and validation parameters
    config.kalman.innovation_threshold = 0.1;           % Threshold for innovation detection
    config.kalman.validation_weight = 0.3;              % Weight given to validation performance
    config.kalman.physics_weight = 0.7;                 % Weight given to physics confidence
    config.kalman.history_length = 20;                  % Number of past innovations to remember
    config.kalman.convergence_tolerance = 1e-4;         % Convergence criterion for gain adaptation
    
    %% ========================================================================
    %% SECTION 6: PERFORMANCE SETTINGS
    %% ========================================================================
    
    config.performance = struct();
    
    % Memory management
    config.performance.memory_limit = 8e9;              % Memory limit in bytes (8GB)
    config.performance.auto_garbage_collection = true;   % Enable automatic garbage collection
    config.performance.memory_monitoring = true;         % Monitor memory usage
    
    % Parallel computing settings
    config.performance.parallel_computing = struct();
    config.performance.parallel_computing.enabled = false; % Enable parallel computing
    config.performance.parallel_computing.auto_detect_workers = true;
    config.performance.parallel_computing.max_workers = 4;  % Maximum number of parallel workers
    config.performance.parallel_computing.prefer_local = true; % Prefer local parallel pool
    
    % Execution time limits
    config.performance.time_limits = struct();
    config.performance.time_limits.total_simulation = 3600;    % Total simulation time limit (seconds)
    config.performance.time_limits.layer_calculation = 600;   % Per-layer time limit (seconds)
    config.performance.time_limits.single_calculation = 120;  % Single calculation time limit (seconds)
    
    % Performance monitoring
    config.performance.monitoring = struct();
    config.performance.monitoring.enabled = true;
    config.performance.monitoring.report_interval = 10;       % Progress report interval (seconds)
    config.performance.monitoring.detailed_timing = false;    % Enable detailed timing analysis
    config.performance.monitoring.memory_snapshots = false;   % Enable memory usage snapshots
    
    %% ========================================================================
    %% SECTION 7: OUTPUT AND REPORTING PREFERENCES
    %% ========================================================================
    
    config.output = struct();
    
    % Console output settings
    config.output.console = struct();
    config.output.console.verbosity_level = 'NORMAL';         % QUIET, NORMAL, VERBOSE, DEBUG
    config.output.console.progress_updates = true;            % Show progress updates
    config.output.console.layer_details = true;               % Show layer execution details
    config.output.console.timing_information = true;          % Show timing information
    config.output.console.confidence_scores = true;           % Show confidence scores
    
    % File output settings
    config.output.files = struct();
    config.output.files.save_results = true;                  % Save results to files
    config.output.files.save_intermediate = false;            % Save intermediate layer results
    config.output.files.save_logs = true;                     % Save execution logs
    config.output.files.compression = true;                   % Compress output files
    
    % Figure and plot settings
    config.output.figures = struct();
    config.output.figures.generate_plots = true;              % Generate result plots
    config.output.figures.save_figures = true;                % Save figures to files
    config.output.figures.figure_format = 'png';              % Figure format (png, jpg, eps, pdf)
    config.output.figures.figure_resolution = 300;            % DPI for saved figures
    config.output.figures.show_interactive = false;           % Show interactive figures
    
    % Report generation settings
    config.output.reports = struct();
    config.output.reports.generate_summary = true;            % Generate summary report
    config.output.reports.generate_detailed = false;          % Generate detailed technical report
    config.output.reports.include_validation = true;          % Include validation analysis
    config.output.reports.include_genealogy = false;          % Include calculation genealogy
    config.output.reports.report_format = 'txt';              % Report format (txt, html, pdf)
    
    %% ========================================================================
    %% SECTION 8: ADVANCED SETTINGS
    %% ========================================================================
    
    config.advanced = struct();
    
    % Error handling and recovery
    config.advanced.error_handling = struct();
    config.advanced.error_handling.auto_recovery = true;      % Enable automatic error recovery
    config.advanced.error_handling.max_retries = 3;           % Maximum retry attempts
    config.advanced.error_handling.fallback_enabled = true;   % Enable fallback mechanisms
    config.advanced.error_handling.emergency_mode = true;     % Enable emergency operation mode
    
    % Quality assurance settings
    config.advanced.quality_assurance = struct();
    config.advanced.quality_assurance.strict_validation = false; % Enable strict validation mode
    config.advanced.quality_assurance.bounds_checking = true;    % Enable physical bounds checking
    config.advanced.quality_assurance.consistency_checks = true; % Enable consistency checking
    config.advanced.quality_assurance.anomaly_detection = true;  % Enable anomaly detection
    
    % Data quality settings
    config.advanced.data_quality = struct();
    config.advanced.data_quality.minimum_confidence = 0.3;    % Minimum acceptable data confidence
    config.advanced.data_quality.outlier_detection = true;    % Enable outlier detection
    config.advanced.data_quality.statistical_validation = true; % Enable statistical validation
    config.advanced.data_quality.source_verification = false;   % Enable data source verification
    
    % Experimental features (use with caution)
    config.advanced.experimental = struct();
    config.advanced.experimental.adaptive_mesh = false;       % Enable adaptive mesh refinement
    config.advanced.experimental.ml_enhancement = false;      % Enable ML-based enhancements
    config.advanced.experimental.gpu_acceleration = false;    % Enable GPU acceleration
    config.advanced.experimental.distributed_computing = false; % Enable distributed computing
    
    %% ========================================================================
    %% SECTION 9: USER-SPECIFIC CUSTOMIZATIONS
    %% ========================================================================
    
    config.user = struct();
    
    % User identification (for logging and traceability)
    config.user.name = getenv('USERNAME');                    % Automatically detect username
    config.user.institution = 'SFDP Research Team';           % User institution
    config.user.email = '';                                   % User email (optional)
    config.user.research_group = '';                          % Research group (optional)
    
    % User preferences
    config.user.preferences = struct();
    config.user.preferences.units = 'METRIC';                 % METRIC, IMPERIAL
    config.user.preferences.decimal_places = 3;               % Number of decimal places in output
    config.user.preferences.scientific_notation = false;      % Use scientific notation
    config.user.preferences.temperature_unit = 'CELSIUS';     % CELSIUS, FAHRENHEIT, KELVIN
    
    % Custom material database (if available)
    config.user.custom_materials = struct();
    config.user.custom_materials.enabled = false;
    config.user.custom_materials.database_path = '';
    config.user.custom_materials.validation_required = true;
    
    % Custom tool database (if available)
    config.user.custom_tools = struct();
    config.user.custom_tools.enabled = false;
    config.user.custom_tools.database_path = '';
    config.user.custom_tools.validation_required = true;
    
    %% ========================================================================
    %% CONFIGURATION VALIDATION AND FINALIZATION
    %% ========================================================================
    
    % Validate configuration settings
    config.validation = validate_user_config(config);
    
    % Configuration summary
    config.summary = generate_config_summary(config);
    
    % Print configuration status
    print_config_status(config);
    
    fprintf('✅ SFDP v17.3 User Configuration Loaded Successfully\n');
end

function validation_results = validate_user_config(config)
    %% Validate user configuration settings
    
    validation_results = struct();
    validation_results.timestamp = datestr(now);
    validation_results.warnings = {};
    validation_results.errors = {};
    validation_results.overall_status = 'VALID';
    
    % Validate MATLAB version
    current_version = version('-release');
    required_version = config.matlab.minimum_required_version;
    
    if str2double(current_version(2:5)) < str2double(required_version(2:5))
        validation_results.warnings{end+1} = sprintf(...
            'MATLAB version %s is older than recommended %s', current_version, required_version);
    end
    
    % Validate data directories
    base_dir = config.data_locations.base_directory;
    if ~exist(base_dir, 'dir')
        validation_results.warnings{end+1} = sprintf(...
            'Base directory does not exist: %s', base_dir);
    end
    
    % Validate memory settings
    if config.performance.memory_limit > 16e9 % 16GB
        validation_results.warnings{end+1} = ...
            'Memory limit set very high - ensure your system has adequate RAM';
    end
    
    % Validate Kalman filter settings
    if config.kalman.gain_bounds(1) >= config.kalman.gain_bounds(2)
        validation_results.errors{end+1} = ...
            'Kalman gain bounds are invalid - lower bound must be less than upper bound';
        validation_results.overall_status = 'INVALID';
    end
    
    % Set overall validation status
    if ~isempty(validation_results.errors)
        validation_results.overall_status = 'INVALID';
    elseif ~isempty(validation_results.warnings)
        validation_results.overall_status = 'VALID_WITH_WARNINGS';
    end
end

function summary = generate_config_summary(config)
    %% Generate configuration summary
    
    summary = struct();
    summary.matlab_version = config.matlab.version;
    summary.base_directory = config.data_locations.base_directory;
    summary.kalman_enabled = config.kalman.enabled;
    summary.parallel_enabled = config.performance.parallel_computing.enabled;
    summary.memory_limit_gb = config.performance.memory_limit / 1e9;
    summary.external_toolboxes_enabled = sum(structfun(@(x) x.enabled, config.external_toolboxes));
    summary.verbosity_level = config.output.console.verbosity_level;
end

function print_config_status(config)
    %% Print configuration status to console
    
    fprintf('  📊 Configuration Summary:\n');
    fprintf('    MATLAB Version: %s\n', config.matlab.version);
    fprintf('    Base Directory: %s\n', config.data_locations.base_directory);
    fprintf('    Kalman Filter: %s\n', iif(config.kalman.enabled, 'ENABLED', 'DISABLED'));
    fprintf('    Memory Limit: %.1f GB\n', config.performance.memory_limit / 1e9);
    fprintf('    Verbosity: %s\n', config.output.console.verbosity_level);
    
    % Check validation status
    if strcmp(config.validation.overall_status, 'INVALID')
        fprintf('  ❌ Configuration validation failed - please check settings\n');
        for i = 1:length(config.validation.errors)
            fprintf('    ERROR: %s\n', config.validation.errors{i});
        end
    elseif strcmp(config.validation.overall_status, 'VALID_WITH_WARNINGS')
        fprintf('  ⚠️  Configuration valid with warnings\n');
        for i = 1:length(config.validation.warnings)
            fprintf('    WARNING: %s\n', config.validation.warnings{i});
        end
    else
        fprintf('  ✅ Configuration validation passed\n');
    end
end

function result = iif(condition, true_value, false_value)
    %% Inline conditional function
    if condition
        result = true_value;
    else
        result = false_value;
    end
end