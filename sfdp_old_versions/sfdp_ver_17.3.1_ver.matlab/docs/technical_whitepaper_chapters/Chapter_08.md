# Chapter 8: Execution Pipeline and Data Flow

## 8.1 Multi-Material and Multi-Machine Support Framework

### 8.1.1 Dynamic Material Selection and Configuration

**재료 변동 지원의 설계 철학**

SFDP v17.3는 단일 재료(Ti-6Al-4V)에 국한되지 않고, **6가지 주요 항공/자동차 소재**를 지원하는 확장 가능한 구조로 설계되었습니다. 이는 실제 제조 환경에서 다양한 재료에 대한 가공 시뮬레이션 요구를 반영한 것입니다.

**지원 재료 및 특성:**

| 재료 | 용도 | 주요 특성 | 가공 난이도 |
|------|------|-----------|-------------|
| **Ti-6Al-4V** | 항공우주 | 높은 강도, 내열성 | 매우 어려움 |
| **Al2024-T3** | 항공구조 | 경량, 고강도 | 보통 |
| **SS316L** | 화학/의료 | 내부식성 | 어려움 |
| **Inconel718** | 고온부품 | 초내열성 | 매우 어려움 |
| **AISI1045** | 기계부품 | 범용성 | 쉬움 |
| **Al6061-T6** | 범용구조 | 가공성 우수 | 쉬움 |

**재료별 물성 데이터베이스 구조:**

```matlab
% SFDP_constants_tables.m에서 정의된 재료 물성
function [material_props] = load_material_properties(material_name)
    
    switch material_name
        case 'Ti6Al4V'
            material_props = struct();
            material_props.density = 4420;                    % kg/m³
            material_props.thermal_conductivity = 6.7;       % W/m·K
            material_props.specific_heat = 526;              % J/kg·K
            material_props.elastic_modulus = 113.8e9;        % Pa
            material_props.yield_strength = 880e6;           % Pa
            material_props.hardness = 334;                   % HV
            
            % Johnson-Cook 소성 파라미터
            material_props.jc_A = 782.7e6;     % 항복강도 [Pa]
            material_props.jc_B = 498.4e6;     % 경화계수 [Pa]
            material_props.jc_n = 0.28;        % 경화지수
            material_props.jc_C = 0.028;       % 변형률속도 민감도
            material_props.jc_m = 1.0;         % 온도 민감도
            
            % Taylor 공구수명 계수
            material_props.taylor_C = 180;      % m/min
            material_props.taylor_n = 0.25;
            material_props.taylor_a = 0.75;
            material_props.taylor_b = 0.15;
            material_props.taylor_c = 0.5;
            
        case 'Al2024_T3'
            material_props = struct();
            material_props.density = 2780;
            material_props.thermal_conductivity = 121;
            material_props.specific_heat = 875;
            material_props.elastic_modulus = 73.1e9;
            material_props.yield_strength = 345e6;
            material_props.hardness = 120;
            
            % 알루미늄 특화 Johnson-Cook 파라미터
            material_props.jc_A = 369e6;
            material_props.jc_B = 684e6;
            material_props.jc_n = 0.73;
            material_props.jc_C = 0.0083;
            material_props.jc_m = 1.7;
            
            % 알루미늄용 Taylor 계수 (고속 가공 최적화)
            material_props.taylor_C = 850;      % 높은 절삭속도 허용
            material_props.taylor_n = 0.35;
            material_props.taylor_a = 0.65;
            material_props.taylor_b = 0.10;
            material_props.taylor_c = 0.3;
            
        case 'Inconel718'
            material_props = struct();
            material_props.density = 8220;
            material_props.thermal_conductivity = 11.4;     % 낮은 열전도도
            material_props.specific_heat = 435;
            material_props.elastic_modulus = 200e9;
            material_props.yield_strength = 1275e6;         % 매우 높은 강도
            material_props.hardness = 415;
            
            % 초합금 특화 파라미터
            material_props.jc_A = 1241e6;
            material_props.jc_B = 622e6;
            material_props.jc_n = 0.6522;
            material_props.jc_C = 0.0134;
            material_props.jc_m = 1.3;
            
            % 저속 가공 최적화 Taylor 계수
            material_props.taylor_C = 85;       % 낮은 절삭속도
            material_props.taylor_n = 0.18;
            material_props.taylor_a = 0.85;
            material_props.taylor_b = 0.20;
            material_props.taylor_c = 0.7;
    end
    
    % 온도 의존성 물성 함수 추가
    material_props.temp_dependent_props = @(T) calculate_temperature_dependent_properties(material_props, T);
    
end
```

### 8.1.2 Multi-Machine Type Support and Adaptation

**머신 타입별 최적화 전략**

각 머신 타입은 고유한 운동학적 특성과 가공 능력을 가지므로, SFDP는 머신별 최적화된 시뮬레이션 파라미터를 제공합니다.

**지원 머신 타입과 특성:**

```matlab
% SFDP_user_config.m에서 정의
function [machine_config] = setup_machine_configuration(machine_type)
    
    switch machine_type
        case 'CNC_Turning'
            machine_config = struct();
            machine_config.max_spindle_speed = 4000;         % RPM
            machine_config.max_feed_rate = 0.5;              % mm/rev
            machine_config.max_depth_of_cut = 15.0;          % mm
            machine_config.power_rating = 15;                % kW
            machine_config.coolant_types = {'flood', 'mist', 'dry'};
            machine_config.thermal_stability = 'medium';
            machine_config.dynamic_stiffness = 2e8;          % N/m
            
            % 선반 특화 시뮬레이션 파라미터
            machine_config.cutting_geometry = 'turning';
            machine_config.chip_formation_model = '2D_orthogonal';
            machine_config.tool_wear_model = 'flank_wear_primary';
            
        case 'CNC_Milling'
            machine_config = struct();
            machine_config.max_spindle_speed = 15000;        % 높은 회전수
            machine_config.max_feed_rate = 0.3;
            machine_config.max_depth_of_cut = 8.0;
            machine_config.power_rating = 22;
            machine_config.coolant_types = {'flood', 'mist', 'air_blast', 'cryogenic'};
            machine_config.thermal_stability = 'high';
            machine_config.dynamic_stiffness = 5e8;          % 높은 강성
            
            % 밀링 특화 파라미터
            machine_config.cutting_geometry = 'milling';
            machine_config.chip_formation_model = '3D_oblique';
            machine_config.tool_wear_model = 'multi_edge_wear';
            machine_config.vibration_analysis = 'enabled';
            
        case 'High_Speed_Machining'
            machine_config = struct();
            machine_config.max_spindle_speed = 40000;        % 초고속
            machine_config.max_feed_rate = 1.5;              % 고이송
            machine_config.max_depth_of_cut = 2.0;           % 얕은 절삭
            machine_config.power_rating = 30;
            machine_config.coolant_types = {'mist', 'air_blast', 'cryogenic'};
            machine_config.thermal_stability = 'very_high';
            machine_config.dynamic_stiffness = 8e8;
            
            % 고속가공 특화 설정
            machine_config.cutting_geometry = 'high_speed_milling';
            machine_config.chip_formation_model = 'thin_chip_formation';
            machine_config.tool_wear_model = 'thermal_wear_dominant';
            machine_config.thermal_analysis = 'enhanced';
            
        case 'Micro_Machining'
            machine_config = struct();
            machine_config.max_spindle_speed = 100000;       % 초고속 스핀들
            machine_config.max_feed_rate = 0.01;             % 미세 이송
            machine_config.max_depth_of_cut = 0.1;           % 미세 절삭
            machine_config.power_rating = 5;
            machine_config.coolant_types = {'dry', 'minimal_quantity'};
            machine_config.thermal_stability = 'ultra_high';
            machine_config.dynamic_stiffness = 1e9;          % 최고 강성
            
            % 미세가공 특화 설정
            machine_config.cutting_geometry = 'micro_cutting';
            machine_config.chip_formation_model = 'size_effect_included';
            machine_config.tool_wear_model = 'edge_sharpness_critical';
            machine_config.surface_analysis = 'nanometric';
    end
    
    % 머신별 물리 모델 보정 계수
    machine_config.physics_correction_factors = calculate_machine_corrections(machine_type);
    
end
```

### 8.1.3 Intelligent Material-Machine Compatibility Matrix

**재료-머신 호환성 분석**

SFDP는 재료와 머신의 조합에 대한 **호환성 매트릭스**를 제공하여 최적의 가공 조건을 추천합니다:

```matlab
function [compatibility_score, recommended_conditions] = evaluate_material_machine_compatibility(material_type, machine_type)
    
    % 호환성 매트릭스 (0-1 스케일, 1이 최적)
    compatibility_matrix = [
        %        Turning  Milling  HSM    Micro
        % Ti6Al4V
        [0.85,   0.75,    0.45,   0.25];   % 선반이 가장 적합
        % Al2024
        [0.65,   0.95,    0.85,   0.70];   % 밀링/고속가공 적합
        % SS316L  
        [0.80,   0.70,    0.50,   0.30];   % 선반 중심
        % Inconel718
        [0.90,   0.60,    0.30,   0.15];   % 저속 선반 최적
        % AISI1045
        [0.95,   0.85,    0.75,   0.60];   % 범용성 우수
        % Al6061
        [0.70,   0.90,    0.95,   0.80];   % 고속가공 최적
    ];
    
    % 재료-머신 인덱스 매핑
    material_idx = get_material_index(material_type);
    machine_idx = get_machine_index(machine_type);
    
    compatibility_score = compatibility_matrix(material_idx, machine_idx);
    
    % 권장 조건 계산
    if compatibility_score >= 0.8
        aggressiveness = 'high';
    elseif compatibility_score >= 0.6
        aggressiveness = 'medium';
    else
        aggressiveness = 'conservative';
    end
    
    recommended_conditions = generate_recommended_conditions(material_type, machine_type, aggressiveness);
    
end
```

## 8.2 System Initialization and Configuration

The SFDP system follows a structured initialization process that sets up the computational environment, loads necessary libraries, and configures system parameters.

### 8.1.1 Main Execution Entry Point

**File: SFDP_v17_3_main.m:1-50**

```matlab
function SFDP_v17_3_main()
    % SFDP v17.3: Smart Fusion-based Dynamic Prediction System
    % 6-Layer Hierarchical Multi-Physics Machining Simulation
    
    fprintf('🚀 SFDP v17.3 시스템 시작\n');
    fprintf('=====================================\n');
    
    % 시스템 초기화
    try
        fprintf('📋 시스템 초기화 중...\n');
        [system_initialized, simulation_state] = SFDP_initialize_system();
        
        if ~system_initialized
            error('시스템 초기화 실패');
        end
        
        fprintf('✅ 시스템 초기화 완료\n');
        
    catch ME
        fprintf('❌ 시스템 초기화 중 오류 발생: %s\n', ME.message);
        return;
    end
    
    % 물리 기반 설정
    try
        fprintf('🔬 물리 기반 설정 중...\n');
        [physics_setup, simulation_state] = SFDP_setup_physics_foundation(simulation_state);
        
        if ~physics_setup
            error('물리 기반 설정 실패');
        end
        
        fprintf('✅ 물리 기반 설정 완료\n');
        
    catch ME
        fprintf('❌ 물리 기반 설정 중 오류 발생: %s\n', ME.message);
        return;
    end
end
```

### 8.1.2 System State Management

The simulation state is a comprehensive structure that maintains all system information throughout the execution:

**Implementation in SFDP_initialize_system.m:45-120**

```matlab
function [success, simulation_state] = SFDP_initialize_system()
    simulation_state = struct();
    
    % 1. 시스템 정보 수집
    simulation_state.system_info = collectSystemInformation();
    
    % 2. 설정 파일 로드
    simulation_state.config = loadSystemConfiguration();
    
    % 3. 데이터 경로 설정
    simulation_state.paths = setupDataPaths();
    
    % 4. 라이브러리 초기화
    simulation_state.libraries = initializeLibraries();
    
    % 5. 계산 환경 준비
    simulation_state.computation = prepareComputationEnvironment();
    
    % 6. 로깅 시스템 초기화
    simulation_state.logging = initializeLoggingSystem();
    
    success = validateSystemInitialization(simulation_state);
end

function system_info = collectSystemInformation()
    system_info = struct();
    
    % MATLAB 버전 정보
    system_info.matlab_version = version;
    system_info.matlab_release = version('-release');
    
    % 시스템 리소스
    if ispc
        [~, memory_info] = memory;
        system_info.total_memory = memory_info.MemAvailableAllArrays;
        system_info.available_memory = memory_info.MemAvailableAllArrays;
    else
        system_info.total_memory = 8e9; % 기본값 8GB
        system_info.available_memory = 6e9; % 기본값 6GB
    end
    
    % CPU 정보
    system_info.cpu_cores = feature('numcores');
    system_info.max_threads = maxNumCompThreads;
    
    % 운영체제 정보
    system_info.os = computer;
    system_info.architecture = computer('arch');
    
    % 현재 시간
    system_info.start_time = datetime('now');
    system_info.session_id = generateSessionID();
end

function session_id = generateSessionID()
    % 고유한 세션 ID 생성
    current_time = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    random_suffix = sprintf('%04d', randi([1000, 9999]));
    session_id = sprintf('SFDP_%s_%s', char(current_time), random_suffix);
end

function config = loadSystemConfiguration()
    config = struct();
    
    % 기본 설정값
    config.max_computation_time = 300;    % 5분 최대 계산 시간
    config.max_memory_usage = 4;          % 4GB 최대 메모리 사용
    config.target_accuracy = 0.85;        % 85% 목표 정확도
    config.enable_parallel = true;        % 병렬 처리 활성화
    config.auto_fallback = true;          % 자동 폴백 활성화
    config.verbose_output = true;         % 상세 출력 활성화
    
    % 설정 파일이 존재하면 로드
    config_file = 'SFDP_config.mat';
    if exist(config_file, 'file')
        try
            loaded_config = load(config_file);
            if isfield(loaded_config, 'SFDP_config')
                % 로드된 설정으로 기본값 덮어쓰기
                config_fields = fieldnames(loaded_config.SFDP_config);
                for i = 1:length(config_fields)
                    field_name = config_fields{i};
                    config.(field_name) = loaded_config.SFDP_config.(field_name);
                end
                fprintf('설정 파일 로드 완료: %s\n', config_file);
            end
        catch ME
            fprintf('설정 파일 로드 실패, 기본값 사용: %s\n', ME.message);
        end
    end
end

function paths = setupDataPaths()
    paths = struct();
    
    % 기본 경로 설정
    paths.root = pwd;
    paths.data = fullfile(paths.root, 'data_set');
    paths.results = fullfile(paths.root, 'results');
    paths.temp = fullfile(paths.root, 'temp');
    paths.logs = fullfile(paths.root, 'logs');
    
    % 필요한 디렉토리 생성
    required_dirs = {paths.data, paths.results, paths.temp, paths.logs};
    for i = 1:length(required_dirs)
        if ~exist(required_dirs{i}, 'dir')
            mkdir(required_dirs{i});
            fprintf('디렉토리 생성: %s\n', required_dirs{i});
        end
    end
    
    % 데이터 파일 경로
    paths.material_database = fullfile(paths.data, 'materials.mat');
    paths.validation_data = fullfile(paths.data, 'validation_experiments.mat');
    paths.calibration_data = fullfile(paths.data, 'calibration_coefficients.mat');
end

function libraries = initializeLibraries()
    libraries = struct();
    
    % 필수 툴박스 확인
    required_toolboxes = {
        'Partial Differential Equation Toolbox',
        'Statistics and Machine Learning Toolbox',
        'Signal Processing Toolbox',
        'Optimization Toolbox'
    };
    
    for i = 1:length(required_toolboxes)
        toolbox_name = required_toolboxes{i};
        if license('test', strrep(toolbox_name, ' ', '_'))
            libraries.(strrep(toolbox_name, ' ', '_')) = true;
            fprintf('✅ %s 사용 가능\n', toolbox_name);
        else
            libraries.(strrep(toolbox_name, ' ', '_')) = false;
            fprintf('⚠️ %s 사용 불가\n', toolbox_name);
        end
    end
    
    % 외부 라이브러리 확인
    libraries.featool_available = exist('featool', 'file') > 0;
    libraries.gibbon_available = exist('febio_spec', 'file') > 0;
    
    if libraries.featool_available
        fprintf('✅ FEATool Multiphysics 사용 가능\n');
    else
        fprintf('⚠️ FEATool Multiphysics 사용 불가 - 해석적 방법 사용\n');
    end
    
    if libraries.gibbon_available
        fprintf('✅ GIBBON 사용 가능\n');
    else
        fprintf('⚠️ GIBBON 사용 불가 - 단순화된 접촉역학 사용\n');
    end
end
```

## 8.2 Data Loading and Preprocessing Pipeline

### 8.2.1 Intelligent Data Loader

The system includes an intelligent data loading mechanism that automatically detects data formats and validates input parameters.

**Implementation in SFDP_intelligent_data_loader.m:1-150**

```matlab
function [data_loaded, material_database, simulation_state] = SFDP_intelligent_data_loader(simulation_state)
    fprintf('📊 지능형 데이터 로더 시작\n');
    
    data_loaded = false;
    material_database = [];
    
    % 1. 데이터 소스 탐지
    try
        fprintf('  🔍 데이터 소스 탐지 중...\n');
        [data_sources, detection_confidence] = detectAvailableDataSources(simulation_state.paths);
        
        if detection_confidence < 0.5
            warning('데이터 소스 탐지 신뢰도가 낮습니다 (%.2f)', detection_confidence);
        end
        
        fprintf('  ✅ %d개 데이터 소스 탐지됨 (신뢰도: %.2f)\n', length(data_sources), detection_confidence);
        
    catch ME
        fprintf('  ❌ 데이터 소스 탐지 실패: %s\n', ME.message);
        return;
    end
    
    % 2. 재료 데이터베이스 로드
    try
        fprintf('  📋 재료 데이터베이스 로드 중...\n');
        [material_database, material_confidence] = loadMaterialDatabase(data_sources, simulation_state);
        
        if material_confidence < 0.7
            warning('재료 데이터베이스 신뢰도가 낮습니다 (%.2f)', material_confidence);
        end
        
        fprintf('  ✅ 재료 데이터베이스 로드 완료 (%d개 재료, 신뢰도: %.2f)\n', ...
                length(material_database.materials), material_confidence);
        
    catch ME
        fprintf('  ❌ 재료 데이터베이스 로드 실패: %s\n', ME.message);
        return;
    end
    
    % 3. 검증 데이터셋 로드
    try
        fprintf('  🧪 검증 데이터셋 로드 중...\n');
        [validation_dataset, validation_confidence] = loadValidationDataset(data_sources, simulation_state);
        
        simulation_state.validation_data = validation_dataset;
        simulation_state.validation_confidence = validation_confidence;
        
        fprintf('  ✅ 검증 데이터셋 로드 완료 (%d개 실험, 신뢰도: %.2f)\n', ...
                length(validation_dataset.experiments), validation_confidence);
        
    catch ME
        fprintf('  ⚠️ 검증 데이터셋 로드 실패 (선택사항): %s\n', ME.message);
        simulation_state.validation_data = [];
        simulation_state.validation_confidence = 0;
    end
    
    data_loaded = true;
end

function [data_sources, confidence] = detectAvailableDataSources(paths)
    data_sources = {};
    confidence = 0;
    
    % 재료 데이터베이스 파일 확인
    if exist(paths.material_database, 'file')
        data_sources{end+1} = struct('type', 'material_database', 'path', paths.material_database, 'priority', 1);
    end
    
    % 검증 데이터 파일 확인
    if exist(paths.validation_data, 'file')
        data_sources{end+1} = struct('type', 'validation_data', 'path', paths.validation_data, 'priority', 2);
    end
    
    % CSV 파일들 스캔
    csv_files = dir(fullfile(paths.data, '*.csv'));
    for i = 1:length(csv_files)
        csv_path = fullfile(csv_files(i).folder, csv_files(i).name);
        data_sources{end+1} = struct('type', 'csv_data', 'path', csv_path, 'priority', 3);
    end
    
    % TXT 파일들 스캔
    txt_files = dir(fullfile(paths.data, '*.txt'));
    for i = 1:length(txt_files)
        txt_path = fullfile(txt_files(i).folder, txt_files(i).name);
        data_sources{end+1} = struct('type', 'text_data', 'path', txt_path, 'priority', 4);
    end
    
    % 신뢰도 계산
    if length(data_sources) >= 2
        confidence = 0.9;
    elseif length(data_sources) == 1
        confidence = 0.6;
    else
        confidence = 0.1;
    end
end
```

### 8.2.2 Material Property Processing

The material database contains comprehensive properties for Ti-6Al-4V and other aerospace alloys.

**Material Database Structure (SFDP_constants_tables.m:150-250)**

```matlab
function material_database = createMaterialDatabase()
    material_database = struct();
    
    % Ti-6Al-4V 속성 (주요 대상 재료)
    Ti6Al4V = struct();
    Ti6Al4V.name = 'Ti-6Al-4V';
    Ti6Al4V.category = 'titanium_alloy';
    
    % 열적 속성
    Ti6Al4V.thermal.conductivity = 6.7;      % W/(m·K) at 20°C
    Ti6Al4V.thermal.specific_heat = 526;     % J/(kg·K)
    Ti6Al4V.thermal.density = 4430;         % kg/m³
    Ti6Al4V.thermal.diffusivity = 2.87e-6;  % m²/s
    Ti6Al4V.thermal.expansion = 8.6e-6;     % /K
    
    % 기계적 속성
    Ti6Al4V.mechanical.youngs_modulus = 113.8e9;  % Pa
    Ti6Al4V.mechanical.poisson_ratio = 0.342;
    Ti6Al4V.mechanical.yield_strength = 880e6;    % Pa
    Ti6Al4V.mechanical.ultimate_strength = 950e6; % Pa
    Ti6Al4V.mechanical.hardness_hv = 349;         % Vickers hardness
    
    % 화학적 속성
    Ti6Al4V.chemical.aluminum_content = 6.0;      % wt%
    Ti6Al4V.chemical.vanadium_content = 4.0;      % wt%
    Ti6Al4V.chemical.titanium_content = 90.0;     % wt%
    Ti6Al4V.chemical.oxygen_limit = 0.2;          % wt% max
    Ti6Al4V.chemical.iron_limit = 0.3;            % wt% max
    
    % 가공성 관련 속성
    Ti6Al4V.machinability.cutting_force_coefficient = 2100; % N/mm²
    Ti6Al4V.machinability.specific_cutting_energy = 2.8;   % J/mm³
    Ti6Al4V.machinability.built_up_edge_tendency = 0.8;    % 0-1 scale
    Ti6Al4V.machinability.tool_wear_factor = 1.4;          % relative to steel
    
    material_database.materials.Ti6Al4V = Ti6Al4V;
    
    % 추가 재료들 (간략화)
    % Inconel 718
    Inconel718 = struct();
    Inconel718.name = 'Inconel 718';
    Inconel718.category = 'nickel_superalloy';
    Inconel718.thermal.conductivity = 11.4;     % W/(m·K)
    Inconel718.thermal.specific_heat = 435;     % J/(kg·K)
    Inconel718.thermal.density = 8220;          % kg/m³
    Inconel718.mechanical.youngs_modulus = 200e9; % Pa
    Inconel718.mechanical.yield_strength = 1035e6; % Pa
    material_database.materials.Inconel718 = Inconel718;
    
    % 316L Stainless Steel
    SS316L = struct();
    SS316L.name = '316L Stainless Steel';
    SS316L.category = 'stainless_steel';
    SS316L.thermal.conductivity = 16.2;         % W/(m·K)
    SS316L.thermal.specific_heat = 500;         % J/(kg·K)
    SS316L.thermal.density = 8000;              % kg/m³
    SS316L.mechanical.youngs_modulus = 200e9;   % Pa
    SS316L.mechanical.yield_strength = 290e6;   % Pa
    material_database.materials.SS316L = SS316L;
    
    % 메타데이터
    material_database.metadata.version = '17.3';
    material_database.metadata.last_updated = datetime('now');
    material_database.metadata.total_materials = length(fieldnames(material_database.materials));
    
    fprintf('재료 데이터베이스 생성 완료: %d개 재료\n', material_database.metadata.total_materials);
end

function [material_props, confidence] = extractMaterialProperties(material_name, material_database)
    % 재료명으로 속성 추출
    
    confidence = 0;
    material_props = struct();
    
    % 재료명 정규화
    normalized_name = strrep(lower(material_name), '-', '');
    normalized_name = strrep(normalized_name, ' ', '');
    
    % 데이터베이스에서 검색
    material_fields = fieldnames(material_database.materials);
    
    for i = 1:length(material_fields)
        db_material_name = strrep(lower(material_fields{i}), '-', '');
        db_material_name = strrep(db_material_name, ' ', '');
        
        if contains(db_material_name, normalized_name) || contains(normalized_name, db_material_name)
            material_props = material_database.materials.(material_fields{i});
            confidence = 0.95;
            fprintf('재료 속성 추출 완료: %s (신뢰도: %.2f)\n', material_props.name, confidence);
            return;
        end
    end
    
    % 기본 Ti-6Al-4V 속성 사용
    if isfield(material_database.materials, 'Ti6Al4V')
        material_props = material_database.materials.Ti6Al4V;
        confidence = 0.5;
        fprintf('⚠️ 재료를 찾을 수 없음. Ti-6Al-4V 기본값 사용 (신뢰도: %.2f)\n', confidence);
    else
        error('기본 재료 속성을 찾을 수 없습니다');
    end
end
```

## 8.3 Computational Workflow Management

### 8.3.1 Layer Execution Controller

The system manages the execution of different computational layers through a centralized controller.

**Implementation in SFDP_execute_6layer_calculations.m:1-100**

```matlab
function [calculation_results, execution_summary] = SFDP_execute_6layer_calculations(...
    cutting_conditions, material_props, simulation_state)
    
    fprintf('🔄 6-Layer 계산 파이프라인 시작\n');
    fprintf('=====================================\n');
    
    execution_start_time = tic;
    calculation_results = struct();
    execution_summary = struct();
    
    % 계산 레이어 선택
    try
        fprintf('🎯 최적 계산 레이어 결정 중...\n');
        
        computation_budget = struct();
        computation_budget.max_time_seconds = simulation_state.config.max_computation_time;
        computation_budget.max_memory_gb = simulation_state.config.max_memory_usage;
        
        accuracy_requirement = simulation_state.config.target_accuracy;
        
        selected_layers = determineOptimalLayerExecution(cutting_conditions, material_props, ...
                                                        computation_budget, accuracy_requirement, simulation_state);
        
        execution_summary.selected_layers = selected_layers;
        execution_summary.layer_selection_time = toc(execution_start_time);
        
        fprintf('✅ 선택된 계산 레이어: %s\n', mat2str(selected_layers));
        
    catch ME
        fprintf('❌ 계산 레이어 선택 실패: %s\n', ME.message);
        calculation_results = [];
        execution_summary.error = ME.message;
        return;
    end
    
    % 선택된 레이어별 순차 실행
    layer_results = cell(1, 6);
    layer_execution_times = zeros(1, 6);
    layer_confidences = zeros(1, 6);
    
    for layer_idx = selected_layers
        layer_start_time = tic;
        
        try
            switch layer_idx
                case 1
                    fprintf('\n🔬 Layer 1: 고급 물리 해석 실행\n');
                    layer_results{1} = executeLayer1AdvancedPhysics(cutting_conditions, material_props, simulation_state);
                    
                case 2
                    fprintf('\n📊 Layer 2: 간소화 물리 해석 실행\n');
                    layer_results{2} = executeLayer2SimplifiedPhysics(cutting_conditions, material_props, simulation_state);
                    
                case 3
                    fprintf('\n📈 Layer 3: 경험적 평가 실행\n');
                    layer_results{3} = executeLayer3EmpiricalAssessment(cutting_conditions, material_props, simulation_state);
                    
                case 4
                    fprintf('\n🔧 Layer 4: 데이터 보정 실행\n');
                    layer_results{4} = executeLayer4DataCorrection(cutting_conditions, material_props, ...
                        layer_results{1}, layer_results{2}, layer_results{3}, simulation_state);
                    
                case 5
                    fprintf('\n🎯 Layer 5: 칼만 필터 융합 실행\n');
                    layer_results{5} = executeLayer5KalmanFusion(cutting_conditions, material_props, ...
                        layer_results{1}, layer_results{2}, layer_results{3}, layer_results{4}, simulation_state);
                    
                case 6
                    fprintf('\n🏁 Layer 6: 최종 처리 및 품질 평가 실행\n');
                    layer_results{6} = executeLayer6FinalProcessing(cutting_conditions, material_props, ...
                        layer_results{1}, layer_results{2}, layer_results{3}, layer_results{4}, ...
                        layer_results{5}, simulation_state);
            end
            
            layer_execution_times(layer_idx) = toc(layer_start_time);
            
            if ~isempty(layer_results{layer_idx}) && isfield(layer_results{layer_idx}, 'overall_confidence')
                layer_confidences(layer_idx) = layer_results{layer_idx}.overall_confidence;
            end
            
            fprintf('Layer %d 완료: %.2f초, 신뢰도: %.2f\n', layer_idx, ...
                   layer_execution_times(layer_idx), layer_confidences(layer_idx));
            
        catch ME
            fprintf('❌ Layer %d 실행 실패: %s\n', layer_idx, ME.message);
            
            % 오류 복구 시도
            recovered_result = handle_layer_error(layer_idx, ME, layer_results, simulation_state);
            if ~isempty(recovered_result)
                layer_results{layer_idx} = recovered_result;
                layer_confidences(layer_idx) = recovered_result.meta.overall_confidence;
                fprintf('🔧 Layer %d 오류 복구 완료\n', layer_idx);
            else
                layer_results{layer_idx} = [];
                layer_confidences(layer_idx) = 0;
            end
            
            layer_execution_times(layer_idx) = toc(layer_start_time);
        end
    end
    
    % 실행 요약 정보 업데이트
    execution_summary.total_execution_time = toc(execution_start_time);
    execution_summary.layer_results = layer_results;
    execution_summary.layer_execution_times = layer_execution_times;
    execution_summary.layer_confidences = layer_confidences;
    execution_summary.executed_layers = find(~cellfun(@isempty, layer_results));
end
```

### 8.3.2 Result Integration and Quality Control

After layer execution, the system integrates results and performs quality control checks.

**Result Integration (SFDP_execute_6layer_calculations.m:3700-3850)**

```matlab
    % 결과 통합 및 품질 제어
    try
        fprintf('\n🔀 결과 통합 및 품질 제어 시작\n');
        
        integration_start_time = tic;
        
        % 1. 유효 결과 식별
        valid_layers = find(~cellfun(@isempty, layer_results));
        fprintf('  📊 유효한 계산 레이어: %s\n', mat2str(valid_layers));
        
        if isempty(valid_layers)
            error('모든 계산 레이어가 실패했습니다');
        end
        
        % 2. 최적 결과 선택
        if ismember(6, valid_layers) && layer_confidences(6) > 0.5
            % Layer 6 (최종 처리) 결과가 신뢰할 만한 경우
            calculation_results.primary = layer_results{6};
            calculation_results.primary_source = 6;
            calculation_results.confidence = layer_confidences(6);
            
        elseif ismember(5, valid_layers) && layer_confidences(5) > 0.5
            % Layer 5 (칼만 융합) 결과가 신뢰할 만한 경우
            calculation_results.primary = layer_results{5};
            calculation_results.primary_source = 5;
            calculation_results.confidence = layer_confidences(5);
            
        else
            % 가장 높은 신뢰도를 가진 레이어 선택
            [max_confidence, max_idx] = max(layer_confidences(valid_layers));
            selected_layer = valid_layers(max_idx);
            
            calculation_results.primary = layer_results{selected_layer};
            calculation_results.primary_source = selected_layer;
            calculation_results.confidence = max_confidence;
        end
        
        % 3. 백업 결과 보관
        calculation_results.all_layers = layer_results;
        calculation_results.layer_confidences = layer_confidences;
        calculation_results.layer_execution_times = layer_execution_times;
        
        % 4. 통합 품질 지표 계산
        calculation_results.quality_metrics = calculateIntegratedQualityMetrics(...
            layer_results, layer_confidences, simulation_state);
        
        integration_time = toc(integration_start_time);
        fprintf('  ✅ 결과 통합 완료: %.2f초, 최종 신뢰도: %.2f (Layer %d)\n', ...
               integration_time, calculation_results.confidence, calculation_results.primary_source);
        
    catch ME
        fprintf('  ❌ 결과 통합 실패: %s\n', ME.message);
        calculation_results.error = ME.message;
        calculation_results.confidence = 0;
    end
    
    % 최종 실행 요약
    execution_summary.integration_time = integration_time;
    execution_summary.final_confidence = calculation_results.confidence;
    execution_summary.primary_source_layer = calculation_results.primary_source;
    execution_summary.total_computation_time = toc(execution_start_time);
    
    fprintf('\n🏁 6-Layer 계산 파이프라인 완료\n');
    fprintf('=====================================\n');
    fprintf('총 실행 시간: %.2f초\n', execution_summary.total_computation_time);
    fprintf('최종 신뢰도: %.2f\n', execution_summary.final_confidence);
    fprintf('주 결과 소스: Layer %d\n', execution_summary.primary_source_layer);
end

function quality_metrics = calculateIntegratedQualityMetrics(layer_results, layer_confidences, simulation_state)
    quality_metrics = struct();
    
    % 1. 계산 커버리지 (실행된 레이어 비율)
    executed_layers = find(~cellfun(@isempty, layer_results));
    quality_metrics.computation_coverage = length(executed_layers) / 6;
    
    % 2. 평균 신뢰도
    valid_confidences = layer_confidences(layer_confidences > 0);
    if ~isempty(valid_confidences)
        quality_metrics.average_confidence = mean(valid_confidences);
        quality_metrics.confidence_std = std(valid_confidences);
    else
        quality_metrics.average_confidence = 0;
        quality_metrics.confidence_std = 0;
    end
    
    % 3. 결과 일관성 (여러 레이어 결과 간 일치도)
    quality_metrics.result_consistency = calculateResultConsistency(layer_results);
    
    % 4. 물리적 타당성
    quality_metrics.physical_validity = assessPhysicalValidity(layer_results);
    
    % 5. 종합 품질 점수
    weights = [0.3, 0.4, 0.2, 0.1];  % 커버리지, 신뢰도, 일관성, 물리적타당성
    scores = [quality_metrics.computation_coverage, quality_metrics.average_confidence, ...
             quality_metrics.result_consistency, quality_metrics.physical_validity];
    
    quality_metrics.overall_quality_score = sum(weights .* scores);
    
    % 6. 품질 등급
    if quality_metrics.overall_quality_score >= 0.9
        quality_metrics.quality_grade = 'A';
    elseif quality_metrics.overall_quality_score >= 0.8
        quality_metrics.quality_grade = 'B';
    elseif quality_metrics.overall_quality_score >= 0.7
        quality_metrics.quality_grade = 'C';
    else
        quality_metrics.quality_grade = 'D';
    end
    
    fprintf('  품질 지표 - 커버리지: %.2f, 신뢰도: %.2f, 일관성: %.2f, 물리타당성: %.2f\n', ...
           quality_metrics.computation_coverage, quality_metrics.average_confidence, ...
           quality_metrics.result_consistency, quality_metrics.physical_validity);
    fprintf('  종합 품질 점수: %.2f (등급: %s)\n', ...
           quality_metrics.overall_quality_score, quality_metrics.quality_grade);
end

function consistency = calculateResultConsistency(layer_results)
    % 여러 레이어 결과 간 일관성 평가
    
    consistency = 1.0;  % 기본값
    
    % 유효한 결과가 있는 레이어들 찾기
    valid_layers = find(~cellfun(@isempty, layer_results));
    
    if length(valid_layers) < 2
        return;  % 비교할 결과가 없음
    end
    
    % 주요 예측값들 추출 및 비교
    try
        temperatures = [];
        wear_rates = [];
        roughness_values = [];
        
        for layer_idx = valid_layers
            result = layer_results{layer_idx};
            
            % 온도 데이터 추출
            if isfield(result, 'thermal') && isfield(result.thermal, 'max_temperature')
                temperatures(end+1) = result.thermal.max_temperature;
            end
            
            % 마모율 데이터 추출
            if isfield(result, 'wear') && isfield(result.wear, 'total_rate')
                wear_rates(end+1) = result.wear.total_rate;
            end
            
            % 표면 거칠기 데이터 추출
            if isfield(result, 'surface') && isfield(result.surface, 'roughness')
                roughness_values(end+1) = result.surface.roughness;
            end
        end
        
        % 변동계수 (CV) 계산으로 일관성 평가
        consistency_scores = [];
        
        if length(temperatures) >= 2
            cv_temp = std(temperatures) / mean(temperatures);
            consistency_scores(end+1) = max(0, 1 - cv_temp);
        end
        
        if length(wear_rates) >= 2
            cv_wear = std(wear_rates) / mean(wear_rates);
            consistency_scores(end+1) = max(0, 1 - cv_wear);
        end
        
        if length(roughness_values) >= 2
            cv_roughness = std(roughness_values) / mean(roughness_values);
            consistency_scores(end+1) = max(0, 1 - cv_roughness);
        end
        
        if ~isempty(consistency_scores)
            consistency = mean(consistency_scores);
        end
        
    catch ME
        fprintf('    ⚠️ 결과 일관성 계산 중 오류: %s\n', ME.message);
        consistency = 0.5;  % 중간값
    end
end

function validity = assessPhysicalValidity(layer_results)
    % 물리적 타당성 평가
    
    validity = 1.0;  % 기본값
    validity_checks = [];
    
    valid_layers = find(~cellfun(@isempty, layer_results));
    
    for layer_idx = valid_layers
        result = layer_results{layer_idx};
        layer_validity = 1.0;
        
        try
            % 온도 타당성 검사
            if isfield(result, 'thermal') && isfield(result.thermal, 'max_temperature')
                max_temp = result.thermal.max_temperature;
                if max_temp > 2000 || max_temp < 0  % 비현실적 온도
                    layer_validity = layer_validity * 0.5;
                end
            end
            
            % 마모율 타당성 검사
            if isfield(result, 'wear') && isfield(result.wear, 'total_rate')
                wear_rate = result.wear.total_rate;
                if wear_rate > 1e-3 || wear_rate < 0  % 비현실적 마모율 (mm/s)
                    layer_validity = layer_validity * 0.5;
                end
            end
            
            % 표면 거칠기 타당성 검사
            if isfield(result, 'surface') && isfield(result.surface, 'roughness')
                roughness = result.surface.roughness;
                if roughness > 100e-6 || roughness < 0  % 비현실적 거칠기 (m)
                    layer_validity = layer_validity * 0.5;
                end
            end
            
            validity_checks(end+1) = layer_validity;
            
        catch ME
            validity_checks(end+1) = 0.5;  % 오류시 중간값
        end
    end
    
    if ~isempty(validity_checks)
        validity = mean(validity_checks);
    end
end
```

## 8.4 Real-time Monitoring and Feedback

### 8.4.1 Progress Tracking System

```matlab
function progress_tracker = initializeProgressTracker(total_layers, simulation_state)
    progress_tracker = struct();
    
    % 진행상황 추적
    progress_tracker.total_layers = total_layers;
    progress_tracker.completed_layers = 0;
    progress_tracker.current_layer = 0;
    progress_tracker.start_time = tic;
    progress_tracker.layer_start_times = zeros(1, total_layers);
    progress_tracker.estimated_remaining_time = 0;
    
    % 성능 모니터링
    progress_tracker.memory_usage = [];
    progress_tracker.cpu_usage = [];
    progress_tracker.computation_efficiency = [];
    
    % 결과 품질 추적
    progress_tracker.confidence_history = [];
    progress_tracker.error_count = 0;
    progress_tracker.warning_count = 0;
    
    fprintf('📊 진행상황 추적기 초기화 완료\n');
end

function updateProgressTracker(progress_tracker, layer_number, layer_result, execution_time)
    % 진행상황 업데이트
    progress_tracker.completed_layers = progress_tracker.completed_layers + 1;
    progress_tracker.current_layer = layer_number;
    
    % 시간 추정
    elapsed_time = toc(progress_tracker.start_time);
    avg_time_per_layer = elapsed_time / progress_tracker.completed_layers;
    remaining_layers = progress_tracker.total_layers - progress_tracker.completed_layers;
    progress_tracker.estimated_remaining_time = avg_time_per_layer * remaining_layers;
    
    % 메모리 사용량 모니터링
    if ispc
        [~, memory_info] = memory;
        current_memory_usage = memory_info.MemUsedMATLAB / memory_info.MemAvailableAllArrays;
    else
        current_memory_usage = 0.5;  % 기본값
    end
    progress_tracker.memory_usage(end+1) = current_memory_usage;
    
    % 신뢰도 추적
    if ~isempty(layer_result) && isfield(layer_result, 'overall_confidence')
        progress_tracker.confidence_history(end+1) = layer_result.overall_confidence;
    else
        progress_tracker.confidence_history(end+1) = 0;
    end
    
    % 진행률 출력
    progress_percentage = (progress_tracker.completed_layers / progress_tracker.total_layers) * 100;
    fprintf('📈 진행률: %.1f%% (%d/%d), 예상 남은 시간: %.1f초\n', ...
           progress_percentage, progress_tracker.completed_layers, progress_tracker.total_layers, ...
           progress_tracker.estimated_remaining_time);
end
```

---

*Chapter 8은 SFDP v17.3의 실행 파이프라인과 데이터 흐름의 핵심을 다룹니다. 시스템 초기화부터 지능형 데이터 로더, 6-Layer 계산 파이프라인 제어, 결과 통합 및 품질 제어까지 전체 시스템의 실행 흐름을 체계적으로 관리합니다. 특히 오류 복구 메커니즘, 실시간 진행상황 추적, 품질 지표 평가 등을 통해 안정적이고 신뢰할 수 있는 시뮬레이션 환경을 제공합니다.*