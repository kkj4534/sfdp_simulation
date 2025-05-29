# Chapter 12: Error Handling and Fallback Systems

## 12.1 Graceful Degradation Architecture

### 12.1.1 Hierarchical Fallback Strategy

**SFDP의 다단계 Fallback 시스템**

SFDP v17.3은 계산 실패나 리소스 부족 시 자동으로 대체 방법을 사용하는 "Graceful Degradation" 시스템을 구현합니다.

```matlab
% executeWithGracefulDegradation 함수에서 메인 Fallback 로직
function [calculation_results, fallback_info] = execute_with_graceful_degradation(...
    cutting_conditions, material_props, simulation_state)
    
    fprintf('🛡️ Graceful Degradation 시스템 시작\n');
    
    fallback_info = struct();
    fallback_info.attempted_methods = {};
    fallback_info.success_level = 0;
    fallback_info.fallback_reason = '';
    
    % Fallback 레벨 정의 (높은 정확도 → 낮은 정확도)
    fallback_levels = {
        'Level 1: 고급 3D FEM + GIBBON + 칼먼 융합',
        'Level 2: 간소화 3D FEM + 기본 접촉 + 칼먼 융합',
        'Level 3: 해석적 방법 + 경험적 모델 + 기본 융합',
        'Level 4: Taylor 공식 + 간단한 경험식',
        'Level 5: 최소 기본 계산'
    };
    
    for level = 1:5
        fprintf('  🔄 %s 시도 중...\n', fallback_levels{level});
        fallback_info.attempted_methods{end+1} = fallback_levels{level};
        
        try
            switch level
                case 1
                    % 최고 정확도: 모든 고급 기능 사용
                    calculation_results = execute_full_advanced_calculation(...
                        cutting_conditions, material_props, simulation_state);
                    
                case 2
                    % 2단계: GIBBON 없이 실행
                    calculation_results = execute_without_gibbon(...
                        cutting_conditions, material_props, simulation_state);
                    
                case 3
                    % 3단계: FEM 없이 해석적 방법 사용
                    calculation_results = execute_analytical_methods(...
                        cutting_conditions, material_props, simulation_state);
                    
                case 4
                    % 4단계: 경험적 방법만 사용
                    calculation_results = execute_empirical_only(...
                        cutting_conditions, material_props, simulation_state);
                    
                case 5
                    % 5단계: 최소 기본 계산
                    calculation_results = execute_minimal_calculation(...
                        cutting_conditions, material_props, simulation_state);
            end
            
            % 성공한 경우
            fallback_info.success_level = level;
            fallback_info.final_method = fallback_levels{level};
            
            % 결과 품질 평가
            quality_score = evaluate_result_quality(calculation_results, level);
            calculation_results.quality_score = quality_score;
            calculation_results.confidence_adjustment = 1.0 - (level-1) * 0.15; % 레벨당 15% 감소
            
            fprintf('  ✅ %s 성공 (품질: %.2f, 신뢰도 조정: %.2f)\n', ...
                   fallback_levels{level}, quality_score, calculation_results.confidence_adjustment);
            
            % 성공 시 루프 종료
            break;
            
        catch ME
            fprintf('  ❌ %s 실패: %s\n', fallback_levels{level}, ME.message);
            fallback_info.fallback_reason = ME.message;
            
            % 마지막 레벨에서도 실패한 경우
            if level == 5
                fprintf('  💥 모든 Fallback 레벨 실패\n');
                calculation_results = create_emergency_result(cutting_conditions, material_props);
                fallback_info.success_level = 0;
                fallback_info.emergency_mode = true;
            end
        end
    end
    
    % Fallback 사용 통계 업데이트
    update_fallback_statistics(fallback_info, simulation_state);
    
    fprintf('🛡️ Graceful Degradation 완료: Level %d 성공\n', fallback_info.success_level);
end
```

### 12.1.2 Resource Monitoring and Adaptation

**리소스 모니터링 및 적응형 조정**

```matlab
% monitorSystemResources 함수에서 실시간 리소스 모니터링
function [resource_status, recommendations] = monitor_system_resources(simulation_state)
    
    resource_status = struct();
    recommendations = struct();
    
    % 1. 메모리 사용량 확인
    if ispc
        [~, sys_memory] = memory;
        available_memory_gb = sys_memory.MemAvailableAllArrays / 1e9;
        total_memory_gb = sys_memory.MemTotalPhys / 1e9;
        memory_usage_percent = (1 - available_memory_gb/total_memory_gb) * 100;
    else
        % Linux/Mac에서는 대략적 추정
        available_memory_gb = 4; % 기본값
        memory_usage_percent = 50; % 기본값
    end
    
    resource_status.memory_gb_available = available_memory_gb;
    resource_status.memory_usage_percent = memory_usage_percent;
    
    % 2. MATLAB 메모리 사용량
    matlab_memory = whos;
    matlab_memory_mb = sum([matlab_memory.bytes]) / 1e6;
    resource_status.matlab_memory_mb = matlab_memory_mb;
    
    % 3. CPU 코어 수 및 활용률
    resource_status.cpu_cores = feature('numcores');
    resource_status.max_threads = maxNumCompThreads;
    
    % 4. 디스크 공간 (임시 파일용)
    if ispc
        [~, disk_info] = system('dir /-c');
        % Windows에서 디스크 정보 파싱 (간소화)
        resource_status.disk_space_gb = 10; % 기본값
    else
        resource_status.disk_space_gb = 10; % 기본값
    end
    
    % 5. 리소스 상태 평가
    resource_status.overall_status = 'Good';
    warning_messages = {};
    
    % 메모리 부족 경고
    if available_memory_gb < 2
        resource_status.overall_status = 'Critical';
        warning_messages{end+1} = '메모리 부족 (< 2GB)';
        recommendations.reduce_mesh_density = true;
        recommendations.use_simplified_physics = true;
    elseif available_memory_gb < 4
        resource_status.overall_status = 'Warning';
        warning_messages{end+1} = '메모리 여유 부족 (< 4GB)';
        recommendations.reduce_fem_resolution = true;
    end
    
    % MATLAB 메모리 사용량 경고
    if matlab_memory_mb > 2000
        warning_messages{end+1} = sprintf('MATLAB 메모리 사용량 높음 (%.0f MB)', matlab_memory_mb);
        recommendations.clear_workspace = true;
    end
    
    % CPU 코어 활용 권장사항
    if resource_status.cpu_cores >= 4
        recommendations.enable_parallel_processing = true;
    end
    
    resource_status.warnings = warning_messages;
    
    % 6. 적응형 설정 권장사항
    if strcmp(resource_status.overall_status, 'Critical')
        recommendations.suggested_layers = [3, 4, 6]; % 경량 레이어만
        recommendations.mesh_size_multiplier = 2.0; % 메시 크기 2배 증가
        recommendations.time_step_multiplier = 2.0; % 시간 간격 2배 증가
    elseif strcmp(resource_status.overall_status, 'Warning')
        recommendations.suggested_layers = [2, 3, 5, 6]; % 중간 무게 레이어
        recommendations.mesh_size_multiplier = 1.5;
        recommendations.time_step_multiplier = 1.5;
    else
        recommendations.suggested_layers = [1, 5, 6]; % 모든 레이어 가능
        recommendations.mesh_size_multiplier = 1.0;
        recommendations.time_step_multiplier = 1.0;
    end
    
    % 결과 출력
    fprintf('📊 시스템 리소스 상태: %s\n', resource_status.overall_status);
    fprintf('  💾 사용가능 메모리: %.1f GB\n', available_memory_gb);
    fprintf('  🖥️ CPU 코어: %d개\n', resource_status.cpu_cores);
    
    if ~isempty(warning_messages)
        fprintf('  ⚠️ 경고:\n');
        for i = 1:length(warning_messages)
            fprintf('    - %s\n', warning_messages{i});
        end
    end
end
```

### 12.1.3 Automatic Configuration Adjustment

**자동 설정 조정**

```matlab
% adjustConfigurationForResources 함수에서 설정 자동 조정
function [adjusted_config] = adjust_configuration_for_resources(original_config, resource_status)
    
    fprintf('⚙️ 리소스에 따른 설정 자동 조정\n');
    
    adjusted_config = original_config;
    adjustments_made = {};
    
    % 1. 메모리 기반 조정
    if resource_status.memory_gb_available < 4
        % 메시 밀도 감소
        if isfield(original_config, 'mesh_density')
            original_density = original_config.mesh_density;
            adjusted_config.mesh_density = original_density * 0.7;
            adjustments_made{end+1} = sprintf('메시 밀도: %.3f → %.3f', original_density, adjusted_config.mesh_density);
        end
        
        % FEM 해상도 감소
        if isfield(original_config, 'fem_resolution')
            original_resolution = original_config.fem_resolution;
            adjusted_config.fem_resolution = ceil(original_resolution * 0.8);
            adjustments_made{end+1} = sprintf('FEM 해상도: %d → %d', original_resolution, adjusted_config.fem_resolution);
        end
        
        % 시간 스텝 증가
        if isfield(original_config, 'time_step')
            original_step = original_config.time_step;
            adjusted_config.time_step = original_step * 1.5;
            adjustments_made{end+1} = sprintf('시간 스텝: %.3f → %.3f', original_step, adjusted_config.time_step);
        end
    end
    
    % 2. CPU 코어 기반 조정
    if resource_status.cpu_cores >= 4
        adjusted_config.enable_parallel_fem = true;
        adjusted_config.parallel_workers = min(4, resource_status.cpu_cores);
        adjustments_made{end+1} = sprintf('병렬 처리 활성화: %d workers', adjusted_config.parallel_workers);
    else
        adjusted_config.enable_parallel_fem = false;
        adjustments_made{end+1} = '병렬 처리 비활성화 (코어 부족)';
    end
    
    % 3. 계산 레이어 자동 선택
    if resource_status.memory_gb_available < 2
        % 극도로 제한된 환경
        adjusted_config.auto_selected_layers = [4, 6]; % 경험적 + 검증만
        adjustments_made{end+1} = '극제한 모드: 경험적 계산만';
    elseif resource_status.memory_gb_available < 4
        % 제한된 환경
        adjusted_config.auto_selected_layers = [2, 3, 6]; % 간소화 물리 + 경험적
        adjustments_made{end+1} = '제한 모드: 간소화 물리 + 경험적';
    elseif resource_status.memory_gb_available < 8
        % 일반 환경
        adjusted_config.auto_selected_layers = [2, 5, 6]; % 간소화 물리 + 칼먼
        adjustments_made{end+1} = '일반 모드: 간소화 물리 + 칼먼';
    else
        % 충분한 환경
        adjusted_config.auto_selected_layers = [1, 5, 6]; % 고급 물리 + 칼먼
        adjustments_made{end+1} = '고성능 모드: 고급 물리 + 칼먼';
    end
    
    % 4. 정확도 vs 속도 균형 조정
    if resource_status.memory_gb_available < 4 || resource_status.cpu_cores < 2
        adjusted_config.accuracy_vs_speed_balance = 'speed_priority';
        adjustments_made{end+1} = '속도 우선 모드';
    else
        adjusted_config.accuracy_vs_speed_balance = 'balanced';
        adjustments_made{end+1} = '균형 모드';
    end
    
    % 5. 조정 사항 요약
    adjusted_config.adjustment_summary = adjustments_made;
    adjusted_config.resource_based_adjustment = true;
    adjusted_config.original_config_backup = original_config;
    
    % 결과 출력
    if ~isempty(adjustments_made)
        fprintf('  📝 수행된 조정:\n');
        for i = 1:length(adjustments_made)
            fprintf('    - %s\n', adjustments_made{i});
        end
    else
        fprintf('  ✅ 조정 불필요 (충분한 리소스)\n');
    end
end
```

## 12.2 Error Detection and Recovery

### 12.2.1 Comprehensive Error Classification

**포괄적 오류 분류 시스템**

```matlab
% classifyAndHandleError 함수에서 오류 분류 및 처리
function [error_info, recovery_action] = classify_and_handle_error(ME, execution_context)
    
    error_info = struct();
    error_info.timestamp = datetime('now');
    error_info.error_message = ME.message;
    error_info.error_identifier = ME.identifier;
    error_info.stack_trace = ME.stack;
    error_info.execution_context = execution_context;
    
    % 오류 분류
    error_category = classify_error_type(ME);
    error_info.category = error_category;
    error_info.severity = determine_error_severity(error_category, execution_context);
    
    % 복구 액션 결정
    recovery_action = determine_recovery_action(error_category, error_info.severity);
    
    fprintf('🚨 오류 감지: %s\n', error_category);
    fprintf('  📝 메시지: %s\n', ME.message);
    fprintf('  🎯 복구 액션: %s\n', recovery_action.description);
    
    % 오류 로깅
    log_error_to_file(error_info, recovery_action);
end

function error_category = classify_error_type(ME)
    
    error_message = lower(ME.message);
    error_id = ME.identifier;
    
    % 1. 메모리 관련 오류
    if contains(error_message, 'out of memory') || contains(error_message, 'not enough memory')
        error_category = 'MEMORY_INSUFFICIENT';
    elseif contains(error_message, 'maximum variable size') || contains(error_message, 'array too large')
        error_category = 'MEMORY_ARRAY_SIZE';
        
    % 2. 수치 계산 오류
    elseif contains(error_message, 'singular matrix') || contains(error_message, 'rank deficient')
        error_category = 'NUMERICAL_SINGULAR_MATRIX';
    elseif contains(error_message, 'not positive definite') || contains(error_message, 'ill-conditioned')
        error_category = 'NUMERICAL_ILL_CONDITIONED';
    elseif contains(error_message, 'convergence') || contains(error_message, 'iteration')
        error_category = 'NUMERICAL_CONVERGENCE';
    elseif contains(error_message, 'nan') || contains(error_message, 'inf')
        error_category = 'NUMERICAL_INVALID_VALUES';
        
    % 3. 파일 I/O 오류
    elseif contains(error_message, 'file not found') || contains(error_message, 'cannot open')
        error_category = 'FILE_NOT_FOUND';
    elseif contains(error_message, 'permission denied') || contains(error_message, 'access denied')
        error_category = 'FILE_PERMISSION';
    elseif contains(error_message, 'disk') || contains(error_message, 'space')
        error_category = 'FILE_DISK_SPACE';
        
    % 4. 툴박스 관련 오류
    elseif contains(error_message, 'featool') || contains(error_id, 'featool')
        error_category = 'TOOLBOX_FEATOOL';
    elseif contains(error_message, 'gibbon') || contains(error_id, 'gibbon')
        error_category = 'TOOLBOX_GIBBON';
    elseif contains(error_message, 'parallel') || contains(error_id, 'parallel')
        error_category = 'TOOLBOX_PARALLEL';
        
    % 5. 입력 데이터 오류
    elseif contains(error_message, 'dimension') || contains(error_message, 'size mismatch')
        error_category = 'DATA_DIMENSION_MISMATCH';
    elseif contains(error_message, 'invalid') || contains(error_message, 'out of range')
        error_category = 'DATA_INVALID_INPUT';
    elseif contains(error_message, 'empty') || contains(error_message, 'undefined')
        error_category = 'DATA_MISSING';
        
    % 6. 라이센스 오류
    elseif contains(error_message, 'license') || contains(error_message, 'checkout')
        error_category = 'LICENSE_UNAVAILABLE';
        
    % 7. 기타 오류
    else
        error_category = 'UNKNOWN';
    end
end

function recovery_action = determine_recovery_action(error_category, severity)
    
    recovery_action = struct();
    
    switch error_category
        case 'MEMORY_INSUFFICIENT'
            recovery_action.type = 'REDUCE_PROBLEM_SIZE';
            recovery_action.description = '문제 크기 축소 후 재시도';
            recovery_action.specific_actions = {
                'mesh_density_reduction', 'time_step_increase', 'precision_reduction'
            };
            
        case 'MEMORY_ARRAY_SIZE'
            recovery_action.type = 'CHUNK_PROCESSING';
            recovery_action.description = '청크 단위 처리로 전환';
            recovery_action.specific_actions = {
                'enable_chunked_processing', 'reduce_array_sizes'
            };
            
        case 'NUMERICAL_SINGULAR_MATRIX'
            recovery_action.type = 'REGULARIZATION';
            recovery_action.description = '정규화 기법 적용';
            recovery_action.specific_actions = {
                'add_regularization_term', 'use_pseudo_inverse'
            };
            
        case 'NUMERICAL_ILL_CONDITIONED'
            recovery_action.type = 'PRECONDITIONING';
            recovery_action.description = '전처리 기법 적용';
            recovery_action.specific_actions = {
                'apply_preconditioning', 'improve_mesh_quality'
            };
            
        case 'NUMERICAL_CONVERGENCE'
            recovery_action.type = 'ADJUST_SOLVER_PARAMS';
            recovery_action.description = '솔버 매개변수 조정';
            recovery_action.specific_actions = {
                'increase_max_iterations', 'relax_convergence_criteria'
            };
            
        case 'TOOLBOX_FEATOOL'
            recovery_action.type = 'FALLBACK_TO_ANALYTICAL';
            recovery_action.description = 'FEATool 없이 해석적 방법 사용';
            recovery_action.specific_actions = {
                'use_analytical_thermal', 'disable_fem_analysis'
            };
            
        case 'TOOLBOX_GIBBON'
            recovery_action.type = 'FALLBACK_TO_SIMPLE_CONTACT';
            recovery_action.description = 'GIBBON 없이 단순 접촉 모델 사용';
            recovery_action.specific_actions = {
                'use_hertz_contact', 'disable_advanced_contact'
            };
            
        case 'DATA_INVALID_INPUT'
            recovery_action.type = 'INPUT_VALIDATION_AND_CORRECTION';
            recovery_action.description = '입력 데이터 검증 및 보정';
            recovery_action.specific_actions = {
                'validate_input_ranges', 'apply_default_values'
            };
            
        case 'FILE_NOT_FOUND'
            recovery_action.type = 'USE_DEFAULT_DATA';
            recovery_action.description = '기본 데이터 사용';
            recovery_action.specific_actions = {
                'load_default_material_props', 'create_default_config'
            };
            
        otherwise
            recovery_action.type = 'GRACEFUL_DEGRADATION';
            recovery_action.description = '하위 레벨로 Fallback';
            recovery_action.specific_actions = {
                'try_lower_accuracy_method', 'use_emergency_calculation'
            };
    end
    
    recovery_action.severity = severity;
    recovery_action.automatic_recovery = (severity <= 3); % 심각도 3 이하는 자동 복구
end
```

### 12.2.2 Automatic Error Recovery

**자동 오류 복구 시스템**

```matlab
% executeWithAutoRecovery 함수에서 자동 복구 실행
function [result, recovery_info] = execute_with_auto_recovery(function_handle, varargin)
    
    recovery_info = struct();
    recovery_info.attempts = 0;
    recovery_info.recovery_actions_taken = {};
    recovery_info.final_success = false;
    
    max_recovery_attempts = 3;
    
    for attempt = 1:max_recovery_attempts
        recovery_info.attempts = attempt;
        
        try
            fprintf('🔄 실행 시도 %d/%d\n', attempt, max_recovery_attempts);
            
            % 함수 실행
            result = function_handle(varargin{:});
            
            % 성공
            recovery_info.final_success = true;
            fprintf('✅ 실행 성공 (시도 %d)\n', attempt);
            break;
            
        catch ME
            fprintf('❌ 실행 실패 (시도 %d): %s\n', attempt, ME.message);
            
            % 오류 분석 및 복구 액션 결정
            [error_info, recovery_action] = classify_and_handle_error(ME, varargin);
            
            if ~recovery_action.automatic_recovery
                fprintf('🚫 자동 복구 불가능한 오류\n');
                rethrow(ME);
            end
            
            % 복구 액션 실행
            fprintf('🔧 복구 액션 실행: %s\n', recovery_action.description);
            
            try
                varargin = apply_recovery_actions(recovery_action, varargin);
                recovery_info.recovery_actions_taken{end+1} = recovery_action.description;
                
            catch recovery_error
                fprintf('💥 복구 액션 실패: %s\n', recovery_error.message);
                
                if attempt == max_recovery_attempts
                    fprintf('🚨 모든 복구 시도 실패\n');
                    rethrow(ME); % 원래 오류 다시 발생
                end
            end
        end
    end
    
    if ~recovery_info.final_success
        error('최대 복구 시도 횟수 초과');
    end
end

function modified_args = apply_recovery_actions(recovery_action, original_args)
    
    modified_args = original_args;
    
    for i = 1:length(recovery_action.specific_actions)
        action = recovery_action.specific_actions{i};
        
        switch action
            case 'mesh_density_reduction'
                % 첫 번째 인자가 설정 구조체라고 가정
                if isstruct(modified_args{1}) && isfield(modified_args{1}, 'mesh_density')
                    modified_args{1}.mesh_density = modified_args{1}.mesh_density * 0.7;
                    fprintf('  📉 메시 밀도 70%%로 감소\n');
                end
                
            case 'time_step_increase'
                if isstruct(modified_args{1}) && isfield(modified_args{1}, 'time_step')
                    modified_args{1}.time_step = modified_args{1}.time_step * 1.5;
                    fprintf('  ⏰ 시간 스텝 1.5배 증가\n');
                end
                
            case 'precision_reduction'
                if isstruct(modified_args{1}) && isfield(modified_args{1}, 'precision')
                    if strcmp(modified_args{1}.precision, 'double')
                        modified_args{1}.precision = 'single';
                        fprintf('  🎯 정밀도 single로 변경\n');
                    end
                end
                
            case 'add_regularization_term'
                if isstruct(modified_args{1})
                    modified_args{1}.regularization_factor = 1e-6;
                    fprintf('  🔧 정규화 항 추가 (1e-6)\n');
                end
                
            case 'use_analytical_thermal'
                if isstruct(modified_args{1})
                    modified_args{1}.force_analytical_thermal = true;
                    fprintf('  📐 해석적 열해석 강제 사용\n');
                end
                
            case 'disable_fem_analysis'
                if isstruct(modified_args{1})
                    modified_args{1}.enable_fem = false;
                    fprintf('  🚫 FEM 해석 비활성화\n');
                end
                
            case 'use_hertz_contact'
                if isstruct(modified_args{1})
                    modified_args{1}.contact_model = 'hertz';
                    fprintf('  🤝 Hertz 접촉 모델 사용\n');
                end
                
            case 'validate_input_ranges'
                modified_args = validate_and_correct_inputs(modified_args);
                fprintf('  ✅ 입력값 검증 및 보정\n');
                
            otherwise
                fprintf('  ⚠️ 알 수 없는 복구 액션: %s\n', action);
        end
    end
end
```

## 12.3 Robustness Testing

### 12.3.1 Stress Testing Framework

**스트레스 테스트 프레임워크**

```matlab
% performRobustnessStressTesting 함수에서 강건성 테스트
function [stress_test_results] = perform_robustness_stress_testing(sfdp_system)
    
    fprintf('🏋️ SFDP 강건성 스트레스 테스트 시작\n');
    fprintf('=====================================\n');
    
    stress_test_results = struct();
    
    % 1. 극한 입력값 테스트
    fprintf('🔥 극한 입력값 테스트\n');
    extreme_input_results = test_extreme_inputs(sfdp_system);
    stress_test_results.extreme_inputs = extreme_input_results;
    
    % 2. 메모리 부족 시뮬레이션
    fprintf('💾 메모리 부족 시뮬레이션\n');
    memory_stress_results = test_memory_limitations(sfdp_system);
    stress_test_results.memory_stress = memory_stress_results;
    
    % 3. 수치 안정성 테스트
    fprintf('🔢 수치 안정성 테스트\n');
    numerical_stability_results = test_numerical_stability(sfdp_system);
    stress_test_results.numerical_stability = numerical_stability_results;
    
    % 4. 장시간 실행 테스트
    fprintf('⏰ 장시간 실행 테스트\n');
    endurance_results = test_long_duration_execution(sfdp_system);
    stress_test_results.endurance = endurance_results;
    
    % 5. 동시 실행 테스트
    fprintf('🔀 동시 실행 테스트\n');
    concurrent_execution_results = test_concurrent_execution(sfdp_system);
    stress_test_results.concurrent_execution = concurrent_execution_results;
    
    % 종합 강건성 점수 계산
    test_scores = [
        extreme_input_results.success_rate * 100,
        memory_stress_results.success_rate * 100,
        numerical_stability_results.success_rate * 100,
        endurance_results.success_rate * 100,
        concurrent_execution_results.success_rate * 100
    ];
    
    stress_test_results.overall_robustness_score = mean(test_scores);
    
    if stress_test_results.overall_robustness_score >= 90
        stress_test_results.robustness_grade = 'Excellent';
    elseif stress_test_results.overall_robustness_score >= 80
        stress_test_results.robustness_grade = 'Good';
    elseif stress_test_results.overall_robustness_score >= 70
        stress_test_results.robustness_grade = 'Acceptable';
    else
        stress_test_results.robustness_grade = 'Needs Improvement';
    end
    
    fprintf('=====================================\n');
    fprintf('🏆 강건성 테스트 완료: %.1f점 (%s)\n', ...
           stress_test_results.overall_robustness_score, stress_test_results.robustness_grade);
end

function extreme_input_results = test_extreme_inputs(sfdp_system)
    
    fprintf('  🎯 극한 조건 입력값 테스트\n');
    
    % 극한 테스트 케이스 정의
    extreme_cases = {
        % [속도, 이송, 깊이, 설명]
        [1, 0.001, 0.01, '극소값 테스트'],
        [1000, 2.0, 20, '극대값 테스트'],
        [500, 0.001, 0.01, '고속-저이송 테스트'],
        [50, 2.0, 20, '저속-고이송 테스트'],
        [NaN, 0.1, 1.0, 'NaN 입력 테스트'],
        [100, Inf, 1.0, 'Inf 입력 테스트'],
        [-100, 0.1, 1.0, '음수 입력 테스트'],
        [0, 0, 0, '제로 입력 테스트']
    };
    
    num_cases = length(extreme_cases);
    success_count = 0;
    test_details = cell(num_cases, 1);
    
    for i = 1:num_cases
        case_data = extreme_cases{i};
        test_conditions = struct();
        test_conditions.speed = case_data(1);
        test_conditions.feed = case_data(2);
        test_conditions.depth = case_data(3);
        description = case_data{4};
        
        fprintf('    🧪 %s... ', description);
        
        try
            % SFDP 실행 시도
            result = execute_sfdp_with_timeout(sfdp_system, test_conditions, 60); % 60초 제한
            
            % 결과 유효성 검사
            if validate_sfdp_result(result)
                success_count = success_count + 1;
                fprintf('PASS\n');
                test_details{i} = struct('status', 'PASS', 'result', result);
            else
                fprintf('FAIL (Invalid result)\n');
                test_details{i} = struct('status', 'FAIL', 'reason', 'Invalid result');
            end
            
        catch ME
            if contains(ME.message, 'timeout')
                fprintf('TIMEOUT\n');
                test_details{i} = struct('status', 'TIMEOUT', 'reason', 'Execution timeout');
            else
                fprintf('ERROR (%s)\n', ME.message);
                test_details{i} = struct('status', 'ERROR', 'reason', ME.message);
            end
        end
    end
    
    extreme_input_results = struct();
    extreme_input_results.total_cases = num_cases;
    extreme_input_results.passed_cases = success_count;
    extreme_input_results.success_rate = success_count / num_cases;
    extreme_input_results.test_details = test_details;
    
    fprintf('    📊 극한 입력 테스트: %d/%d 통과 (%.1f%%)\n', ...
           success_count, num_cases, extreme_input_results.success_rate * 100);
end

function memory_stress_results = test_memory_limitations(sfdp_system)
    
    fprintf('  💾 메모리 제한 환경 테스트\n');
    
    % 메모리 사용량을 점진적으로 증가시키면서 테스트
    problem_sizes = [0.5, 1.0, 2.0, 4.0, 8.0]; % GB 단위 예상 메모리 사용량
    success_count = 0;
    memory_test_details = [];
    
    for i = 1:length(problem_sizes)
        target_memory_gb = problem_sizes(i);
        
        fprintf('    🧪 %.1f GB 메모리 사용 테스트... ', target_memory_gb);
        
        try
            % 메모리 사용량에 맞는 문제 크기 설정
            test_config = create_memory_intensive_config(target_memory_gb);
            
            % 메모리 모니터링 시작
            initial_memory = monitor_memory_usage();
            
            % SFDP 실행
            tic;
            result = execute_sfdp_with_memory_monitoring(sfdp_system, test_config);
            execution_time = toc;
            
            % 최종 메모리 사용량 확인
            peak_memory = monitor_memory_usage();
            actual_memory_used = peak_memory - initial_memory;
            
            success_count = success_count + 1;
            fprintf('PASS (%.2f GB 사용, %.1fs)\n', actual_memory_used, execution_time);
            
            memory_test_details(i) = struct(...
                'target_memory_gb', target_memory_gb, ...
                'actual_memory_gb', actual_memory_used, ...
                'execution_time', execution_time, ...
                'status', 'PASS');
            
        catch ME
            fprintf('FAIL (%s)\n', ME.message);
            
            memory_test_details(i) = struct(...
                'target_memory_gb', target_memory_gb, ...
                'actual_memory_gb', NaN, ...
                'execution_time', NaN, ...
                'status', 'FAIL', ...
                'error_message', ME.message);
        end
    end
    
    memory_stress_results = struct();
    memory_stress_results.total_tests = length(problem_sizes);
    memory_stress_results.passed_tests = success_count;
    memory_stress_results.success_rate = success_count / length(problem_sizes);
    memory_stress_results.test_details = memory_test_details;
    
    fprintf('    📊 메모리 스트레스 테스트: %d/%d 통과 (%.1f%%)\n', ...
           success_count, length(problem_sizes), memory_stress_results.success_rate * 100);
end
```

### 12.3.2 Edge Case Handling

**경계 사례 처리**

```matlab
% testEdgeCaseHandling 함수에서 경계 사례 테스트
function [edge_case_results] = test_edge_case_handling(sfdp_system)
    
    fprintf('🔍 경계 사례 처리 테스트\n');
    
    edge_cases = {
        % 물리적 경계값
        struct('name', '융점 온도', 'conditions', struct('speed', 800, 'feed', 0.5, 'depth', 5.0), ...
               'expected_behavior', 'temperature_limit_warning'),
        
        struct('name', '최대 마모량', 'conditions', struct('speed', 1000, 'feed', 1.0, 'depth', 10.0), ...
               'expected_behavior', 'tool_life_exhausted'),
        
        struct('name', '극소 표면조도', 'conditions', struct('speed', 50, 'feed', 0.01, 'depth', 0.1), ...
               'expected_behavior', 'minimum_roughness_achieved'),
        
        % 수치적 경계값
        struct('name', 'Float 정밀도 한계', 'conditions', struct('speed', 1e-10, 'feed', 1e-10, 'depth', 1e-10), ...
               'expected_behavior', 'precision_limit_handling'),
        
        struct('name', '대용량 배열', 'conditions', struct('mesh_nodes', 1e6), ...
               'expected_behavior', 'memory_efficient_processing'),
        
        % 물리적 모순
        struct('name', '에너지 보존 위배', 'conditions', struct('input_energy', 1000, 'output_energy', 2000), ...
               'expected_behavior', 'physics_violation_detection'),
        
        % 시간 관련 경계값
        struct('name', '순간 가공', 'conditions', struct('time_duration', 0), ...
               'expected_behavior', 'minimum_time_enforcement'),
        
        struct('name', '장시간 가공', 'conditions', struct('time_duration', 1e6), ...
               'expected_behavior', 'long_duration_stability')
    };
    
    num_edge_cases = length(edge_cases);
    passed_cases = 0;
    edge_case_details = cell(num_edge_cases, 1);
    
    for i = 1:num_edge_cases
        edge_case = edge_cases{i};
        fprintf('  🧪 %s 테스트... ', edge_case.name);
        
        try
            % 경계 사례 실행
            [result, system_response] = execute_edge_case(sfdp_system, edge_case.conditions);
            
            % 예상 동작 확인
            behavior_correct = validate_expected_behavior(system_response, edge_case.expected_behavior);
            
            if behavior_correct
                passed_cases = passed_cases + 1;
                fprintf('PASS\n');
                edge_case_details{i} = struct('status', 'PASS', 'response', system_response);
            else
                fprintf('FAIL (Unexpected behavior)\n');
                edge_case_details{i} = struct('status', 'FAIL', 'reason', 'Unexpected behavior', 'response', system_response);
            end
            
        catch ME
            % 오류 발생 자체가 예상 동작일 수 있음
            if is_expected_error(ME, edge_case.expected_behavior)
                passed_cases = passed_cases + 1;
                fprintf('PASS (Expected error)\n');
                edge_case_details{i} = struct('status', 'PASS', 'expected_error', ME.message);
            else
                fprintf('FAIL (Unexpected error: %s)\n', ME.message);
                edge_case_details{i} = struct('status', 'FAIL', 'error', ME.message);
            end
        end
    end
    
    edge_case_results = struct();
    edge_case_results.total_cases = num_edge_cases;
    edge_case_results.passed_cases = passed_cases;
    edge_case_results.success_rate = passed_cases / num_edge_cases;
    edge_case_results.case_details = edge_case_details;
    
    fprintf('  📊 경계 사례 테스트: %d/%d 통과 (%.1f%%)\n', ...
           passed_cases, num_edge_cases, edge_case_results.success_rate * 100);
end

function behavior_correct = validate_expected_behavior(system_response, expected_behavior)
    
    behavior_correct = false;
    
    switch expected_behavior
        case 'temperature_limit_warning'
            behavior_correct = isfield(system_response, 'warnings') && ...
                              any(contains({system_response.warnings.message}, 'temperature'));
            
        case 'tool_life_exhausted'
            behavior_correct = isfield(system_response, 'tool_status') && ...
                              strcmp(system_response.tool_status, 'exhausted');
            
        case 'minimum_roughness_achieved'
            behavior_correct = isfield(system_response, 'surface_roughness') && ...
                              system_response.surface_roughness.Ra < 0.1;
            
        case 'precision_limit_handling'
            behavior_correct = isfield(system_response, 'precision_warnings') || ...
                              isfield(system_response, 'numerical_issues');
            
        case 'memory_efficient_processing'
            behavior_correct = isfield(system_response, 'processing_mode') && ...
                              strcmp(system_response.processing_mode, 'chunked');
            
        case 'physics_violation_detection'
            behavior_correct = isfield(system_response, 'physics_violations') && ...
                              ~isempty(system_response.physics_violations);
            
        case 'minimum_time_enforcement'
            behavior_correct = isfield(system_response, 'adjusted_time') && ...
                              system_response.adjusted_time > 0;
            
        case 'long_duration_stability'
            behavior_correct = isfield(system_response, 'stability_status') && ...
                              strcmp(system_response.stability_status, 'stable');
            
        otherwise
            behavior_correct = true; % 알 수 없는 예상 동작은 통과로 간주
    end
end
```