# Chapter 14: Integration with External Libraries

## 14.1 FEATool Multiphysics Integration

### 14.1.1 FEATool Interface Implementation

**FEATool Multiphysics와의 통합 구현**

SFDP v17.3은 FEATool Multiphysics와의 깊은 통합을 통해 고급 3D 유한요소 해석 기능을 제공합니다.

```matlab
% SFDP_physics_suite.m:151-200에서 FEATool 통합 구현
function [thermal_result, thermal_confidence] = calculate3DThermalFEATool(cutting_speed, feed_rate, depth_of_cut, material_props, simulation_state)
    
    fprintf('🔥 FEATool 3D 열해석 시작\n');
    
    thermal_result = struct();
    thermal_confidence = 0;
    
    try
        % 1. FEATool 가용성 확인
        featool_available = check_featool_availability();
        
        if ~featool_available
            warning('FEATool이 사용 불가능합니다. 대체 방법을 사용합니다.');
            [thermal_result, thermal_confidence] = fallback_to_analytical_thermal(...
                cutting_speed, feed_rate, depth_of_cut, material_props, simulation_state);
            return;
        end
        
        % 2. FEATool 모델 초기화
        fprintf('  📋 FEATool 모델 초기화\n');
        fea_model = initialize_featool_model(material_props, simulation_state);
        
        % 3. 3D 기하학 생성
        fprintf('  🏗️ 3D 워크피스 기하학 생성\n');
        workpiece_geometry = create_3d_workpiece_geometry(depth_of_cut, simulation_state);
        fea_model = featool('geom', fea_model, workpiece_geometry);
        
        % 4. 물리 방정식 설정
        fprintf('  ⚗️ 열전도 방정식 설정\n');
        fea_model = setup_heat_transfer_physics(fea_model, material_props);
        
        % 5. 메시 생성 및 적응형 세분화
        fprintf('  🕸️ 적응형 메시 생성\n');
        fea_model = generate_adaptive_mesh(fea_model, cutting_speed, simulation_state);
        
        % 6. 경계조건 적용
        fprintf('  🚧 고급 경계조건 적용\n');
        fea_model = apply_advanced_thermal_boundary_conditions(fea_model, cutting_speed, feed_rate, material_props);
        
        % 7. 움직이는 열원 구현
        fprintf('  🔄 움직이는 열원 구현\n');
        fea_model = implement_moving_heat_source(fea_model, cutting_speed, feed_rate, depth_of_cut);
        
        % 8. 시간 의존적 해석 실행
        fprintf('  ⏰ 시간 의존적 FEM 해석 실행\n');
        [fea_solution, solver_info] = solve_transient_thermal_fem(fea_model, simulation_state);
        
        % 9. 결과 후처리 및 검증
        fprintf('  📊 결과 후처리 및 검증\n');
        [thermal_result, thermal_confidence] = postprocess_thermal_results(fea_solution, solver_info, material_props);
        
        % 10. FEATool 세션 정리
        cleanup_featool_session(fea_model);
        
        fprintf('  ✅ FEATool 3D 열해석 완료 (신뢰도: %.3f)\n', thermal_confidence);
        
    catch ME
        fprintf('  ❌ FEATool 열해석 실패: %s\n', ME.message);
        
        % Graceful fallback to analytical method
        warning('FEATool 실행 실패. 해석적 방법으로 대체합니다.');
        [thermal_result, thermal_confidence] = fallback_to_analytical_thermal(...
            cutting_speed, feed_rate, depth_of_cut, material_props, simulation_state);
        
        % 신뢰도 조정 (fallback 사용시)
        thermal_confidence = thermal_confidence * 0.7;
    end
end

function fea_model = initialize_featool_model(material_props, simulation_state)
    
    % FEATool 모델 기본 설정
    fea_model = struct();
    fea_model.sdim = 3; % 3D 해석
    fea_model.dvar = {'T'}; % 온도 변수
    fea_model.prob = 'heat_transfer'; % 열전달 문제
    
    % 물리적 상수 설정
    fea_model.phys.ht.eqn.coef{1,end} = {material_props.density}; % 밀도 ρ
    fea_model.phys.ht.eqn.coef{2,end} = {material_props.thermal_conductivity}; % 열전도계수 k
    fea_model.phys.ht.eqn.coef{3,end} = {material_props.specific_heat}; % 비열 cp
    
    % 초기 조건
    fea_model.phys.ht.bdr.coef{1,end} = {simulation_state.ambient_temperature}; % 초기 온도
    
    % 솔버 설정
    fea_model.sol.fid = 1; % 파일 ID
    fea_model.sol.maxnit = 100; % 최대 반복 횟수
    fea_model.sol.nlrlx = 0.8; % 비선형 완화 계수
    fea_model.sol.tol = 1e-6; % 수렴 허용 오차
    
    % 시간 설정
    fea_model.sol.dt = simulation_state.time_step; % 시간 스텝
    fea_model.sol.tmax = simulation_state.total_time; % 총 시간
    
    fprintf('    ✅ FEATool 모델 초기화 완료\n');
end

function workpiece_geometry = create_3d_workpiece_geometry(depth_of_cut, simulation_state)
    
    % 워크피스 치수 정의
    workpiece_length = simulation_state.workpiece.length; % 50 mm
    workpiece_width = simulation_state.workpiece.width;   % 30 mm  
    workpiece_height = simulation_state.workpiece.height; % 20 mm
    
    % 3D 박스 기하학 생성
    workpiece_geometry = struct();
    workpiece_geometry.type = 'box';
    workpiece_geometry.params = [
        0, workpiece_length;  % x 범위
        0, workpiece_width;   % y 범위
        0, workpiece_height   % z 범위
    ];
    
    % 절삭 영역 정의 (상부 표면)
    cutting_zone_height = workpiece_height - depth_of_cut;
    
    workpiece_geometry.cutting_zone = struct();
    workpiece_geometry.cutting_zone.z_start = cutting_zone_height;
    workpiece_geometry.cutting_zone.z_end = workpiece_height;
    workpiece_geometry.cutting_zone.type = 'surface_region';
    
    % 도구 경로 정의
    workpiece_geometry.tool_path = struct();
    workpiece_geometry.tool_path.start_point = [5, workpiece_width/2, workpiece_height];
    workpiece_geometry.tool_path.end_point = [workpiece_length-5, workpiece_width/2, workpiece_height];
    workpiece_geometry.tool_path.direction = [1, 0, 0]; % x 방향
    
    fprintf('    🏗️ 3D 워크피스 기하학 생성 완료 (%.1f×%.1f×%.1f mm)\n', ...
           workpiece_length, workpiece_width, workpiece_height);
end

function fea_model = implement_moving_heat_source(fea_model, cutting_speed, feed_rate, depth_of_cut)
    
    % 열발생률 계산
    heat_generation_rate = calculate_heat_generation_rate(cutting_speed, feed_rate, depth_of_cut);
    
    % 움직이는 열원 매개변수
    cutting_velocity = cutting_speed / 60; % m/min → m/s
    heat_source_length = feed_rate * 2; % mm
    heat_source_width = depth_of_cut; % mm
    heat_source_depth = 0.5; % mm (침투 깊이)
    
    % 시간 의존적 열원 위치
    cutting_position_expr = sprintf('%.6f * t', cutting_velocity / 1000); % mm/s
    
    % 3D 가우시안 열원 표현식
    heat_source_expr = sprintf(['%.3e * exp(-((x-(%s))^2/(%.6f)^2 + ' ...
                               'y^2/(%.6f)^2 + (z-%.6f)^2/(%.6f)^2))'], ...
        heat_generation_rate, cutting_position_expr, heat_source_length/2, ...
        heat_source_width/2, fea_model.workpiece_height, heat_source_depth/2);
    
    % FEATool에 열원 적용
    fea_model.phys.ht.eqn.coef{4,end} = {heat_source_expr}; % 체적 열원
    
    fprintf('    🔥 움직이는 3D 가우시안 열원 구현 (%.2e W/m³)\n', heat_generation_rate);
end
```

### 14.1.2 Advanced FEM Capabilities

**고급 FEM 기능 활용**

```matlab
% generateAdaptiveMesh 함수에서 적응형 메시 생성
function fea_model = generate_adaptive_mesh(fea_model, cutting_speed, simulation_state)
    
    fprintf('    🕸️ 적응형 메시 생성 시작\n');
    
    % 기본 메시 설정
    base_mesh_size = simulation_state.mesh.base_size; % 1.0 mm
    cutting_zone_mesh_size = simulation_state.mesh.cutting_zone_size; % 0.2 mm
    
    % 절삭속도에 따른 메시 조정
    speed_factor = cutting_speed / 100; % 100 m/min 기준
    adapted_cutting_mesh_size = cutting_zone_mesh_size / sqrt(speed_factor);
    
    % 메시 크기 제한
    min_mesh_size = 0.05; % mm
    max_mesh_size = 2.0; % mm
    adapted_cutting_mesh_size = max(min_mesh_size, min(max_mesh_size, adapted_cutting_mesh_size));
    
    % FEATool 메시 생성
    try
        % 1단계: 조악한 초기 메시
        fea_model = featool('geommesh', fea_model, 'hmax', base_mesh_size, 'hgrad', 1.5);
        
        % 2단계: 절삭 영역 세분화
        cutting_zone_elements = identify_cutting_zone_elements(fea_model);
        fea_model = featool('meshadapt', fea_model, 'elements', cutting_zone_elements, ...
                           'hmax', adapted_cutting_mesh_size);
        
        % 3단계: 열원 주변 추가 세분화
        heat_source_elements = identify_heat_source_elements(fea_model, cutting_speed);
        fea_model = featool('meshadapt', fea_model, 'elements', heat_source_elements, ...
                           'hmax', adapted_cutting_mesh_size * 0.5);
        
        % 메시 품질 검사
        mesh_quality = evaluate_mesh_quality(fea_model);
        
        if mesh_quality.min_angle < 15 || mesh_quality.max_aspect_ratio > 10
            fprintf('    ⚠️ 메시 품질 불량 - 개선 시도\n');
            fea_model = improve_mesh_quality(fea_model);
        end
        
        fprintf('    ✅ 적응형 메시 생성 완료 (%d nodes, %d elements)\n', ...
               size(fea_model.grid.p, 2), size(fea_model.grid.c, 2));
        
    catch ME
        warning('적응형 메시 생성 실패: %s. 균등 메시 사용.', ME.message);
        fea_model = featool('geommesh', fea_model, 'hmax', cutting_zone_mesh_size);
    end
end

function [fea_solution, solver_info] = solve_transient_thermal_fem(fea_model, simulation_state)
    
    fprintf('    ⏰ 시간 의존적 FEM 해석 실행\n');
    
    solver_info = struct();
    
    try
        % 솔버 옵션 설정
        solver_options = struct();
        solver_options.solver = 'fensolver'; % FEATool 기본 솔버
        solver_options.method = 'implicit'; % 음해법 (안정성)
        solver_options.preconditioner = 'ilu'; % Incomplete LU
        solver_options.maxiter = 1000;
        solver_options.reltol = 1e-6;
        solver_options.abstol = 1e-8;
        
        % 시간 적분 설정
        time_settings = struct();
        time_settings.scheme = 'backward_euler'; % 1차 후진 오일러
        time_settings.dt = simulation_state.time_step;
        time_settings.t_end = simulation_state.total_time;
        time_settings.output_times = 0:time_settings.dt:time_settings.t_end;
        
        % 비선형 솔버 설정 (온도 의존적 물성치 고려)
        nonlinear_settings = struct();
        nonlinear_settings.enable = true;
        nonlinear_settings.max_iterations = 20;
        nonlinear_settings.tolerance = 1e-6;
        nonlinear_settings.relaxation = 0.8;
        
        % FEATool 솔버 실행
        tic;
        [fea_solution, solver_convergence] = featool('solve', fea_model, ...
            'solver_options', solver_options, ...
            'time_settings', time_settings, ...
            'nonlinear_settings', nonlinear_settings);
        
        solve_time = toc;
        
        % 솔버 정보 수집
        solver_info.solve_time = solve_time;
        solver_info.convergence_history = solver_convergence;
        solver_info.final_residual = solver_convergence.final_residual;
        solver_info.iterations_used = solver_convergence.iterations;
        solver_info.solver_success = solver_convergence.converged;
        
        if solver_info.solver_success
            fprintf('    ✅ FEM 해석 완료 (%.1fs, %d회 반복)\n', ...
                   solve_time, solver_info.iterations_used);
        else
            warning('FEM 솔버 수렴 실패');
            solver_info.solver_success = false;
        end
        
    catch ME
        fprintf('    ❌ FEM 해석 중 오류: %s\n', ME.message);
        fea_solution = [];
        solver_info.error = ME.message;
        solver_info.solver_success = false;
    end
end
```

## 14.2 GIBBON Integration

### 14.2.1 3D Contact Mechanics with GIBBON

**GIBBON을 활용한 3D 접촉역학 해석**

```matlab
% calculateCoupledWearGIBBON 함수에서 GIBBON 통합 (Lines 481-560)
function [contact_results, contact_confidence] = calculateCoupledWearGIBBON(cutting_speed, feed_rate, depth_of_cut, material_props, thermal_results, simulation_state)
    
    fprintf('🤝 GIBBON 3D 접촉역학 해석 시작\n');
    
    contact_results = struct();
    contact_confidence = 0;
    
    try
        % 1. GIBBON 가용성 확인
        gibbon_available = check_gibbon_availability();
        
        if ~gibbon_available
            warning('GIBBON이 사용 불가능합니다. 단순화된 접촉 모델을 사용합니다.');
            [contact_results, contact_confidence] = fallback_to_hertz_contact(...
                cutting_speed, feed_rate, depth_of_cut, material_props, thermal_results);
            return;
        end
        
        % 2. 3D 접촉 기하학 생성
        fprintf('  🏗️ 3D 접촉 기하학 생성\n');
        [tool_geometry, workpiece_geometry] = create_3d_contact_geometry(...
            cutting_speed, feed_rate, depth_of_cut, simulation_state);
        
        % 3. 온도 의존적 재료 특성 계산
        fprintf('  🌡️ 온도 의존적 재료 특성 계산\n');
        temperature_dependent_props = calculate_temperature_dependent_properties(...
            material_props, thermal_results);
        
        % 4. GIBBON FEM 메시 생성
        fprintf('  🕸️ GIBBON 접촉 메시 생성\n');
        [contact_mesh, mesh_quality] = generate_gibbon_contact_mesh(...
            tool_geometry, workpiece_geometry, simulation_state);
        
        % 5. 접촉 문제 설정
        fprintf('  ⚙️ 접촉 문제 설정\n');
        contact_problem = setup_gibbon_contact_problem(contact_mesh, ...
            temperature_dependent_props, simulation_state);
        
        % 6. 3D 접촉 해석 실행
        fprintf('  🔄 3D 접촉 FEM 해석 실행\n');
        [gibbon_solution, contact_info] = solve_gibbon_contact_problem(...
            contact_problem, cutting_speed, feed_rate);
        
        % 7. 접촉 결과 후처리
        fprintf('  📊 접촉 결과 후처리\n');
        [contact_results, contact_confidence] = postprocess_gibbon_contact_results(...
            gibbon_solution, contact_info, material_props);
        
        % 8. 접촉 검증 및 품질 평가
        contact_validation = validate_contact_results(contact_results, thermal_results);
        contact_confidence = contact_confidence * contact_validation.confidence_factor;
        
        fprintf('  ✅ GIBBON 접촉 해석 완료 (신뢰도: %.3f)\n', contact_confidence);
        
    catch ME
        fprintf('  ❌ GIBBON 접촉 해석 실패: %s\n', ME.message);
        
        % Graceful fallback
        [contact_results, contact_confidence] = fallback_to_hertz_contact(...
            cutting_speed, feed_rate, depth_of_cut, material_props, thermal_results);
        contact_confidence = contact_confidence * 0.6; % 신뢰도 감소
    end
end

function [tool_geometry, workpiece_geometry] = create_3d_contact_geometry(cutting_speed, feed_rate, depth_of_cut, simulation_state)
    
    % 도구 기하학 정의
    tool_geometry = struct();
    tool_geometry.type = 'end_mill';
    tool_geometry.diameter = simulation_state.tool.diameter; % 10 mm
    tool_geometry.length = simulation_state.tool.length; % 50 mm
    tool_geometry.helix_angle = simulation_state.tool.helix_angle; % 30 degrees
    tool_geometry.number_of_flutes = simulation_state.tool.flutes; % 4
    
    % 도구 재료 특성
    tool_geometry.material = struct();
    tool_geometry.material.youngs_modulus = 600e9; % Pa (carbide)
    tool_geometry.material.poisson_ratio = 0.25;
    tool_geometry.material.density = 15000; % kg/m³
    tool_geometry.material.hardness = 1800; % HV
    
    % 도구 형상 매개변수화
    tool_radius = tool_geometry.diameter / 2;
    
    % GIBBON에서 사용할 도구 표면 메시 생성
    tool_geometry.surface_mesh = generate_tool_surface_mesh(tool_geometry);
    
    % 워크피스 기하학 정의
    workpiece_geometry = struct();
    workpiece_geometry.type = 'rectangular_block';
    workpiece_geometry.dimensions = [
        simulation_state.workpiece.length,  % 50 mm
        simulation_state.workpiece.width,   % 30 mm
        simulation_state.workpiece.height   % 20 mm
    ];
    
    % 절삭 영역 정의
    workpiece_geometry.cutting_region = struct();
    workpiece_geometry.cutting_region.depth = depth_of_cut;
    workpiece_geometry.cutting_region.width = feed_rate * 5; % 접촉 폭
    workpiece_geometry.cutting_region.length = tool_geometry.diameter * 1.5;
    
    % 워크피스 표면 메시 생성
    workpiece_geometry.surface_mesh = generate_workpiece_surface_mesh(workpiece_geometry);
    
    fprintf('    🏗️ 접촉 기하학 생성 완료\n');
    fprintf('      도구: Ø%.1fmm, %d날 엔드밀\n', tool_geometry.diameter, tool_geometry.number_of_flutes);
    fprintf('      워크피스: %.1f×%.1f×%.1fmm\n', workpiece_geometry.dimensions);
end

function [contact_mesh, mesh_quality] = generate_gibbon_contact_mesh(tool_geometry, workpiece_geometry, simulation_state)
    
    % GIBBON 메시 생성 매개변수
    mesh_params = struct();
    mesh_params.element_size = simulation_state.contact_mesh.element_size; % 0.5 mm
    mesh_params.surface_refinement = 2; % 접촉면 2배 세분화
    mesh_params.contact_zone_refinement = 4; % 접촉 영역 4배 세분화
    
    try
        % 1. 도구 메시 생성
        fprintf('    🔧 도구 메시 생성\n');
        tool_mesh = gibbon_create_tool_mesh(tool_geometry, mesh_params);
        
        % 2. 워크피스 메시 생성
        fprintf('    📦 워크피스 메시 생성\n');
        workpiece_mesh = gibbon_create_workpiece_mesh(workpiece_geometry, mesh_params);
        
        % 3. 접촉면 식별 및 세분화
        fprintf('    🤝 접촉면 식별 및 세분화\n');
        [contact_surfaces, contact_pairs] = identify_contact_surfaces(tool_mesh, workpiece_mesh);
        
        % 4. 접촉 메시 통합
        contact_mesh = struct();
        contact_mesh.tool = tool_mesh;
        contact_mesh.workpiece = workpiece_mesh;
        contact_mesh.contact_surfaces = contact_surfaces;
        contact_mesh.contact_pairs = contact_pairs;
        
        % 5. 메시 품질 평가
        mesh_quality = evaluate_gibbon_mesh_quality(contact_mesh);
        
        if mesh_quality.overall_score < 0.7
            fprintf('    ⚠️ 메시 품질 개선 필요\n');
            contact_mesh = improve_gibbon_mesh_quality(contact_mesh);
            mesh_quality = evaluate_gibbon_mesh_quality(contact_mesh);
        end
        
        fprintf('    ✅ GIBBON 메시 생성 완료\n');
        fprintf('      도구 요소: %d개, 워크피스 요소: %d개\n', ...
               size(tool_mesh.elements, 1), size(workpiece_mesh.elements, 1));
        fprintf('      메시 품질: %.2f/1.0\n', mesh_quality.overall_score);
        
    catch ME
        error('GIBBON 메시 생성 실패: %s', ME.message);
    end
end

function [gibbon_solution, contact_info] = solve_gibbon_contact_problem(contact_problem, cutting_speed, feed_rate)
    
    fprintf('    🔄 GIBBON 접촉 문제 해석\n');
    
    % GIBBON FEBio 솔버 설정
    solver_settings = struct();
    solver_settings.analysis_type = 'static'; % 정적 해석
    solver_settings.contact_algorithm = 'augmented_lagrange';
    solver_settings.penalty_factor = 1e5;
    solver_settings.augmentation_tolerance = 0.1;
    solver_settings.max_augmentations = 50;
    
    % 수렴 기준
    solver_settings.convergence = struct();
    solver_settings.convergence.max_iterations = 100;
    solver_settings.convergence.displacement_tolerance = 1e-6;
    solver_settings.convergence.force_tolerance = 1e-3;
    solver_settings.convergence.energy_tolerance = 1e-6;
    
    % 하중 조건 설정
    cutting_force = estimate_cutting_force(cutting_speed, feed_rate);
    solver_settings.applied_loads = struct();
    solver_settings.applied_loads.cutting_force = cutting_force;
    solver_settings.applied_loads.feed_force = cutting_force * 0.3;
    solver_settings.applied_loads.thrust_force = cutting_force * 0.5;
    
    try
        % FEBio 입력 파일 생성
        febio_input_file = generate_febio_input_file(contact_problem, solver_settings);
        
        % GIBBON을 통한 FEBio 실행
        tic;
        [febio_results, run_info] = runMonitorFEBio(febio_input_file);
        solve_time = toc;
        
        % 해석 결과 확인
        if run_info.run_flag == 1
            fprintf('      ✅ FEBio 해석 성공 (%.1fs)\n', solve_time);
            
            % 결과 후처리
            gibbon_solution = process_febio_results(febio_results, contact_problem);
            
            contact_info = struct();
            contact_info.solve_time = solve_time;
            contact_info.iterations = run_info.iterations;
            contact_info.convergence_achieved = true;
            contact_info.final_residual = run_info.final_residual;
            
        else
            error('FEBio 해석 실패: %s', run_info.error_message);
        end
        
    catch ME
        fprintf('      ❌ GIBBON 해석 오류: %s\n', ME.message);
        gibbon_solution = [];
        contact_info = struct('error', ME.message, 'convergence_achieved', false);
    end
end
```

## 14.3 Machine Learning Library Integration

### 14.3.1 MATLAB Statistics and Machine Learning Toolbox

**MATLAB 통계 및 머신러닝 툴박스 통합**

```matlab
% SFDP_empirical_ml_suite.m에서 ML 툴박스 통합
function [ml_models, training_performance] = train_integrated_ml_models(training_data, validation_data, ml_config)
    
    fprintf('🤖 통합 머신러닝 모델 훈련 시작\n');
    
    ml_models = struct();
    training_performance = struct();
    
    % 1. Statistics Toolbox 가용성 확인
    stats_available = license('test', 'Statistics_Toolbox');
    ml_available = license('test', 'Neural_Network_Toolbox');
    
    if ~stats_available
        warning('Statistics and Machine Learning Toolbox가 없습니다. 기본 구현을 사용합니다.');
        [ml_models, training_performance] = fallback_ml_implementation(training_data, validation_data);
        return;
    end
    
    fprintf('  ✅ Statistics and Machine Learning Toolbox 사용 가능\n');
    
    % 2. 특성 엔지니어링
    fprintf('  🔧 고급 특성 엔지니어링\n');
    [engineered_features, feature_info] = perform_advanced_feature_engineering(training_data, ml_config);
    
    % 3. Random Forest 모델 훈련
    fprintf('  🌳 Random Forest 모델 훈련\n');
    [rf_model, rf_performance] = train_optimized_random_forest(engineered_features, ml_config);
    ml_models.random_forest = rf_model;
    training_performance.random_forest = rf_performance;
    
    % 4. Support Vector Machine 훈련
    fprintf('  🎯 SVM 모델 훈련\n');
    [svm_model, svm_performance] = train_optimized_svm(engineered_features, ml_config);
    ml_models.svm = svm_model;
    training_performance.svm = svm_performance;
    
    % 5. Gaussian Process Regression (고급 기능)
    if ml_config.enable_gaussian_process
        fprintf('  📊 Gaussian Process 모델 훈련\n');
        [gpr_model, gpr_performance] = train_gaussian_process_regression(engineered_features, ml_config);
        ml_models.gaussian_process = gpr_model;
        training_performance.gaussian_process = gpr_performance;
    end
    
    % 6. Neural Network (Deep Learning Toolbox 사용)
    if ml_available && ml_config.enable_neural_networks
        fprintf('  🧠 신경망 모델 훈련\n');
        [nn_model, nn_performance] = train_advanced_neural_network(engineered_features, ml_config);
        ml_models.neural_network = nn_model;
        training_performance.neural_network = nn_performance;
    end
    
    % 7. 앙상블 모델 생성
    fprintf('  🎼 앙상블 모델 구성\n');
    [ensemble_model, ensemble_performance] = create_ensemble_model(ml_models, validation_data);
    ml_models.ensemble = ensemble_model;
    training_performance.ensemble = ensemble_performance;
    
    % 8. 모델 성능 비교 및 선택
    fprintf('  📊 모델 성능 비교\n');
    [best_model, model_comparison] = compare_and_select_best_model(ml_models, training_performance);
    ml_models.best_model = best_model;
    training_performance.model_comparison = model_comparison;
    
    fprintf('🤖 통합 ML 모델 훈련 완료: %s 선택됨\n', best_model.type);
end

function [rf_model, rf_performance] = train_optimized_random_forest(training_data, ml_config)
    
    % Random Forest 하이퍼파라미터 최적화
    fprintf('    🔍 Random Forest 하이퍼파라미터 최적화\n');
    
    % 최적화할 하이퍼파라미터 정의
    rf_hyperparams = struct();
    rf_hyperparams.NumTrees = optimizableVariable('NumTrees', [50, 500], 'Type', 'integer');
    rf_hyperparams.MinLeafSize = optimizableVariable('MinLeafSize', [1, 20], 'Type', 'integer');
    rf_hyperparams.MaxNumSplits = optimizableVariable('MaxNumSplits', [10, 1000], 'Type', 'integer');
    rf_hyperparams.NumVariablesToSample = optimizableVariable('NumVariablesToSample', [1, size(training_data.features, 2)], 'Type', 'integer');
    
    % 베이지안 최적화 설정
    optimization_options = struct();
    optimization_options.AcquisitionFunctionName = 'expected-improvement-plus';
    optimization_options.MaxObjectiveEvaluations = 30;
    optimization_options.UseParallel = ml_config.use_parallel;
    optimization_options.Verbose = 0;
    
    % 목적 함수 정의 (교차 검증 오차)
    objective_function = @(params) rf_objective_function(params, training_data);
    
    try
        % 베이지안 최적화 실행
        [optimal_params, min_error] = bayesopt(objective_function, ...
            struct2table(rf_hyperparams), optimization_options);
        
        % 최적 매개변수로 모델 훈련
        rf_model = TreeBagger(optimal_params.NumTrees, training_data.features, training_data.targets, ...
            'Method', 'regression', ...
            'MinLeafSize', optimal_params.MinLeafSize, ...
            'MaxNumSplits', optimal_params.MaxNumSplits, ...
            'NumVariablesToSample', optimal_params.NumVariablesToSample, ...
            'OOBPrediction', 'on', ...
            'OOBPredictorImportance', 'on');
        
        % 성능 평가
        rf_performance = evaluate_rf_performance(rf_model, training_data);
        rf_performance.optimization_error = min_error;
        rf_performance.optimal_params = optimal_params;
        
        fprintf('      ✅ Random Forest 최적화 완료 (CV 오차: %.4f)\n', min_error);
        
    catch ME
        warning('Random Forest 최적화 실패: %s. 기본 매개변수 사용.', ME.message);
        
        % 기본 매개변수로 모델 훈련
        rf_model = TreeBagger(100, training_data.features, training_data.targets, ...
            'Method', 'regression', 'OOBPrediction', 'on');
        
        rf_performance = evaluate_rf_performance(rf_model, training_data);
        rf_performance.optimization_error = NaN;
    end
end

function [gpr_model, gpr_performance] = train_gaussian_process_regression(training_data, ml_config)
    
    fprintf('    📊 Gaussian Process Regression 훈련\n');
    
    try
        % GPR 커널 함수 정의
        kernel_functions = {
            'matern32',
            'matern52', 
            'squaredexponential',
            'exponential',
            'rationalquadratic'
        };
        
        best_gpr = [];
        best_performance = inf;
        best_kernel = '';
        
        % 다양한 커널로 모델 훈련 및 비교
        for i = 1:length(kernel_functions)
            kernel = kernel_functions{i};
            
            try
                % GPR 모델 훈련
                gpr_temp = fitrgp(training_data.features, training_data.targets, ...
                    'KernelFunction', kernel, ...
                    'OptimizeHyperparameters', 'auto', ...
                    'HyperparameterOptimizationOptions', ...
                    struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
                           'MaxObjectiveEvaluations', 20, ...
                           'Verbose', 0));
                
                % 교차 검증으로 성능 평가
                cv_loss = kfoldLoss(crossval(gpr_temp, 'KFold', 5));
                
                if cv_loss < best_performance
                    best_performance = cv_loss;
                    best_gpr = gpr_temp;
                    best_kernel = kernel;
                end
                
                fprintf('      %s 커널: CV 손실 = %.4f\n', kernel, cv_loss);
                
            catch kernel_error
                fprintf('      %s 커널 실패: %s\n', kernel, kernel_error.message);
            end
        end
        
        if ~isempty(best_gpr)
            gpr_model = best_gpr;
            
            % 성능 지표 계산
            gpr_performance = struct();
            gpr_performance.cv_loss = best_performance;
            gpr_performance.best_kernel = best_kernel;
            gpr_performance.hyperparameters = best_gpr.KernelInformation;
            
            % 불확실성 정량화 능력 평가
            [predictions, prediction_intervals] = predict(gpr_model, training_data.features);
            gpr_performance.uncertainty_quality = evaluate_uncertainty_quality(predictions, prediction_intervals, training_data.targets);
            
            fprintf('      ✅ GPR 훈련 완료: %s 커널 (CV 손실: %.4f)\n', best_kernel, best_performance);
        else
            error('모든 GPR 커널 훈련 실패');
        end
        
    catch ME
        warning('Gaussian Process Regression 훈련 실패: %s', ME.message);
        gpr_model = [];
        gpr_performance = struct('error', ME.message);
    end
end
```

### 14.3.2 External Python Integration

**외부 Python 라이브러리 통합**

```matlab
% integratePythonML 함수에서 Python 머신러닝 라이브러리 통합
function [python_ml_results] = integrate_python_ml_libraries(training_data, ml_config)
    
    fprintf('🐍 Python 머신러닝 라이브러리 통합\n');
    
    python_ml_results = struct();
    
    % 1. Python 환경 확인
    python_available = check_python_environment();
    
    if ~python_available
        warning('Python 환경이 설정되지 않았습니다. MATLAB 전용 모델을 사용합니다.');
        python_ml_results.status = 'unavailable';
        return;
    end
    
    fprintf('  ✅ Python 환경 확인됨\n');
    
    try
        % 2. 필요한 Python 패키지 확인
        required_packages = {'scikit-learn', 'xgboost', 'lightgbm', 'tensorflow', 'torch'};
        available_packages = check_python_packages(required_packages);
        
        % 3. XGBoost 모델 (고성능 gradient boosting)
        if available_packages.xgboost
            fprintf('  🚀 XGBoost 모델 훈련\n');
            xgboost_results = train_xgboost_model(training_data, ml_config);
            python_ml_results.xgboost = xgboost_results;
        end
        
        % 4. LightGBM 모델 (빠른 gradient boosting)
        if available_packages.lightgbm
            fprintf('  💡 LightGBM 모델 훈련\n');
            lightgbm_results = train_lightgbm_model(training_data, ml_config);
            python_ml_results.lightgbm = lightgbm_results;
        end
        
        % 5. TensorFlow/Keras 딥러닝 모델
        if available_packages.tensorflow && ml_config.enable_deep_learning
            fprintf('  🧠 TensorFlow 딥러닝 모델 훈련\n');
            tensorflow_results = train_tensorflow_model(training_data, ml_config);
            python_ml_results.tensorflow = tensorflow_results;
        end
        
        % 6. PyTorch 딥러닝 모델
        if available_packages.torch && ml_config.enable_deep_learning
            fprintf('  🔥 PyTorch 딥러닝 모델 훈련\n');
            pytorch_results = train_pytorch_model(training_data, ml_config);
            python_ml_results.pytorch = pytorch_results;
        end
        
        % 7. Scikit-learn 앙상블 모델
        if available_packages.sklearn
            fprintf('  🔬 Scikit-learn 앙상블 모델\n');
            sklearn_results = train_sklearn_ensemble(training_data, ml_config);
            python_ml_results.sklearn = sklearn_results;
        end
        
        python_ml_results.status = 'success';
        python_ml_results.available_packages = available_packages;
        
        fprintf('🐍 Python ML 통합 완료\n');
        
    catch ME
        fprintf('❌ Python ML 통합 실패: %s\n', ME.message);
        python_ml_results.status = 'failed';
        python_ml_results.error = ME.message;
    end
end

function xgboost_results = train_xgboost_model(training_data, ml_config)
    
    % Python XGBoost 스크립트 생성
    python_script = generate_xgboost_training_script(training_data, ml_config);
    
    % 임시 파일에 데이터 저장
    temp_data_file = save_training_data_for_python(training_data);
    
    try
        % Python 스크립트 실행
        cmd = sprintf('python "%s" "%s"', python_script, temp_data_file);
        [status, result] = system(cmd);
        
        if status == 0
            % 결과 로드
            xgboost_results = load_python_results('xgboost_results.mat');
            
            % 성능 지표 추가
            xgboost_results.model_type = 'XGBoost';
            xgboost_results.training_time = xgboost_results.training_time;
            xgboost_results.feature_importance = xgboost_results.feature_importance;
            
            fprintf('    ✅ XGBoost 훈련 완료 (R² = %.4f)\n', xgboost_results.r2_score);
        else
            error('XGBoost 훈련 실패: %s', result);
        end
        
    catch ME
        warning('XGBoost 실행 오류: %s', ME.message);
        xgboost_results = struct('error', ME.message);
    finally
        % 임시 파일 정리
        cleanup_temp_files({temp_data_file, python_script, 'xgboost_results.mat'});
    end
end

function python_script_path = generate_xgboost_training_script(training_data, ml_config)
    
    % XGBoost Python 스크립트 생성
    script_content = [
        'import numpy as np\n'
        'import pandas as pd\n'
        'import xgboost as xgb\n'
        'from sklearn.model_selection import train_test_split, GridSearchCV\n'
        'from sklearn.metrics import mean_squared_error, r2_score\n'
        'import scipy.io as sio\n'
        'import sys\n'
        'import time\n'
        '\n'
        '# 데이터 로드\n'
        'data_file = sys.argv[1]\n'
        'data = sio.loadmat(data_file)\n'
        'X = data["features"]\n'
        'y = data["targets"].flatten()\n'
        '\n'
        '# 훈련/검증 분할\n'
        'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n'
        '\n'
        '# XGBoost 하이퍼파라미터 그리드\n'
        'param_grid = {\n'
        '    "n_estimators": [100, 200, 300],\n'
        '    "max_depth": [3, 6, 9],\n'
        '    "learning_rate": [0.01, 0.1, 0.2],\n'
        '    "subsample": [0.8, 0.9, 1.0]\n'
        '}\n'
        '\n'
        '# Grid Search with Cross Validation\n'
        'start_time = time.time()\n'
        'xgb_regressor = xgb.XGBRegressor(random_state=42)\n'
        'grid_search = GridSearchCV(xgb_regressor, param_grid, cv=5, scoring="r2", n_jobs=-1)\n'
        'grid_search.fit(X_train, y_train)\n'
        'training_time = time.time() - start_time\n'
        '\n'
        '# 최적 모델 평가\n'
        'best_model = grid_search.best_estimator_\n'
        'y_pred = best_model.predict(X_test)\n'
        '\n'
        '# 성능 지표 계산\n'
        'mse = mean_squared_error(y_test, y_pred)\n'
        'r2 = r2_score(y_test, y_pred)\n'
        'feature_importance = best_model.feature_importances_\n'
        '\n'
        '# 결과 저장\n'
        'results = {\n'
        '    "best_params": grid_search.best_params_,\n'
        '    "training_time": training_time,\n'
        '    "mse": mse,\n'
        '    "r2_score": r2,\n'
        '    "feature_importance": feature_importance,\n'
        '    "cv_score": grid_search.best_score_\n'
        '}\n'
        '\n'
        'sio.savemat("xgboost_results.mat", results)\n'
        'print(f"XGBoost training completed. R2 = {r2:.4f}")\n'
    ];
    
    % 스크립트 파일 저장
    python_script_path = 'temp_xgboost_training.py';
    fid = fopen(python_script_path, 'w');
    fprintf(fid, '%s', script_content);
    fclose(fid);
end
```

## 14.4 Compatibility and Version Management

### 14.4.1 Toolbox Version Compatibility

**툴박스 버전 호환성 관리**

```matlab
% checkToolboxCompatibility 함수에서 호환성 확인
function [compatibility_report] = check_toolbox_compatibility()
    
    fprintf('🔍 툴박스 호환성 검사 시작\n');
    
    compatibility_report = struct();
    compatibility_report.matlab_version = version;
    compatibility_report.matlab_release = version('-release');
    compatibility_report.check_timestamp = datetime('now');
    
    % 필수 툴박스 목록과 최소 버전
    required_toolboxes = {
        'Symbolic Math Toolbox', '8.0';
        'Curve Fitting Toolbox', '3.5';
        'Statistics and Machine Learning Toolbox', '11.0';
        'Optimization Toolbox', '8.0';
        'Parallel Computing Toolbox', '6.0'
    };
    
    % 선택적 툴박스 목록
    optional_toolboxes = {
        'Deep Learning Toolbox', '12.0';
        'Financial Toolbox', '5.0';
        'Signal Processing Toolbox', '8.0'
    };
    
    % 외부 툴박스 목록
    external_toolboxes = {
        'FEATool Multiphysics', '1.17';
        'GIBBON', '3.5';
        'CFDTool', '1.10';
        'Iso2Mesh', '1.9'
    };
    
    % 1. 필수 툴박스 확인
    fprintf('  📋 필수 툴박스 확인\n');
    compatibility_report.required = check_toolbox_list(required_toolboxes, 'required');
    
    % 2. 선택적 툴박스 확인
    fprintf('  🔧 선택적 툴박스 확인\n');
    compatibility_report.optional = check_toolbox_list(optional_toolboxes, 'optional');
    
    % 3. 외부 툴박스 확인
    fprintf('  🌐 외부 툴박스 확인\n');
    compatibility_report.external = check_external_toolbox_list(external_toolboxes);
    
    % 4. 호환성 등급 결정
    [overall_grade, missing_critical] = determine_compatibility_grade(compatibility_report);
    compatibility_report.overall_grade = overall_grade;
    compatibility_report.missing_critical = missing_critical;
    
    % 5. 권장사항 생성
    compatibility_report.recommendations = generate_compatibility_recommendations(compatibility_report);
    
    fprintf('🔍 호환성 검사 완료: %s\n', overall_grade);
    
    if ~isempty(missing_critical)
        fprintf('⚠️ 누락된 필수 툴박스: %s\n', strjoin(missing_critical, ', '));
    end
end

function toolbox_results = check_toolbox_list(toolbox_list, category)
    
    toolbox_results = struct();
    toolbox_results.category = category;
    toolbox_results.total_checked = size(toolbox_list, 1);
    toolbox_results.available_count = 0;
    toolbox_results.details = cell(size(toolbox_list, 1), 1);
    
    for i = 1:size(toolbox_list, 1)
        toolbox_name = toolbox_list{i, 1};
        required_version = toolbox_list{i, 2};
        
        % 툴박스 가용성 확인
        [is_available, installed_version, license_valid] = check_individual_toolbox(toolbox_name);
        
        % 버전 호환성 확인
        version_compatible = false;
        if is_available && ~isempty(installed_version)
            version_compatible = compare_versions(installed_version, required_version) >= 0;
        end
        
        % 결과 기록
        toolbox_info = struct();
        toolbox_info.name = toolbox_name;
        toolbox_info.required_version = required_version;
        toolbox_info.installed_version = installed_version;
        toolbox_info.is_available = is_available;
        toolbox_info.license_valid = license_valid;
        toolbox_info.version_compatible = version_compatible;
        toolbox_info.overall_status = is_available && license_valid && version_compatible;
        
        toolbox_results.details{i} = toolbox_info;
        
        if toolbox_info.overall_status
            toolbox_results.available_count = toolbox_results.available_count + 1;
        end
        
        % 상태 출력
        if toolbox_info.overall_status
            fprintf('    ✅ %s (v%s)\n', toolbox_name, installed_version);
        elseif is_available && license_valid && ~version_compatible
            fprintf('    ⚠️ %s (v%s < v%s 필요)\n', toolbox_name, installed_version, required_version);
        elseif is_available && ~license_valid
            fprintf('    🔒 %s (라이센스 없음)\n', toolbox_name);
        else
            fprintf('    ❌ %s (미설치)\n', toolbox_name);
        end
    end
    
    toolbox_results.availability_rate = toolbox_results.available_count / toolbox_results.total_checked;
    
    fprintf('    📊 %s 툴박스: %d/%d 사용 가능 (%.1f%%)\n', ...
           category, toolbox_results.available_count, toolbox_results.total_checked, ...
           toolbox_results.availability_rate * 100);
end

function [is_available, version_str, license_valid] = check_individual_toolbox(toolbox_name)
    
    is_available = false;
    version_str = '';
    license_valid = false;
    
    try
        % 툴박스별 특화된 확인 방법
        switch toolbox_name
            case 'Symbolic Math Toolbox'
                license_valid = license('test', 'Symbolic_Toolbox');
                if license_valid
                    v = ver('symbolic');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            case 'Statistics and Machine Learning Toolbox'
                license_valid = license('test', 'Statistics_Toolbox');
                if license_valid
                    v = ver('stats');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            case 'Curve Fitting Toolbox'
                license_valid = license('test', 'Curve_Fitting_Toolbox');
                if license_valid
                    v = ver('curvefit');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            case 'Optimization Toolbox'
                license_valid = license('test', 'Optimization_Toolbox');
                if license_valid
                    v = ver('optim');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            case 'Parallel Computing Toolbox'
                license_valid = license('test', 'Distrib_Computing_Toolbox');
                if license_valid
                    v = ver('parallel');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            case 'Deep Learning Toolbox'
                license_valid = license('test', 'Neural_Network_Toolbox');
                if license_valid
                    v = ver('nnet');
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                    end
                end
                
            otherwise
                % 일반적인 방법으로 확인
                try
                    v = ver(lower(strrep(toolbox_name, ' ', '')));
                    if ~isempty(v)
                        is_available = true;
                        version_str = v.Version;
                        license_valid = true; % 설치되어 있다면 라이센스 유효로 가정
                    end
                catch
                    % 확인 실패
                end
        end
        
    catch ME
        % 오류 발생시 사용 불가로 처리
        warning('툴박스 확인 중 오류: %s', ME.message);
    end
end

function external_results = check_external_toolbox_list(external_toolboxes)
    
    external_results = struct();
    external_results.total_checked = size(external_toolboxes, 1);
    external_results.available_count = 0;
    external_results.details = cell(size(external_toolboxes, 1), 1);
    
    for i = 1:size(external_toolboxes, 1)
        toolbox_name = external_toolboxes{i, 1};
        required_version = external_toolboxes{i, 2};
        
        [is_available, installed_version] = check_external_toolbox(toolbox_name);
        
        % 버전 호환성 확인
        version_compatible = false;
        if is_available && ~isempty(installed_version)
            version_compatible = compare_versions(installed_version, required_version) >= 0;
        end
        
        toolbox_info = struct();
        toolbox_info.name = toolbox_name;
        toolbox_info.required_version = required_version;
        toolbox_info.installed_version = installed_version;
        toolbox_info.is_available = is_available;
        toolbox_info.version_compatible = version_compatible;
        toolbox_info.overall_status = is_available && version_compatible;
        
        external_results.details{i} = toolbox_info;
        
        if toolbox_info.overall_status
            external_results.available_count = external_results.available_count + 1;
        end
        
        % 상태 출력
        if toolbox_info.overall_status
            fprintf('    ✅ %s (v%s)\n', toolbox_name, installed_version);
        elseif is_available && ~version_compatible
            fprintf('    ⚠️ %s (v%s < v%s 필요)\n', toolbox_name, installed_version, required_version);
        else
            fprintf('    ❌ %s (미설치)\n', toolbox_name);
        end
    end
    
    external_results.availability_rate = external_results.available_count / external_results.total_checked;
    
    fprintf('    📊 외부 툴박스: %d/%d 사용 가능 (%.1f%%)\n', ...
           external_results.available_count, external_results.total_checked, ...
           external_results.availability_rate * 100);
end

function [is_available, version_str] = check_external_toolbox(toolbox_name)
    
    is_available = false;
    version_str = '';
    
    switch toolbox_name
        case 'FEATool Multiphysics'
            % FEATool 확인
            try
                if exist('featool', 'file') == 2
                    is_available = true;
                    % 버전 정보 추출 시도
                    try
                        version_info = featool('version');
                        if isstruct(version_info) && isfield(version_info, 'version')
                            version_str = version_info.version;
                        else
                            version_str = '1.17'; % 기본값
                        end
                    catch
                        version_str = '1.17'; % 기본값
                    end
                end
            catch
                % FEATool 확인 실패
            end
            
        case 'GIBBON'
            % GIBBON 확인
            try
                if exist('gibbon_version', 'file') == 2
                    is_available = true;
                    try
                        version_str = gibbon_version();
                    catch
                        version_str = '3.5'; % 기본값
                    end
                elseif exist('gibbonver', 'file') == 2
                    is_available = true;
                    try
                        version_str = gibbonver();
                    catch
                        version_str = '3.5'; % 기본값
                    end
                end
            catch
                % GIBBON 확인 실패
            end
            
        case 'CFDTool'
            % CFDTool 확인
            try
                if exist('cfdtool', 'file') == 2
                    is_available = true;
                    version_str = '1.10'; % 기본값
                end
            catch
                % CFDTool 확인 실패
            end
            
        case 'Iso2Mesh'
            % Iso2Mesh 확인
            try
                if exist('iso2mesh_version', 'file') == 2
                    is_available = true;
                    try
                        version_str = iso2mesh_version();
                    catch
                        version_str = '1.9'; % 기본값
                    end
                elseif exist('vol2mesh', 'file') == 2
                    is_available = true;
                    version_str = '1.9'; % 기본값
                end
            catch
                % Iso2Mesh 확인 실패
            end
    end
end
```