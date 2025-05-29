# Chapter 5: Mechanical-Tribological Engine

## 5.1 Extended Taylor Tool Life Theory and Implementation

### 5.1.1 Classical Taylor Tool Life Equation and Its Limitations

**Taylor 공구 수명 방정식의 역사적 배경**

1907년 Frederick Winslow Taylor가 제안한 공구 수명 방정식은 가공 공학의 가장 중요한 경험식 중 하나입니다:

```
V × T^n = C
```

여기서:
- **V**: 절삭속도 (m/min)
- **T**: 공구 수명 (min)  
- **n**: Taylor 지수 (재료별 상수, 일반적으로 0.1-0.5)
- **C**: Taylor 상수 (m/min)

**기존 Taylor 방정식의 한계:**

1. **단일 변수 의존성**: 절삭속도만 고려, 이송량과 절삭깊이 무시
2. **재료 특성 미반영**: 워크피스 경도나 재료 특성 고려 부족
3. **가공 조건 단순화**: 냉각, 공구 코팅, 기계 특성 등 무시
4. **온도 효과 미고려**: 절삭온도와 공구 수명의 직접적 관계 부재

### 5.1.2 SFDP Extended Taylor Equation Development

**확장된 Taylor 방정식의 이론적 배경**

SFDP v17.3에서는 다변수 Taylor 방정식을 도입하여 현실적인 가공 조건을 반영합니다:

```
V × T^n × f^a × d^b × Q^c = C_extended
```

새로운 변수들:
- **f**: 이송량 (mm/rev) - 지수 a
- **d**: 절삭깊이 (mm) - 지수 b  
- **Q**: 재료 경도 (HV) - 지수 c
- **C_extended**: 확장된 Taylor 상수

**물리적 의미와 지수 값:**

각 지수는 해당 변수가 공구 마모에 미치는 영향의 강도를 나타냅니다:

**Ti-6Al-4V 가공용 기본 계수:**
- **n = 0.25**: 속도의 영향 (확산 마모 지배)
- **a = 0.75**: 이송량의 영향 (기계적 마모 지배)
- **b = 0.15**: 깊이의 영향 (접촉 영역 증가)
- **c = 0.5**: 경도의 영향 (재료 저항성)

**온도 의존성 모델링:**

절삭온도 T_cut를 고려한 활성화 에너지 기반 모델:

```
C_effective = C_base × exp(-E_a/(R×T_cut))
```

여기서:
- **E_a**: 활성화 에너지 (45,000 J/mol for Ti-6Al-4V)
- **R**: 기체상수 (8.314 J/mol·K)
- **T_cut**: 절삭온도 (K)

### 5.1.3 Multi-Mechanism Wear Physics Integration

**다중 마모 메커니즘의 통합 모델링**

실제 공구 마모는 여러 메커니즘의 복합 작용으로 발생합니다:

**1. Archard 마모 (기계적 마모)**
```
W_archard = k_archard × (N/H) × L
```
- N: 법선력, H: 경도, L: 미끄럼 거리

**2. 확산 마모 (고온 마모)**  
```
W_diffusion = D_0 × exp(-Q/(R×T)) × t
```
- D_0: 확산 계수, Q: 활성화 에너지, t: 시간

**3. 산화 마모 (화학적 마모)**
```
W_oxidation = k_ox × P_O2^n × exp(-E_ox/(R×T))
```
- P_O2: 산소 분압, E_ox: 산화 활성화 에너지

**통합 마모 모델:**

SFDP에서는 이러한 메커니즘들을 확장된 Taylor 방정식에 통합합니다:

```matlab
% SFDP_taylor_coefficient_processor.m 에서 구현
function [total_wear_rate] = calculate_multi_mechanism_wear(cutting_conditions, material_props, temperature)
    
    % 각 메커니즘별 기여도 계산
    archard_contribution = calculate_archard_wear(cutting_conditions, material_props);
    diffusion_contribution = calculate_diffusion_wear(temperature, cutting_conditions.time);
    oxidation_contribution = calculate_oxidation_wear(temperature, cutting_conditions.environment);
    
    % 온도 의존성 가중치
    temp_weight_archard = 1.0;  % 온도 의존성 낮음
    temp_weight_diffusion = exp(-45000/(8.314*temperature));  % 높은 온도 의존성
    temp_weight_oxidation = exp(-35000/(8.314*temperature));  % 중간 온도 의존성
    
    % 총 마모율 계산
    total_wear_rate = archard_contribution * temp_weight_archard + ...
                     diffusion_contribution * temp_weight_diffusion + ...
                     oxidation_contribution * temp_weight_oxidation;
                     
    % Extended Taylor 계수에 반영
    taylor_correction_factor = 1 + 0.1 * log(total_wear_rate/reference_wear_rate);
    
end
```

### 5.1.4 Database-Driven Coefficient Adaptation

**재료별 Taylor 계수 데이터베이스**

SFDP는 다양한 재료에 대한 검증된 Taylor 계수를 데이터베이스로 관리합니다:

| 재료 | C (m/min) | n | a | b | c | 신뢰도 |
|------|-----------|---|---|---|---|--------|
| Ti-6Al-4V | 180 | 0.25 | 0.75 | 0.15 | 0.5 | ⭐⭐⭐⭐⭐ |
| Al2024-T3 | 850 | 0.35 | 0.65 | 0.10 | 0.3 | ⭐⭐⭐⭐ |
| SS316L | 120 | 0.20 | 0.80 | 0.18 | 0.6 | ⭐⭐⭐⭐ |
| Inconel718 | 85 | 0.18 | 0.85 | 0.20 | 0.7 | ⭐⭐⭐ |

**적응적 계수 선택 알고리즘:**

```matlab
function [optimized_coefficients] = select_taylor_coefficients(material_type, tool_type, cutting_conditions)
    
    % 데이터베이스에서 기본 계수 로드
    base_coeffs = load_taylor_database(material_type, tool_type);
    
    % 현재 조건에 맞는 보정
    condition_corrections = struct();
    condition_corrections.temperature = adjust_for_temperature(cutting_conditions.temperature);
    condition_corrections.cooling = adjust_for_cooling(cutting_conditions.cooling_type);
    condition_corrections.tool_coating = adjust_for_coating(tool_type.coating);
    
    % 최적화된 계수 계산
    optimized_coefficients = apply_corrections(base_coeffs, condition_corrections);
    
    % 신뢰도 평가
    confidence_score = evaluate_coefficient_confidence(material_type, cutting_conditions);
    optimized_coefficients.confidence = confidence_score;
    
end
```

## 5.2 GIBBON-Based Contact Mechanics (`calculateCoupledWearGIBBON`)

### 5.1.1 Contact Detection and Pressure Distribution (Lines 481-560)

**함수 정의 및 GIBBON 통합**

```matlab
function [contact_results, mechanical_confidence] = calculateCoupledWearGIBBON(cutting_speed, feed_rate, depth_of_cut, material_props, temperature_field, simulation_state)
```

이 함수는 GIBBON 툴박스를 활용하여 3D 접촉역학 해석을 수행합니다. 도구-워크피스 간의 복잡한 접촉 현상을 유한요소법으로 모델링합니다.

**3D 접촉 기하학 생성**

```matlab
% Lines 485-510: GIBBON 기하학 설정
function contact_geometry = setup_3D_contact_geometry(tool_geometry, workpiece_geometry)
    % 도구 형상 생성 (엔드밀 가정)
    tool_diameter = 10e-3;        % 10mm 도구 직경
    tool_helix_angle = 30;        % 30도 나선각
    tool_relief_angle = 12;       % 12도 여유각
    
    % 도구 절삭날 형상 (GIBBON 메시 생성)
    [tool_vertices, tool_faces] = create_endmill_geometry(tool_diameter, tool_helix_angle, tool_relief_angle);
    
    % 워크피스 형상 (직육면체)
    workpiece_length = 50e-3;
    workpiece_width = 20e-3;
    workpiece_height = 10e-3;
    
    [workpiece_vertices, workpiece_faces] = create_workpiece_geometry(workpiece_length, workpiece_width, workpiece_height);
    
    % 절삭 위치 계산
    cutting_position = calculate_cutting_position(cutting_speed, simulation_state.time_current);
    
    % 도구 위치 업데이트
    tool_vertices_positioned = transform_tool_position(tool_vertices, cutting_position, depth_of_cut);
    
    contact_geometry = struct();
    contact_geometry.tool_vertices = tool_vertices_positioned;
    contact_geometry.tool_faces = tool_faces;
    contact_geometry.workpiece_vertices = workpiece_vertices;
    contact_geometry.workpiece_faces = workpiece_faces;
    contact_geometry.cutting_position = cutting_position;
    
    fprintf('3D 접촉 기하학 생성:\n');
    fprintf('  - 도구 정점 수: %d\n', size(tool_vertices, 1));
    fprintf('  - 도구 면 수: %d\n', size(tool_faces, 1));
    fprintf('  - 워크피스 정점 수: %d\n', size(workpiece_vertices, 1));
    fprintf('  - 워크피스 면 수: %d\n', size(workpiece_faces, 1));
end

function [vertices, faces] = create_endmill_geometry(diameter, helix_angle, relief_angle)
    % 엔드밀 절삭날 기하학 생성
    radius = diameter / 2;
    num_flutes = 4;              % 4날 엔드밀
    flute_length = 20e-3;        % 20mm 절삭날 길이
    
    % 각 절삭날에 대한 나선 형상 생성
    vertices = [];
    faces = [];
    
    for flute = 1:num_flutes
        angular_position = (flute - 1) * 2 * pi / num_flutes;
        
        % 나선 형상 매개변수
        z_positions = linspace(0, flute_length, 50);
        
        for i = 1:length(z_positions)
            z = z_positions(i);
            theta = angular_position + z * tan(helix_angle * pi/180) / radius;
            
            % 절삭날 좌표
            x_cutting = radius * cos(theta);
            y_cutting = radius * sin(theta);
            
            % 여유각 적용
            x_relief = x_cutting - 0.1e-3 * sin(relief_angle * pi/180);
            y_relief = y_cutting - 0.1e-3 * cos(relief_angle * pi/180);
            
            vertices = [vertices; x_cutting, y_cutting, z; x_relief, y_relief, z];
        end
    end
    
    % 면 연결성 생성 (삼각형 메시)
    faces = delaunay(vertices(:,1), vertices(:,2));
    
    fprintf('엔드밀 형상 생성 완료: %d 정점, %d 면\n', size(vertices,1), size(faces,1));
end
```

**접촉 감지 알고리즘**

```matlab
% Lines 511-535: 접촉 감지 및 침투 계산
function contact_pairs = detect_contact_penetration(tool_geometry, workpiece_geometry, tolerance)
    % 가장 가까운 점 알고리즘 (Closest Point Algorithm)
    
    tool_vertices = tool_geometry.tool_vertices;
    workpiece_vertices = workpiece_geometry.workpiece_vertices;
    
    contact_pairs = [];
    penetration_tolerance = tolerance;  % 1e-6 m (1 μm)
    
    % 도구의 각 정점에 대해 워크피스와의 최단거리 계산
    for tool_idx = 1:size(tool_vertices, 1)
        tool_point = tool_vertices(tool_idx, :);
        
        % 워크피스 표면까지의 거리 계산
        distances = sqrt(sum((workpiece_vertices - tool_point).^2, 2));
        [min_distance, closest_workpiece_idx] = min(distances);
        
        % 침투 여부 확인
        if min_distance < penetration_tolerance
            penetration_depth = penetration_tolerance - min_distance;
            
            % 접촉 법선벡터 계산
            contact_normal = (tool_point - workpiece_vertices(closest_workpiece_idx, :)) / min_distance;
            
            contact_pair = struct();
            contact_pair.tool_node = tool_idx;
            contact_pair.workpiece_node = closest_workpiece_idx;
            contact_pair.penetration_depth = penetration_depth;
            contact_pair.contact_normal = contact_normal;
            contact_pair.contact_point = tool_point - contact_normal * penetration_depth/2;
            
            contact_pairs = [contact_pairs; contact_pair];
        end
    end
    
    fprintf('접촉 감지 결과:\n');
    fprintf('  - 접촉 점 쌍: %d 개\n', length(contact_pairs));
    if ~isempty(contact_pairs)
        max_penetration = max([contact_pairs.penetration_depth]);
        avg_penetration = mean([contact_pairs.penetration_depth]);
        fprintf('  - 최대 침투 깊이: %.2e m\n', max_penetration);
        fprintf('  - 평균 침투 깊이: %.2e m\n', avg_penetration);
    end
end
```

**압력 분포 계산**

```matlab
% Lines 536-560: 접촉압력 분포 계산
function pressure_distribution = solve_contact_pressure(contact_pairs, material_props, cutting_conditions)
    % Hertz 접촉 이론 + 유한요소법 결합
    
    if isempty(contact_pairs)
        pressure_distribution = [];
        return;
    end
    
    n_contact_points = length(contact_pairs);
    pressure_distribution = zeros(n_contact_points, 1);
    
    % 재료 물성
    E1 = material_props.elastic_modulus_tool;      % 도구 탄성계수 (GPa)
    E2 = material_props.elastic_modulus_workpiece; % 워크피스 탄성계수 (GPa)
    nu1 = material_props.poisson_ratio_tool;       % 도구 푸아송비
    nu2 = material_props.poisson_ratio_workpiece;  % 워크피스 푸아송비
    
    % 등가 탄성계수 계산
    E_equivalent = 1 / ((1-nu1^2)/E1 + (1-nu2^2)/E2);
    
    % 총 절삭력 추정 (Merchant 이론)
    cutting_force_total = calculate_cutting_force(cutting_conditions, material_props);
    
    for i = 1:n_contact_points
        contact_pair = contact_pairs(i);
        penetration = contact_pair.penetration_depth;
        
        % 등가 곡률반경 (국소 접촉)
        local_radius = estimate_local_curvature_radius(contact_pair);
        
        % Hertz 접촉압력 (수정된 공식)
        hertz_pressure = (3 * cutting_force_total / (2 * pi * local_radius^2)) * ...
                        (1 - (penetration/local_radius)^2)^0.5;
        
        % 온도 연화 효과 적용
        if isfield(material_props, 'temperature_field')
            local_temperature = interpolate_temperature_at_point(contact_pair.contact_point, material_props.temperature_field);
            temperature_softening = calculate_temperature_softening_factor(local_temperature, material_props);
            hertz_pressure = hertz_pressure * temperature_softening;
        end
        
        % 물리적 제한 적용
        max_pressure = material_props.yield_strength * 3;  % 소성 한계
        pressure_distribution(i) = min(hertz_pressure, max_pressure);
    end
    
    fprintf('접촉압력 분포:\n');
    fprintf('  - 평균 압력: %.1f MPa\n', mean(pressure_distribution)/1e6);
    fprintf('  - 최대 압력: %.1f MPa\n', max(pressure_distribution)/1e6);
    fprintf('  - 총 절삭력: %.1f N\n', cutting_force_total);
end
```

### 5.1.2 Deformation Analysis and Stress Calculation

**변형 해석 설정**

```matlab
% Lines 561-590: GIBBON FEA 변형 해석
function deformation_results = solve_mechanical_deformation(contact_geometry, pressure_distribution, material_props)
    % GIBBON 기반 유한요소 변형 해석
    
    try
        % GIBBON이 있는지 확인
        if ~exist('febio_spec', 'file')
            fprintf('⚠️ GIBBON 툴박스를 찾을 수 없습니다. 단순화된 해석을 사용합니다.\n');
            deformation_results = solve_simplified_deformation(contact_geometry, pressure_distribution, material_props);
            return;
        end
        
        % FEBio 입력 파일 생성
        febio_spec = febio_spec_generator();
        
        % 기하학 정의
        febio_spec.Mesh.Nodes = contact_geometry.workpiece_vertices;
        febio_spec.Mesh.Elements{1}.ATTR.type = 'tet4';
        febio_spec.Mesh.Elements{1}.elem = contact_geometry.workpiece_faces;
        
        % 재료 모델 정의 (탄소성)
        febio_spec.Material.material{1}.ATTR.type = 'elastic';
        febio_spec.Material.material{1}.E = material_props.elastic_modulus_workpiece;
        febio_spec.Material.material{1}.v = material_props.poisson_ratio_workpiece;
        
        % 경계조건 설정
        % 1. 고정 경계 (워크피스 바닥)
        bottom_nodes = find(contact_geometry.workpiece_vertices(:,3) < 1e-6);
        febio_spec.Boundary.bc{1}.ATTR.type = 'fix';
        febio_spec.Boundary.bc{1}.ATTR.node_set = bottom_nodes;
        febio_spec.Boundary.bc{1}.dofs = 'x,y,z';
        
        % 2. 압력 하중 (접촉면)
        contact_surface_faces = identify_contact_surface_faces(contact_geometry, pressure_distribution);
        febio_spec.Loads.surface_load{1}.ATTR.type = 'pressure';
        febio_spec.Loads.surface_load{1}.ATTR.surface = contact_surface_faces;
        febio_spec.Loads.surface_load{1}.pressure.ATTR.lc = 1;  % 하중 곡선
        febio_spec.Loads.surface_load{1}.pressure.VAL = mean(pressure_distribution);
        
        % FEBio 해석 실행
        [febio_results] = runMonitorFEBio(febio_spec);
        
        % 결과 후처리
        deformation_results = process_febio_results(febio_results);
        
    catch ME
        fprintf('GIBBON/FEBio 해석 실패: %s\n', ME.message);
        fprintf('단순화된 해석으로 전환합니다.\n');
        deformation_results = solve_simplified_deformation(contact_geometry, pressure_distribution, material_props);
    end
end

function deformation_results = solve_simplified_deformation(contact_geometry, pressure_distribution, material_props)
    % 단순화된 탄성 변형 해석 (GIBBON 없이)
    
    % 평균 접촉압력
    avg_pressure = mean(pressure_distribution);
    
    % 탄성 변형 (Hertz 이론 기반)
    E_workpiece = material_props.elastic_modulus_workpiece;
    nu_workpiece = material_props.poisson_ratio_workpiece;
    
    % 등가 접촉 반경
    contact_area = length(pressure_distribution) * 1e-6;  % 대략적 추정 (mm²)
    equivalent_radius = sqrt(contact_area / pi);
    
    % 최대 변형량 (중심부)
    max_deformation = (1 - nu_workpiece^2) * avg_pressure * equivalent_radius / E_workpiece;
    
    % 접촉영역에서의 변형 분포 (반타원 분포 가정)
    n_nodes = size(contact_geometry.workpiece_vertices, 1);
    deformation_field = zeros(n_nodes, 3);  % x, y, z 변형
    
    for i = 1:n_nodes
        node_position = contact_geometry.workpiece_vertices(i, :);
        
        % 접촉 중심으로부터의 거리
        distance_from_contact = norm(node_position - mean(contact_geometry.workpiece_vertices));
        
        if distance_from_contact < equivalent_radius
            % 접촉영역 내부: 반타원 분포
            relative_distance = distance_from_contact / equivalent_radius;
            local_deformation = max_deformation * sqrt(1 - relative_distance^2);
        else
            % 접촉영역 외부: 지수적 감소
            decay_factor = exp(-(distance_from_contact - equivalent_radius) / equivalent_radius);
            local_deformation = max_deformation * 0.1 * decay_factor;
        end
        
        % z 방향 변형이 주도적 (압축)
        deformation_field(i, 3) = -local_deformation;  % 음수: 압축
        
        % x, y 방향 변형 (푸아송 효과)
        lateral_strain = nu_workpiece * local_deformation / (1 - nu_workpiece);
        deformation_field(i, 1) = lateral_strain * sign(node_position(1));
        deformation_field(i, 2) = lateral_strain * sign(node_position(2));
    end
    
    deformation_results = struct();
    deformation_results.displacement_field = deformation_field;
    deformation_results.max_deformation = max_deformation;
    deformation_results.contact_area = contact_area;
    deformation_results.avg_pressure = avg_pressure;
    
    fprintf('단순화된 변형 해석 결과:\n');
    fprintf('  - 최대 변형: %.2e m\n', max_deformation);
    fprintf('  - 접촉 면적: %.2f mm²\n', contact_area * 1e6);
    fprintf('  - 평균 압력: %.1f MPa\n', avg_pressure / 1e6);
end
```

**응력 장 계산**

```matlab
% Lines 591-620: 응력 텐서 계산
function stress_results = calculate_stress_field(deformation_results, material_props)
    % 변형-응력 관계를 통한 응력 계산
    
    displacement_field = deformation_results.displacement_field;
    n_nodes = size(displacement_field, 1);
    
    % 탄성 상수
    E = material_props.elastic_modulus_workpiece;
    nu = material_props.poisson_ratio_workpiece;
    
    % 라메 상수
    lambda = E * nu / ((1 + nu) * (1 - 2*nu));
    mu = E / (2 * (1 + nu));
    
    % 응력 텐서 초기화 [σxx, σyy, σzz, σxy, σyz, σzx]
    stress_tensor = zeros(n_nodes, 6);
    
    for i = 1:n_nodes
        % 변형률 계산 (간단한 차분법)
        if i > 1 && i < n_nodes
            % 중앙차분
            du_dx = (displacement_field(i+1, 1) - displacement_field(i-1, 1)) / 2e-3;  % 대략적 격자 간격
            dv_dy = (displacement_field(i+1, 2) - displacement_field(i-1, 2)) / 2e-3;
            dw_dz = (displacement_field(i+1, 3) - displacement_field(i-1, 3)) / 2e-3;
            
            % 전단 변형률 (단순화)
            gamma_xy = 0;  % 생략
            gamma_yz = 0;
            gamma_zx = 0;
        else
            % 경계 노드는 0으로 가정
            du_dx = 0; dv_dy = 0; dw_dz = 0;
            gamma_xy = 0; gamma_yz = 0; gamma_zx = 0;
        end
        
        % 체적 변형률
        volumetric_strain = du_dx + dv_dy + dw_dz;
        
        % 응력 계산 (일반화된 훅의 법칙)
        stress_tensor(i, 1) = lambda * volumetric_strain + 2 * mu * du_dx;  % σxx
        stress_tensor(i, 2) = lambda * volumetric_strain + 2 * mu * dv_dy;  % σyy  
        stress_tensor(i, 3) = lambda * volumetric_strain + 2 * mu * dw_dz;  % σzz
        stress_tensor(i, 4) = mu * gamma_xy;                                % σxy
        stress_tensor(i, 5) = mu * gamma_yz;                                % σyz
        stress_tensor(i, 6) = mu * gamma_zx;                                % σzx
    end
    
    % 주응력 계산
    principal_stresses = zeros(n_nodes, 3);
    von_mises_stress = zeros(n_nodes, 1);
    
    for i = 1:n_nodes
        % 응력 텐서 행렬
        stress_matrix = [stress_tensor(i,1), stress_tensor(i,4), stress_tensor(i,6);
                        stress_tensor(i,4), stress_tensor(i,2), stress_tensor(i,5);
                        stress_tensor(i,6), stress_tensor(i,5), stress_tensor(i,3)];
        
        % 주응력 (고유값)
        principal_stresses(i, :) = sort(eig(stress_matrix), 'descend');
        
        % von Mises 응력
        s1 = principal_stresses(i, 1);
        s2 = principal_stresses(i, 2);
        s3 = principal_stresses(i, 3);
        von_mises_stress(i) = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2));
    end
    
    stress_results = struct();
    stress_results.stress_tensor = stress_tensor;
    stress_results.principal_stresses = principal_stresses;
    stress_results.von_mises_stress = von_mises_stress;
    stress_results.max_von_mises = max(von_mises_stress);
    stress_results.max_principal_stress = max(principal_stresses(:));
    
    fprintf('응력 해석 결과:\n');
    fprintf('  - 최대 von Mises 응력: %.1f MPa\n', stress_results.max_von_mises / 1e6);
    fprintf('  - 최대 주응력: %.1f MPa\n', stress_results.max_principal_stress / 1e6);
    fprintf('  - 항복강도 대비: %.1f%%\n', stress_results.max_von_mises / material_props.yield_strength * 100);
end
```

### 5.1.3 Integration with Thermal Analysis

**열-기계 결합 해석**

```matlab
% Lines 621-650: 열-기계 커플링
function coupled_results = integrate_thermal_mechanical(thermal_results, mechanical_results, material_props)
    % 온도장과 기계적 응력의 상호작용
    
    if isempty(thermal_results) || isempty(mechanical_results)
        fprintf('⚠️ 열해석 또는 기계해석 결과가 없습니다. 커플링을 건너뜁니다.\n');
        coupled_results = mechanical_results;
        return;
    end
    
    % 온도 의존적 재료 물성 업데이트
    temperature_field = thermal_results.temperature;
    n_nodes = length(temperature_field);
    
    % 온도에 따른 탄성계수 변화 (Ti-6Al-4V)
    E_ref = material_props.elastic_modulus_workpiece;  % 기준 온도에서의 값
    T_ref = 20;  % 기준 온도 [°C]
    
    % 온도 의존성 계수 (실험적 상관식)
    alpha_E = -2.5e-4;  % /°C (Ti-6Al-4V의 탄성계수 온도 계수)
    E_temperature_dependent = zeros(n_nodes, 1);
    
    for i = 1:n_nodes
        T_local = temperature_field(i);
        E_temperature_dependent(i) = E_ref * (1 + alpha_E * (T_local - T_ref));
        
        % 물리적 제한 (완전히 연화되지 않도록)
        E_temperature_dependent(i) = max(E_temperature_dependent(i), 0.1 * E_ref);
    end
    
    % 열응력 계산 (열팽창에 의한)
    alpha_thermal = material_props.thermal_expansion_coefficient;  % /°C
    
    thermal_strains = zeros(n_nodes, 3);
    thermal_stresses = zeros(n_nodes, 6);
    
    for i = 1:n_nodes
        T_local = temperature_field(i);
        thermal_strain = alpha_thermal * (T_local - T_ref);
        
        % 등방성 열팽창 (x, y, z 방향 동일)
        thermal_strains(i, :) = [thermal_strain, thermal_strain, thermal_strain];
        
        % 제약이 있는 경우의 열응력
        E_local = E_temperature_dependent(i);
        nu = material_props.poisson_ratio_workpiece;
        
        if T_local > T_ref + 50  % 50°C 이상에서만 열응력 고려
            thermal_stress_magnitude = E_local * alpha_thermal * (T_local - T_ref) / (1 - 2*nu);
            thermal_stresses(i, 1:3) = thermal_stress_magnitude;  % σxx, σyy, σzz
        end
    end
    
    % 기계적 응력과 열응력 중첩
    total_stress_tensor = mechanical_results.stress_tensor + thermal_stresses;
    
    % 온도 연화 효과 적용 (항복강도 감소)
    yield_strength_temperature = zeros(n_nodes, 1);
    for i = 1:n_nodes
        T_local = temperature_field(i);
        
        % Ti-6Al-4V 항복강도 온도 의존성
        if T_local < 400
            yield_factor = 1.0;  % 상온 강도 유지
        elseif T_local < 600
            yield_factor = 1.0 - 0.001 * (T_local - 400);  % 선형 감소
        else
            yield_factor = 0.8 - 0.002 * (T_local - 600);  % 급격한 감소
        end
        
        yield_factor = max(yield_factor, 0.1);  % 최소 10% 강도 유지
        yield_strength_temperature(i) = material_props.yield_strength * yield_factor;
    end
    
    % 소성 변형 발생 여부 확인
    von_mises_total = calculate_von_mises_from_tensor(total_stress_tensor);
    plastic_nodes = find(von_mises_total > yield_strength_temperature);
    
    coupled_results = struct();
    coupled_results.mechanical_stress = mechanical_results.stress_tensor;
    coupled_results.thermal_stress = thermal_stresses;
    coupled_results.total_stress = total_stress_tensor;
    coupled_results.von_mises_total = von_mises_total;
    coupled_results.yield_strength_temp = yield_strength_temperature;
    coupled_results.plastic_nodes = plastic_nodes;
    coupled_results.elastic_modulus_temp = E_temperature_dependent;
    
    fprintf('열-기계 커플링 결과:\n');
    fprintf('  - 최대 열응력: %.1f MPa\n', max(thermal_stresses(:)) / 1e6);
    fprintf('  - 최대 총 von Mises: %.1f MPa\n', max(von_mises_total) / 1e6);
    fprintf('  - 소성 변형 노드: %d/%d (%.1f%%)\n', length(plastic_nodes), n_nodes, length(plastic_nodes)/n_nodes*100);
    fprintf('  - 최대 온도: %.1f°C\n', max(temperature_field));
    fprintf('  - 최소 탄성계수: %.1f GPa\n', min(E_temperature_dependent) / 1e9);
end
```

## 5.2 Multi-Mechanism Wear Physics (`calculateAdvancedWearPhysics`)

### 5.2.1 Archard Wear Implementation (Lines 561-620)

**통합 마모 모델 프레임워크**

```matlab
function [wear_results, wear_confidence] = calculateAdvancedWearPhysics(cutting_speed, feed_rate, depth_of_cut, material_props, temperature_field, contact_results, simulation_state)
```

**Archard 마모 법칙 구현**

```matlab
% Lines 565-590: Archard 마모 메커니즘
function archard_wear = calculate_archard_wear(contact_results, material_props, cutting_conditions, temperature_field)
    % V = k × (F × s) / H
    % 여기서 V: 마모 체적, k: 마모 계수, F: 법선력, s: 슬라이딩 거리, H: 경도
    
    if isempty(contact_results) || isempty(contact_results.pressure_distribution)
        fprintf('⚠️ 접촉 결과가 없습니다. Archard 마모 계산을 건너뜁니다.\n');
        archard_wear = struct('volume', 0, 'depth', 0, 'rate', 0);
        return;
    end
    
    % 기본 Archard 상수 (Ti-6Al-4V vs 카바이드)
    k_base = 1.5e-7;  % 기본 마모 계수 (무차원) - 실험값
    
    % 접촉력 계산
    pressure_distribution = contact_results.pressure_distribution;
    contact_area_per_point = 1e-6;  % 1 mm² per contact point (추정)
    normal_forces = pressure_distribution * contact_area_per_point;  % N
    total_normal_force = sum(normal_forces);
    
    % 슬라이딩 거리 계산
    cutting_speed_ms = cutting_conditions.speed / 60;  % m/min to m/s
    cutting_time = simulation_state.time_current;
    sliding_distance = cutting_speed_ms * cutting_time;  % m
    
    % 경도 계산 (온도 의존적)
    base_hardness = material_props.hardness;  % Pa
    
    % 온도 효과 반영 (있는 경우)
    if ~isempty(temperature_field)
        avg_contact_temperature = calculate_average_contact_temperature(contact_results, temperature_field);
        hardness_temperature_factor = calculate_hardness_temperature_effect(avg_contact_temperature, material_props);
        effective_hardness = base_hardness * hardness_temperature_factor;
    else
        effective_hardness = base_hardness;
        avg_contact_temperature = 20;  % 기본값
    end
    
    % 압력 의존성 (Archard 상수의 압력 효과)
    avg_pressure = mean(pressure_distribution);
    pressure_effect = 1 + 0.1 * log10(avg_pressure / 1e6);  % MPa 단위 기준
    k_effective = k_base * pressure_effect;
    
    % 속도 의존성 (마찰열 증가로 인한 마모 가속)
    velocity_effect = 1 + 0.05 * log10(cutting_speed_ms / 1.0);  % 1 m/s 기준
    k_effective = k_effective * velocity_effect;
    
    % Archard 마모 체적 계산
    wear_volume = k_effective * (total_normal_force * sliding_distance) / effective_hardness;  % m³
    
    % 마모 깊이 계산 (접촉 면적으로 나눔)
    total_contact_area = length(pressure_distribution) * contact_area_per_point;
    wear_depth = wear_volume / total_contact_area;  % m
    
    % 마모율 계산 (단위시간당)
    if cutting_time > 0
        wear_rate = wear_depth / cutting_time;  % m/s
    else
        wear_rate = 0;
    end
    
    archard_wear = struct();
    archard_wear.volume = wear_volume;
    archard_wear.depth = wear_depth;
    archard_wear.rate = wear_rate;
    archard_wear.effective_hardness = effective_hardness;
    archard_wear.wear_coefficient = k_effective;
    archard_wear.sliding_distance = sliding_distance;
    archard_wear.normal_force = total_normal_force;
    
    fprintf('Archard 마모 계산:\n');
    fprintf('  - 마모 체적: %.2e m³\n', wear_volume);
    fprintf('  - 마모 깊이: %.2e m (%.1f μm)\n', wear_depth, wear_depth * 1e6);
    fprintf('  - 마모율: %.2e m/s\n', wear_rate);
    fprintf('  - 유효 마모계수: %.2e\n', k_effective);
    fprintf('  - 접촉 온도: %.1f°C\n', avg_contact_temperature);
    fprintf('  - 유효 경도: %.1f GPa\n', effective_hardness / 1e9);
end

function hardness_factor = calculate_hardness_temperature_effect(temperature, material_props)
    % Ti-6Al-4V 경도의 온도 의존성
    T_ref = 20;  % 기준 온도 [°C]
    
    if temperature < 200
        hardness_factor = 1.0;  % 저온에서는 변화 없음
    elseif temperature < 500
        hardness_factor = 1.0 - 0.0005 * (temperature - 200);  % 선형 감소
    elseif temperature < 800
        hardness_factor = 0.85 - 0.001 * (temperature - 500);  % 가속 감소
    else
        hardness_factor = 0.55 - 0.0005 * (temperature - 800);  % 완만한 감소
    end
    
    hardness_factor = max(hardness_factor, 0.2);  % 최소 20% 경도 유지
end
```

### 5.2.2 Diffusion Wear Modeling (Lines 621-680)

**확산 마모 메커니즘**

```matlab
% Lines 625-660: 확산 마모 구현
function diffusion_wear = calculate_diffusion_wear(material_props, temperature_field, contact_results, cutting_conditions)
    % 고온에서 원자 확산에 의한 마모
    % 확산 플럭스: J = -D(∂C/∂x)
    % D = D₀ × exp(-Q/(RT))
    
    if isempty(temperature_field)
        fprintf('⚠️ 온도장이 없습니다. 확산 마모를 계산할 수 없습니다.\n');
        diffusion_wear = struct('volume', 0, 'depth', 0, 'rate', 0);
        return;
    end
    
    % 확산 활성화 에너지 (Ti-6Al-4V에서 C 확산)
    Q_activation = 150000;  % J/mol (Ti에서 탄소 확산)
    D0_pre_exponential = 2.3e-4;  % m²/s (전지수 인자)
    R_gas_constant = 8.314;  % J/mol·K
    
    % 접촉 영역 평균 온도
    if ~isempty(contact_results)
        avg_contact_temperature = calculate_average_contact_temperature(contact_results, temperature_field);
    else
        avg_contact_temperature = max(temperature_field.temperature);
    end
    
    T_kelvin = avg_contact_temperature + 273.15;
    
    % 확산계수 계산
    diffusion_coefficient = D0_pre_exponential * exp(-Q_activation / (R_gas_constant * T_kelvin));
    
    % 농도 구배 추정 (도구-워크피스 계면)
    carbon_concentration_tool = 0.8;     % 카바이드 도구 (80% C)
    carbon_concentration_workpiece = 0.08; % Ti-6Al-4V (0.08% C)
    interface_thickness = 1e-9;         % 1 nm (계면 두께)
    
    concentration_gradient = (carbon_concentration_tool - carbon_concentration_workpiece) / interface_thickness;  % kg/m⁴
    
    % 확산 플럭스
    diffusion_flux = diffusion_coefficient * concentration_gradient;  % kg/m²·s
    
    % 접촉 시간 및 면적
    cutting_time = simulation_state.time_current;
    if ~isempty(contact_results) && isfield(contact_results, 'contact_area')
        contact_area = contact_results.contact_area;
    else
        contact_area = 1e-6;  % 1 mm² 가정
    end
    
    % 확산으로 인한 질량 손실
    diffused_mass = diffusion_flux * contact_area * cutting_time;  % kg
    
    % 마모 체적 계산 (밀도로 나눔)
    wear_volume = diffused_mass / material_props.density;  % m³
    wear_depth = wear_volume / contact_area;  % m
    
    % 마모율
    if cutting_time > 0
        wear_rate = wear_depth / cutting_time;  % m/s
    else
        wear_rate = 0;
    end
    
    diffusion_wear = struct();
    diffusion_wear.volume = wear_volume;
    diffusion_wear.depth = wear_depth;
    diffusion_wear.rate = wear_rate;
    diffusion_wear.diffusion_coefficient = diffusion_coefficient;
    diffusion_wear.contact_temperature = avg_contact_temperature;
    diffusion_wear.diffusion_flux = diffusion_flux;
    diffusion_wear.diffused_mass = diffused_mass;
    
    fprintf('확산 마모 계산:\n');
    fprintf('  - 접촉 온도: %.1f°C\n', avg_contact_temperature);
    fprintf('  - 확산계수: %.2e m²/s\n', diffusion_coefficient);
    fprintf('  - 확산 플럭스: %.2e kg/m²·s\n', diffusion_flux);
    fprintf('  - 확산 마모 깊이: %.2e m (%.1f nm)\n', wear_depth, wear_depth * 1e9);
    fprintf('  - 확산 마모율: %.2e m/s\n', wear_rate);
end
```

### 5.2.3 Oxidation and Thermal Softening (Lines 681-750)

**산화 마모 모델**

```matlab
% Lines 685-725: 산화 마모 메커니즘
function oxidation_wear = calculate_oxidation_wear(material_props, temperature_field, contact_results, cutting_conditions)
    % 고온 산화에 의한 마모
    % 산화막 성장: x² = kt (포물선 법칙)
    
    if isempty(temperature_field)
        fprintf('⚠️ 온도장이 없습니다. 산화 마모를 계산할 수 없습니다.\n');
        oxidation_wear = struct('volume', 0, 'depth', 0, 'rate', 0);
        return;
    end
    
    % 산화 시작 온도 확인
    oxidation_threshold_temp = 500;  % °C (Ti-6Al-4V 산화 시작 온도)
    
    if ~isempty(contact_results)
        avg_contact_temperature = calculate_average_contact_temperature(contact_results, temperature_field);
    else
        avg_contact_temperature = max(temperature_field.temperature);
    end
    
    if avg_contact_temperature < oxidation_threshold_temp
        fprintf('접촉 온도(%.1f°C)가 산화 임계온도(%.1f°C)보다 낮습니다.\n', avg_contact_temperature, oxidation_threshold_temp);
        oxidation_wear = struct('volume', 0, 'depth', 0, 'rate', 0);
        return;
    end
    
    % 산화 속도 상수 (Ti → TiO₂)
    % k = k₀ × exp(-Q_ox/(RT))
    k0_oxidation = 1.2e-8;     % m²/s (전지수 인자)
    Q_oxidation = 180000;      % J/mol (산화 활성화 에너지)
    R_gas_constant = 8.314;    % J/mol·K
    
    T_kelvin = avg_contact_temperature + 273.15;
    oxidation_rate_constant = k0_oxidation * exp(-Q_oxidation / (R_gas_constant * T_kelvin));
    
    % 산화막 두께 계산 (포물선 법칙)
    oxidation_time = simulation_state.time_current;
    oxide_thickness = sqrt(oxidation_rate_constant * oxidation_time);  % m
    
    % 산화막 박리에 의한 마모
    % 기계적 응력이 산화막 부착강도를 초과하면 박리
    oxide_adhesion_strength = 10e6;  % Pa (TiO₂ 부착강도 추정)
    
    if ~isempty(contact_results) && isfield(contact_results, 'pressure_distribution')
        max_contact_stress = max(contact_results.pressure_distribution);
        if max_contact_stress > oxide_adhesion_strength
            spallation_factor = min(max_contact_stress / oxide_adhesion_strength, 3.0);  % 최대 3배
        else
            spallation_factor = 0;  % 박리 없음
        end
    else
        spallation_factor = 1.0;  % 기본값
    end
    
    % 실제 마모되는 산화막 두께
    effective_wear_depth = oxide_thickness * spallation_factor;
    
    % 접촉 면적
    if ~isempty(contact_results) && isfield(contact_results, 'contact_area')
        contact_area = contact_results.contact_area;
    else
        contact_area = 1e-6;  % 1 mm² 가정
    end
    
    % 마모 체적 (산화막 밀도 고려)
    oxide_density = 4230;  % kg/m³ (TiO₂ 밀도)
    wear_volume = effective_wear_depth * contact_area;  % m³
    
    % 마모율
    if oxidation_time > 0
        wear_rate = effective_wear_depth / oxidation_time;  % m/s
    else
        wear_rate = 0;
    end
    
    oxidation_wear = struct();
    oxidation_wear.volume = wear_volume;
    oxidation_wear.depth = effective_wear_depth;
    oxidation_wear.rate = wear_rate;
    oxidation_wear.oxide_thickness = oxide_thickness;
    oxidation_wear.oxidation_rate_constant = oxidation_rate_constant;
    oxidation_wear.contact_temperature = avg_contact_temperature;
    oxidation_wear.spallation_factor = spallation_factor;
    
    fprintf('산화 마모 계산:\n');
    fprintf('  - 접촉 온도: %.1f°C\n', avg_contact_temperature);
    fprintf('  - 산화 속도상수: %.2e m²/s\n', oxidation_rate_constant);
    fprintf('  - 산화막 두께: %.2e m (%.1f μm)\n', oxide_thickness, oxide_thickness * 1e6);
    fprintf('  - 박리 인자: %.2f\n', spallation_factor);
    fprintf('  - 산화 마모 깊이: %.2e m (%.1f μm)\n', effective_wear_depth, effective_wear_depth * 1e6);
    fprintf('  - 산화 마모율: %.2e m/s\n', wear_rate);
end
```

**열연화 효과 통합**

```matlab
% Lines 726-750: 열연화 효과 모델링
function thermal_softening = calculate_thermal_softening_effects(material_props, temperature_field, contact_results)
    % 고온에서의 재료 연화가 마모에 미치는 영향
    
    if isempty(temperature_field)
        thermal_softening = struct('factor', 1.0, 'temperature', 20);
        return;
    end
    
    if ~isempty(contact_results)
        avg_contact_temperature = calculate_average_contact_temperature(contact_results, temperature_field);
    else
        avg_contact_temperature = max(temperature_field.temperature);
    end
    
    % Ti-6Al-4V 열연화 모델
    T_ref = 20;              % 기준 온도 [°C]
    T_alpha_beta = 995;      % α+β → β 상변태 온도 [°C]
    T_melt = 1668;           % 융점 [°C]
    
    if avg_contact_temperature < 400
        % 저온: 연화 효과 미미
        softening_factor = 1.0;
    elseif avg_contact_temperature < T_alpha_beta
        % 중온: 선형 연화
        softening_factor = 1.0 + 0.5 * (avg_contact_temperature - 400) / (T_alpha_beta - 400);
    elseif avg_contact_temperature < T_melt
        % 고온: 급격한 연화
        softening_factor = 1.5 + 2.0 * (avg_contact_temperature - T_alpha_beta) / (T_melt - T_alpha_beta);
    else
        % 융점 이상: 최대 연화
        softening_factor = 3.5;
    end
    
    % 상변태 효과 (α+β → β)
    if avg_contact_temperature > T_alpha_beta
        phase_transformation_factor = 1.2;  % β상에서 추가 연화
        softening_factor = softening_factor * phase_transformation_factor;
    end
    
    % 변형률 속도 효과 (고온에서 변형률 속도 민감성 증가)
    if ~isempty(contact_results) && isfield(contact_results, 'strain_rate')
        strain_rate = contact_results.strain_rate;
    else
        % 절삭속도로부터 추정
        cutting_speed_ms = simulation_state.cutting_speed / 60;
        strain_rate = cutting_speed_ms / 1e-3;  % 1/s (대략적 추정)
    end
    
    % 고온에서의 변형률 속도 연화
    if avg_contact_temperature > 600
        strain_rate_factor = 1 + 0.1 * log10(strain_rate / 1000);  % 1000/s 기준
        strain_rate_factor = max(strain_rate_factor, 1.0);
        softening_factor = softening_factor * strain_rate_factor;
    end
    
    thermal_softening = struct();
    thermal_softening.factor = softening_factor;
    thermal_softening.temperature = avg_contact_temperature;
    thermal_softening.phase_transformation = (avg_contact_temperature > T_alpha_beta);
    thermal_softening.strain_rate = strain_rate;
    
    fprintf('열연화 효과:\n');
    fprintf('  - 접촉 온도: %.1f°C\n', avg_contact_temperature);
    fprintf('  - 연화 인자: %.2f\n', softening_factor);
    fprintf('  - 상변태 발생: %s\n', thermal_softening.phase_transformation ? '예' : '아니오');
    fprintf('  - 변형률 속도: %.1e /s\n', strain_rate);
end
```

---

*Chapter 5는 SFDP v17.3의 기계-트리볼로지 엔진의 핵심 구현을 다룹니다. GIBBON 툴박스를 활용한 3D 접촉역학 해석과 다중 메커니즘 마모 물리 모델을 통해 실제 가공 환경에서의 복잡한 기계적 상호작용을 정확하게 모사합니다. Archard 마모, 확산 마모, 산화 마모 등 다양한 마모 메커니즘과 열-기계 결합 효과를 종합적으로 고려하여 도구 수명 예측의 정확도를 크게 향상시켰습니다.*