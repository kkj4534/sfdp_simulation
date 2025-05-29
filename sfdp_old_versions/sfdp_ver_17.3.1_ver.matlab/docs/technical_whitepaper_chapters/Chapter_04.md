# Chapter 4: 3D Thermal Analysis Engine

## 4.1 `calculate3DThermalFEATool` Function Deep-Dive

### 4.1.1 Function Signature and Parameter Validation (Lines 78-150)

**함수 정의와 입력 매개변수**

```matlab
function [temperature_field, thermal_confidence] = calculate3DThermalFEATool(cutting_speed, feed_rate, depth_of_cut, material_props, simulation_state)
```

**입력 매개변수 상세 분석**

SFDP_physics_suite.m:78에서 함수가 시작되며, 다음 매개변수들을 받습니다:

1. **cutting_speed** [m/min]: 절삭속도
   - 일반적 범위: 50-500 m/min (Ti-6Al-4V)
   - 열발생률에 직접적 영향
   - 움직이는 열원의 속도 결정

2. **feed_rate** [mm/rev]: 이송속도
   - 일반적 범위: 0.05-0.5 mm/rev
   - 절삭 두께와 열발생량 결정
   - 표면 거칠기에 직접 영향

3. **depth_of_cut** [mm]: 절삭깊이
   - 일반적 범위: 0.2-5.0 mm
   - 절삭 단면적 결정
   - 열원의 크기 영향

4. **material_props**: 재료 물성 구조체
5. **simulation_state**: 시뮬레이션 상태 정보

**매개변수 유효성 검사 구현**

```matlab
% Lines 85-120: 입력값 검증 로직
if cutting_speed <= 0 || cutting_speed > 1000
    error('절삭속도는 0-1000 m/min 범위여야 합니다: %.1f', cutting_speed);
end

if feed_rate <= 0 || feed_rate > 2.0
    error('이송속도는 0-2.0 mm/rev 범위여야 합니다: %.3f', feed_rate);
end

if depth_of_cut <= 0 || depth_of_cut > 20
    error('절삭깊이는 0-20 mm 범위여야 합니다: %.1f', depth_of_cut);
end

% 재료 물성 필수 필드 검사
required_fields = {'thermal_conductivity', 'density', 'specific_heat', 'melting_point'};
for i = 1:length(required_fields)
    if ~isfield(material_props, required_fields{i})
        error('재료 물성에 %s 필드가 없습니다', required_fields{i});
    end
end
```

**재료 물성 추출 및 온도 의존성 처리**

```matlab
% Lines 121-140: Ti-6Al-4V 물성값 추출
base_thermal_conductivity = material_props.thermal_conductivity;  % 6.7 W/m·K
base_density = material_props.density;                           % 4420 kg/m³  
base_specific_heat = material_props.specific_heat;               % 526 J/kg·K

% 온도 의존적 물성값 함수 정의
% k(T) = k₀ + k₁×T + k₂×T² (W/m·K)
thermal_conductivity_coeffs = [6.7, 0.012, -2.5e-6];  % Ti-6Al-4V 실험값
% cp(T) = cp₀ + cp₁×T + cp₂×T² (J/kg·K)  
specific_heat_coeffs = [526, 0.15, 8.5e-5];

% 온도 의존성 함수 생성
k_func = @(T) thermal_conductivity_coeffs(1) + thermal_conductivity_coeffs(2)*T + thermal_conductivity_coeffs(3)*T.^2;
cp_func = @(T) specific_heat_coeffs(1) + specific_heat_coeffs(2)*T + specific_heat_coeffs(3)*T.^2;
```

### 4.1.2 FEATool Geometry Setup (Lines 151-173)

**3D 워크피스 기하학 생성**

```matlab
% Lines 154-159: 워크피스 치수 정의
workpiece_length = 50e-3;   % 50mm - 절삭 방향 길이
workpiece_width = 20e-3;    % 20mm - 폭
workpiece_height = 10e-3;   % 10mm - 높이 (절삭깊이보다 충분히 큼)

% FEATool 좌표계 설정 (3차원)
fea.sdim = {'x', 'y', 'z'};  % x: 절삭방향, y: 폭방향, z: 깊이방향

% 3D 직육면체 형상 생성
fea.geom.objects = {gobj_block([0, workpiece_length], [0, workpiece_width], [0, workpiece_height])};
```

**적응적 메시 생성 알고리즘**

```matlab
% Lines 165-173: 메시 크기 전략
mesh_size_cutting_zone = 0.2e-3;  % 200μm - 절삭영역 (온도구배 큰 영역)
mesh_size_far_field = 1.0e-3;     % 1mm - 원거리 영역 (계산효율 고려)
mesh_growth_rate = 1.3;            % 인접 요소 크기 비율 제한

% 초기 메시 생성
fea = meshgeom(fea, 'hmax', mesh_size_far_field, 'hgrad', mesh_growth_rate);

% 절삭영역 식별 및 세분화
cutting_zone_tolerance = 2e-3;  % 2mm 이내를 절삭영역으로 정의
cutting_zone_elements = find(fea.grid.p(3, fea.grid.c(1:4, :)) > workpiece_height - cutting_zone_tolerance);

% 절삭영역 메시 2단계 세분화 (8배 조밀해짐)
fea = refine_mesh_elements(fea, cutting_zone_elements, 2);
```

**메시 품질 검사**

```matlab
% 메시 품질 지표 계산
element_quality = calculate_mesh_quality(fea.grid);
min_quality = min(element_quality);
avg_quality = mean(element_quality);

if min_quality < 0.1
    warning('메시 품질이 낮습니다 (최소: %.3f). 결과 정확도에 영향 가능', min_quality);
end

fprintf('메시 정보:\n');
fprintf('  - 총 절점 수: %d\n', size(fea.grid.p, 2));
fprintf('  - 총 요소 수: %d\n', size(fea.grid.c, 2));
fprintf('  - 평균 요소 품질: %.3f\n', avg_quality);
fprintf('  - 최소 요소 품질: %.3f\n', min_quality);
```

### 4.1.3 Physics Equation Configuration (Lines 174-199)

**열전도 방정식 계수 설정**

```matlab
% Lines 177-178: FEATool 물리방정식 계수 할당
% ∇·(k∇T) + Q = ρcp(∂T/∂t) 형태

% 열전도계수 할당 (온도 의존적)
fea.phys.ht.eqn.coef{2,end} = {material_props.thermal_conductivity};

% 열용량 계수 할당 (ρcp)
thermal_capacity = material_props.density * material_props.specific_heat;
fea.phys.ht.eqn.coef{3,end} = {thermal_capacity};

% 대류 계수 설정 (공기 중 자연대류)
convection_coefficient = 25;  % W/m²·K (일반적 공기 중 값)
fea.phys.ht.eqn.coef{4,end} = {convection_coefficient};

% 주변 온도 설정
ambient_temperature = 20;  % °C
fea.phys.ht.eqn.coef{5,end} = {ambient_temperature};
```

**시간 스텝핑 설정**

```matlab
% Lines 180-185: 시간 적분 파라미터
% 열확산 시간상수 기반 시간 스텝 계산
thermal_diffusivity = material_props.thermal_conductivity / (material_props.density * material_props.specific_heat);
element_size = mesh_size_cutting_zone;
critical_time_step = element_size^2 / (6 * thermal_diffusivity);  % 안정성 조건

% 실제 시간 스텝 (안정성 여유 포함)
time_step = 0.5 * critical_time_step;  % 안전계수 0.5
total_simulation_time = 10.0;  % 10초 시뮬레이션

% 시간 벡터 생성
time_vector = 0:time_step:total_simulation_time;
fea.sol.time = time_vector;

fprintf('시간 적분 설정:\n');
fprintf('  - 임계 시간 스텝: %.4f 초\n', critical_time_step);
fprintf('  - 사용 시간 스텝: %.4f 초\n', time_step);
fprintf('  - 총 시간 스텝 수: %d\n', length(time_vector));
```

### 4.1.4 Moving Heat Source Implementation (Lines 200-230)

**절삭 파워 계산**

```matlab
% Lines 182-184: 절삭 파워 계산
function cutting_power = calculate_cutting_power(cutting_speed, feed_rate, depth_of_cut, material_props)
    % 비절삭에너지 (Specific Cutting Energy) 계산
    % Ti-6Al-4V의 경우 약 2.5-4.0 GJ/m³
    
    specific_cutting_energy_base = 3.0e9;  % J/m³ (Ti-6Al-4V 평균값)
    
    % 절삭조건에 따른 보정 (Kienzle 공식 기반)
    % u = u₀ × h^(-mc)
    chip_thickness = feed_rate * sin(45*pi/180);  % 45도 가정 (일반적 엔드밀)
    kienzle_exponent = 0.25;  % Ti-6Al-4V 실험값
    
    specific_cutting_energy = specific_cutting_energy_base * (chip_thickness/1.0)^(-kienzle_exponent);
    
    % 절삭 체적 유량 계산
    material_removal_rate = cutting_speed/60 * feed_rate/1000 * depth_of_cut/1000;  % m³/s
    
    % 총 절삭 파워
    cutting_power = specific_cutting_energy * material_removal_rate;  % W
    
    fprintf('절삭 파워 계산:\n');
    fprintf('  - 비절삭에너지: %.1f GJ/m³\n', specific_cutting_energy/1e9);
    fprintf('  - 재료제거율: %.2e m³/s\n', material_removal_rate);
    fprintf('  - 총 절삭파워: %.1f W\n', cutting_power);
end
```

**열분배 모델**

```matlab
% Lines 186-190: 열분배 계수
% Komanduri & Hou (2000) 모델 기반
heat_partition_workpiece = 0.8;  % 80% 워크피스로
heat_partition_tool = 0.15;      % 15% 도구로  
heat_partition_chip = 0.05;      % 5% 칩으로

% 워크피스로 가는 열량
heat_generation_rate = cutting_power * heat_partition_workpiece;  % W

fprintf('열분배:\n');
fprintf('  - 워크피스: %.1f W (%.0f%%)\n', heat_generation_rate, heat_partition_workpiece*100);
fprintf('  - 도구: %.1f W (%.0f%%)\n', cutting_power * heat_partition_tool, heat_partition_tool*100);
fprintf('  - 칩: %.1f W (%.0f%%)\n', cutting_power * heat_partition_chip, heat_partition_chip*100);
```

**가우시안 열원 분포 구현**

```matlab
% Lines 191-198: 3D 가우시안 열원 모델
% Goldak et al. (1984) 이중 타원체 열원 단순화 버전

% 열원 크기 매개변수 (도구 형상 기반)
heat_source_width = 2e-3;   % 2mm - 도구 직경 기반
heat_source_length = 1e-3;  % 1mm - 접촉 길이
heat_source_depth = 0.5e-3; % 0.5mm - 침투 깊이

% 시간에 따른 절삭 위치 계산
cutting_position_x = @(t) cutting_speed/60 * t;  % m/min을 m/s로 변환

% 3D 가우시안 분포 함수 생성
create_heat_source = @(t) sprintf('%.3e * exp(-((x-%.6f)^2/(%.6f)^2 + y^2/(%.6f)^2 + (z-%.6f)^2/(%.6f)^2))', ...
    heat_generation_rate, cutting_position_x(t), heat_source_length/2, ...
    heat_source_width/2, workpiece_height, heat_source_depth/2);

% 초기 시간에서의 열원 설정
initial_heat_source = create_heat_source(0);
fea.phys.ht.eqn.coef{1,end} = {initial_heat_source};
```

**움직이는 열원 업데이트 로직**

```matlab
% 시간 스텝마다 열원 위치 업데이트
for time_idx = 2:length(time_vector)
    current_time = time_vector(time_idx);
    
    % 새로운 열원 위치 계산
    new_heat_source = create_heat_source(current_time);
    
    % FEATool에 업데이트
    fea.phys.ht.eqn.coef{1,end} = {new_heat_source};
    
    % 진행상황 출력 (매 50 스텝마다)
    if mod(time_idx, 50) == 0
        current_position = cutting_position_x(current_time);
        fprintf('시간: %.3f s, 절삭위치: %.1f mm\n', current_time, current_position*1000);
    end
end
```

### 4.1.5 Boundary Condition Application (Lines 231-280)

**경계면 식별 및 분류**

```matlab
% Lines 231-245: 경계면 자동 식별
% FEATool 메시에서 경계면 추출
boundary_faces = fea.grid.b;  % 경계면 정보

% 각 경계면을 위치에 따라 분류
top_surface_faces = [];     % z = workpiece_height (절삭면)
bottom_surface_faces = [];  % z = 0 (고정면)
side_faces = [];           % x, y 경계면

tolerance = 1e-6;  % 기하학적 허용오차

for face_idx = 1:size(boundary_faces, 2)
    face_nodes = boundary_faces(:, face_idx);
    face_coords = fea.grid.p(:, face_nodes);
    
    % 면의 중심점 계산
    face_center = mean(face_coords, 2);
    
    if abs(face_center(3) - workpiece_height) < tolerance
        top_surface_faces = [top_surface_faces, face_idx];
    elseif abs(face_center(3)) < tolerance  
        bottom_surface_faces = [bottom_surface_faces, face_idx];
    else
        side_faces = [side_faces, face_idx];
    end
end

fprintf('경계조건 설정:\n');
fprintf('  - 상부면 (절삭면): %d 개 면\n', length(top_surface_faces));
fprintf('  - 하부면 (고정면): %d 개 면\n', length(bottom_surface_faces));
fprintf('  - 측면: %d 개 면\n', length(side_faces));
```

**도구-워크피스 접촉면 조건**

```matlab
% Lines 246-260: 접촉 열전달 계수 계산
function h_contact = calculate_contact_heat_transfer(contact_pressure, surface_roughness, temperature)
    % Bahrami et al. (2004) 접촉 열저항 모델
    
    % 기본 매개변수
    thermal_conductivity_solid = 6.7;  % W/m·K (Ti-6Al-4V)
    thermal_conductivity_gas = 0.026;  % W/m·K (공기)
    
    % 표면 거칠기 기반 접촉 저항
    contact_resistance_base = surface_roughness / thermal_conductivity_solid;
    
    % 압력 의존성 (압력이 높을수록 저항 감소)
    pressure_factor = 1 / (1 + contact_pressure/1e6);  % MPa 단위
    
    % 온도 의존성 (고온에서 접촉 개선)
    temperature_factor = 1 + 0.001 * (temperature - 20);
    
    contact_resistance = contact_resistance_base * pressure_factor / temperature_factor;
    h_contact = 1 / contact_resistance;  % W/m²·K
    
    % 현실적 범위 제한
    h_contact = max(min(h_contact, 50000), 100);  % 100-50000 W/m²·K
end

% 절삭영역에서의 접촉 열전달 계수 (온도와 압력 추정값 사용)
estimated_contact_pressure = 100e6;  % 100 MPa (일반적 절삭압력)
estimated_surface_roughness = 1e-6;  % 1 μm
estimated_temperature = 600;         % 600°C (초기 추정)

contact_htc = calculate_contact_heat_transfer(estimated_contact_pressure, estimated_surface_roughness, estimated_temperature);
```

**대류 및 복사 경계조건**

```matlab
% Lines 261-275: 복합 열전달 경계조건
% 대류 + 복사 결합 모델

% 자연대류 계수 (Churchill & Chu 상관식)
function h_natural = calculate_natural_convection(surface_temp, ambient_temp, characteristic_length)
    % 무차원수 계산
    temp_diff = surface_temp - ambient_temp;
    beta = 1 / (ambient_temp + 273.15);  % 체적팽창계수 (1/K)
    
    % 공기 물성 (20°C 기준)
    kinematic_viscosity = 1.5e-5;  % m²/s
    thermal_diffusivity_air = 2.2e-5;  % m²/s
    prandtl_number = kinematic_viscosity / thermal_diffusivity_air;
    
    % Rayleigh 수
    g = 9.81;  % m/s²
    rayleigh = g * beta * temp_diff * characteristic_length^3 / (kinematic_viscosity * thermal_diffusivity_air);
    
    % Nusselt 수 (수평면 기준)
    if rayleigh < 1e7
        nusselt = 0.54 * rayleigh^0.25;
    else
        nusselt = 0.15 * rayleigh^(1/3);
    end
    
    % 대류 열전달계수
    k_air = 0.026;  % W/m·K
    h_natural = nusselt * k_air / characteristic_length;
end

% 복사 열전달 계수 (선형화)
function h_radiation = calculate_radiation_htc(surface_temp, ambient_temp, emissivity)
    stefan_boltzmann = 5.67e-8;  % W/m²·K⁴
    T_s = surface_temp + 273.15;  % K
    T_inf = ambient_temp + 273.15;  % K
    
    % 선형화된 복사 계수
    h_radiation = emissivity * stefan_boltzmann * (T_s^4 - T_inf^4) / (T_s - T_inf);
end
```

**경계조건 적용**

```matlab
% Lines 276-280: FEATool 경계조건 설정
% 상부면: 대류 + 복사 (절삭영역 제외)
for face_idx = top_surface_faces
    % 해당 면이 절삭영역인지 확인
    face_center = calculate_face_center(face_idx);
    
    if is_in_cutting_zone(face_center)
        % 절삭영역: 접촉 열전달
        fea.phys.ht.bdr.coef{face_idx} = {contact_htc, ambient_temperature};
    else
        % 자유면: 대류 + 복사
        h_total = h_natural + h_radiation;
        fea.phys.ht.bdr.coef{face_idx} = {h_total, ambient_temperature};
    end
end

% 하부면: 고정 온도 (척에 의한 냉각)
chuck_temperature = 25;  % °C
for face_idx = bottom_surface_faces
    fea.phys.ht.bdr.coef{face_idx} = {[], chuck_temperature};  % Dirichlet 조건
end

% 측면: 자연대류
h_side = 15;  % W/m²·K (수직면 자연대류)
for face_idx = side_faces
    fea.phys.ht.bdr.coef{face_idx} = {h_side, ambient_temperature};
end
```

### 4.1.6 Solution Process and Post-Processing (Lines 281-350)

**FEM 해석기 실행**

```matlab
% Lines 281-300: 솔버 설정 및 실행
% 비선형 반복 설정 (온도 의존 물성 때문)
solver_options = struct();
solver_options.nonlinear_max_iter = 20;      % 최대 비선형 반복
solver_options.nonlinear_tolerance = 1e-6;   % 수렴 허용오차
solver_options.linear_solver = 'mumps';      % 직접법 솔버 (안정적)
solver_options.time_scheme = 'backward_euler'; % 후진 오일러 (안정적)

% 초기조건 설정
initial_temperature = ambient_temperature * ones(size(fea.grid.p, 2), 1);
fea.sol.u = initial_temperature;

% 진행상황 모니터링 설정
progress_callback = @(t, u) monitor_solution_progress(t, u, workpiece_height);

try
    % FEATool 솔버 실행
    fprintf('FEM 해석 시작...\n');
    tic;
    fea = solvetime(fea, 'solver_options', solver_options, 'callback', progress_callback);
    solution_time = toc;
    
    fprintf('FEM 해석 완료: %.1f 초\n', solution_time);
    
catch ME
    fprintf('FEM 해석 실패: %s\n', ME.message);
    % 폴백 방법으로 전환
    temperature_field = fallback_thermal_solution(cutting_speed, feed_rate, depth_of_cut, material_props);
    thermal_confidence = 0.3;  % 낮은 신뢰도
    return;
end
```

**결과 후처리 및 추출**

```matlab
% Lines 301-330: 온도장 후처리
% 최종 시간에서의 온도 분포 추출
final_time_step = length(time_vector);
temperature_distribution = fea.sol.u(:, final_time_step);

% 주요 위치에서의 온도 추출
[max_temperature, max_temp_node] = max(temperature_distribution);
max_temp_coords = fea.grid.p(:, max_temp_node);

% 절삭 가장자리 온도 (공학적으로 중요)
cutting_edge_nodes = find_cutting_edge_nodes(fea.grid, workpiece_height, cutting_speed);
cutting_edge_temperatures = temperature_distribution(cutting_edge_nodes);
avg_cutting_edge_temp = mean(cutting_edge_temperatures);

% 온도 구배 계산 (열응력 평가용)
[temp_gradient_x, temp_gradient_y, temp_gradient_z] = calculate_temperature_gradient(fea, temperature_distribution);
max_gradient_magnitude = max(sqrt(temp_gradient_x.^2 + temp_gradient_y.^2 + temp_gradient_z.^2));

% 결과 구조체 생성
temperature_field = struct();
temperature_field.nodes = fea.grid.p';                    % 절점 좌표 [N×3]
temperature_field.elements = fea.grid.c(1:4,:)';          % 요소 연결성 [M×4]
temperature_field.temperature = temperature_distribution;  % 절점 온도 [N×1]
temperature_field.max_temperature = max_temperature;       % 최대 온도 [°C]
temperature_field.cutting_edge_temp = avg_cutting_edge_temp; % 절삭 가장자리 온도 [°C]
temperature_field.max_gradient = max_gradient_magnitude;    % 최대 온도 구배 [°C/m]
temperature_field.solution_time = solution_time;           % 계산 시간 [s]

% 시간 이력 데이터 (선택된 점들)
monitor_points = select_monitor_points(fea.grid, workpiece_height);
temperature_field.time_history = struct();
temperature_field.time_history.time = time_vector;
temperature_field.time_history.temperatures = fea.sol.u(monitor_points, :);
temperature_field.time_history.locations = fea.grid.p(:, monitor_points)';

fprintf('온도 해석 결과:\n');
fprintf('  - 최대 온도: %.1f °C (위치: [%.1f, %.1f, %.1f] mm)\n', ...
    max_temperature, max_temp_coords*1000);
fprintf('  - 절삭 가장자리 평균 온도: %.1f °C\n', avg_cutting_edge_temp);
fprintf('  - 최대 온도 구배: %.1e °C/m\n', max_gradient_magnitude);
```

**신뢰도 평가 알고리즘**

```matlab
% Lines 331-350: 해석 결과 신뢰도 평가
function confidence = assess_thermal_confidence(temperature_field, material_props, solver_info)
    confidence_factors = [];
    
    % 1. 물리적 타당성 검사
    physics_score = 1.0;
    
    % 융점 초과 확인
    melting_point = material_props.melting_point;  % Ti-6Al-4V: 1668°C
    if temperature_field.max_temperature > melting_point
        physics_score = physics_score * 0.3;  % 심각한 페널티
        fprintf('경고: 최대온도(%.1f°C)가 융점(%.1f°C)을 초과했습니다\n', ...
            temperature_field.max_temperature, melting_point);
    end
    
    % 비현실적 온도 구배 확인
    critical_gradient = 1e6;  % °C/m (경험적 임계값)
    if temperature_field.max_gradient > critical_gradient
        physics_score = physics_score * 0.7;
        fprintf('경고: 과도한 온도구배(%.1e°C/m)가 감지되었습니다\n', temperature_field.max_gradient);
    end
    
    confidence_factors = [confidence_factors, physics_score];
    
    % 2. 수치적 수렴성 검사
    convergence_score = 1.0;
    
    if isfield(solver_info, 'nonlinear_residual')
        final_residual = solver_info.nonlinear_residual(end);
        if final_residual > 1e-4
            convergence_score = convergence_score * 0.6;
            fprintf('경고: 비선형 수렴이 완전하지 않습니다 (잔차: %.2e)\n', final_residual);
        end
    end
    
    confidence_factors = [confidence_factors, convergence_score];
    
    % 3. 메시 의존성 평가
    mesh_score = 1.0;
    
    % 요소 품질 확인
    if isfield(temperature_field, 'element_quality')
        min_quality = min(temperature_field.element_quality);
        if min_quality < 0.2
            mesh_score = mesh_score * 0.8;
            fprintf('주의: 메시 품질이 낮습니다 (최소: %.3f)\n', min_quality);
        end
    end
    
    confidence_factors = [confidence_factors, mesh_score];
    
    % 4. 계산 시간 기반 평가 (너무 빠르면 의심)
    time_score = 1.0;
    
    if temperature_field.solution_time < 1.0  % 1초 미만
        time_score = 0.7;  % 너무 빨라서 의심스러움
        fprintf('주의: 계산시간이 매우 짧습니다 (%.2f초)\n', temperature_field.solution_time);
    end
    
    confidence_factors = [confidence_factors, time_score];
    
    % 종합 신뢰도 (가중평균)
    weights = [0.4, 0.3, 0.2, 0.1];  % 물리성 > 수렴성 > 메시 > 시간
    confidence = sum(confidence_factors .* weights);
    
    % 0-1 범위 제한
    confidence = max(0, min(1, confidence));
    
    fprintf('해석 신뢰도: %.2f\n', confidence);
end

% 신뢰도 계산 실행
thermal_confidence = assess_thermal_confidence(temperature_field, material_props, solver_info);
```

## 4.2 `calculate3DThermalAdvanced` Function Analysis

### 4.2.1 Analytical Solution Framework (Lines 351-420)

**함수 정의 및 목적**

```matlab
function [temperature_field, thermal_confidence] = calculate3DThermalAdvanced(cutting_speed, feed_rate, depth_of_cut, material_props, simulation_state)
```

이 함수는 FEATool이 없거나 빠른 계산이 필요할 때 사용하는 해석적 열해석 방법입니다. Jaeger의 움직이는 열원 이론을 3D로 확장한 솔루션을 구현합니다.

**Jaeger 이동 열원 이론 확장**

```matlab
% Lines 355-375: 3D Jaeger 솔루션 구현
function T = jaeger_3d_moving_source(x, y, z, t, heat_input, velocity, material_props)
    % 3D 확장 Jaeger 해석해
    % 기본 Jaeger (1942) 이론: T = (Q/4πk) ∫[0,t] (1/τ)exp(-(x-vτ)²+y²)/(4ατ) dτ
    % 3D 확장: z 방향 열확산 추가
    
    % 재료 상수
    k = material_props.thermal_conductivity;    % W/m·K
    rho = material_props.density;               % kg/m³
    cp = material_props.specific_heat;          % J/kg·K
    alpha = k / (rho * cp);                     % 열확산계수 m²/s
    
    % 수치적분을 위한 시간 분할
    tau_max = t;
    tau_steps = max(100, round(t * 1000));  % 최소 100 스텝, 1ms 해상도
    tau_vector = linspace(1e-6, tau_max, tau_steps);  % 0 피하기 위해 1μs부터 시작
    d_tau = tau_vector(2) - tau_vector(1);
    
    % Jaeger 적분 계산
    integrand = zeros(size(tau_vector));
    
    for i = 1:length(tau_vector)
        tau = tau_vector(i);
        
        % 3D 가우시안 커널
        r_squared = (x - velocity * tau)^2 + y^2 + z^2;
        exponential_term = exp(-r_squared / (4 * alpha * tau));
        
        % 3D 열확산 분모 (2D에서 4πατ → 3D에서 (4πατ)^(3/2))
        denominator = (4 * pi * alpha * tau)^(3/2);
        
        integrand(i) = exponential_term / denominator;
    end
    
    % 수치적분 (사다리꼴 법칙)
    integral_result = trapz(tau_vector, integrand);
    
    % 최종 온도 계산
    T = (heat_input / k) * integral_result;
end
```

**다중 점열원 중첩법**

```matlab
% Lines 376-395: 유한 크기 열원을 점열원들의 중첩으로 모델링
function temperature_field = multiple_point_sources(cutting_conditions, material_props, geometry)
    % 절삭영역을 여러 점열원으로 분할
    cutting_width = 2e-3;    % 2mm 도구 폭
    cutting_length = 1e-3;   % 1mm 접촉 길이  
    cutting_depth = 0.2e-3;  % 0.2mm 침투 깊이
    
    % 점열원 격자 생성
    n_width = 5;   % 폭 방향 분할수
    n_length = 3;  % 길이 방향 분할수  
    n_depth = 2;   % 깊이 방향 분할수
    
    total_points = n_width * n_length * n_depth;
    
    % 각 점열원의 위치 계산
    [Y_grid, X_grid, Z_grid] = ndgrid(...
        linspace(-cutting_width/2, cutting_width/2, n_width), ...
        linspace(-cutting_length/2, cutting_length/2, n_length), ...
        linspace(geometry.workpiece_height - cutting_depth, geometry.workpiece_height, n_depth));
    
    source_positions = [X_grid(:), Y_grid(:), Z_grid(:)];
    
    % 총 열입력을 점열원들에 균등 분배
    total_heat_input = calculate_cutting_power(cutting_conditions.speed, ...
        cutting_conditions.feed, cutting_conditions.depth, material_props) * 0.8;  % 80% 워크피스로
    
    heat_per_source = total_heat_input / total_points;
    
    fprintf('다중 점열원 설정:\n');
    fprintf('  - 총 점열원 수: %d\n', total_points);
    fprintf('  - 점열원당 열입력: %.1f W\n', heat_per_source);
    fprintf('  - 열원 분포: %d×%d×%d\n', n_width, n_length, n_depth);
    
    % 각 계산점에서 온도 계산
    evaluation_points = generate_evaluation_points(geometry);
    n_eval_points = size(evaluation_points, 1);
    
    temperature_distribution = zeros(n_eval_points, 1);
    calculation_time = cutting_conditions.time;  % 계산할 시간점
    
    % 각 평가점에서 모든 점열원의 기여 합산 (중첩원리)
    for eval_idx = 1:n_eval_points
        eval_point = evaluation_points(eval_idx, :);
        temp_sum = 0;
        
        for source_idx = 1:total_points
            source_pos = source_positions(source_idx, :);
            
            % 상대 위치 계산
            rel_x = eval_point(1) - source_pos(1);
            rel_y = eval_point(2) - source_pos(2); 
            rel_z = eval_point(3) - source_pos(3);
            
            % 해당 점열원의 온도 기여
            temp_contribution = jaeger_3d_moving_source(rel_x, rel_y, rel_z, ...
                calculation_time, heat_per_source, cutting_conditions.speed/60, material_props);
            
            temp_sum = temp_sum + temp_contribution;
        end
        
        temperature_distribution(eval_idx) = temp_sum + 20;  % 주변온도 20°C 추가
        
        % 진행상황 출력 (매 100점마다)
        if mod(eval_idx, 100) == 0
            fprintf('온도 계산 진행: %d/%d (%.1f%%)\n', eval_idx, n_eval_points, eval_idx/n_eval_points*100);
        end
    end
    
    return temperature_distribution;
end
```

### 4.2.2 Algorithm Flow and Key Variables

**시간 진행 루프 구조**

```matlab
% Lines 396-415: 시간에 따른 온도장 진화
function temperature_time_history = solve_transient_analytical(cutting_conditions, material_props, geometry)
    % 시간 벡터 설정
    total_time = 10.0;  % 10초 시뮬레이션
    time_step = 0.1;    % 0.1초 간격
    time_vector = 0:time_step:total_time;
    n_time_steps = length(time_vector);
    
    % 평가점 생성 (고정)
    evaluation_points = generate_evaluation_points(geometry);
    n_eval_points = size(evaluation_points, 1);
    
    % 시간 이력 저장 배열
    temperature_time_history = zeros(n_eval_points, n_time_steps);
    
    for t_idx = 1:n_time_steps
        current_time = time_vector(t_idx);
        
        % 현재 시간에서의 절삭 위치
        cutting_position_x = cutting_conditions.speed/60 * current_time;  % m
        
        % 열원이 워크피스를 벗어났는지 확인
        if cutting_position_x > geometry.workpiece_length
            % 열원이 벗어남: 냉각만 계산
            temperature_time_history(:, t_idx) = calculate_cooling_phase(...
                temperature_time_history(:, t_idx-1), time_step, material_props);
            continue;
        end
        
        % 각 평가점에서 온도 계산
        for eval_idx = 1:n_eval_points
            eval_point = evaluation_points(eval_idx, :);
            
            % 절삭 위치를 고려한 상대 좌표
            rel_x = eval_point(1) - cutting_position_x;
            rel_y = eval_point(2);
            rel_z = eval_point(3);
            
            % Jaeger 해석해로 온도 계산
            if current_time > 0
                temp = jaeger_3d_moving_source(rel_x, rel_y, rel_z, ...
                    current_time, heat_input, cutting_conditions.speed/60, material_props);
                temperature_time_history(eval_idx, t_idx) = temp + 20;  % 주변온도 추가
            else
                temperature_time_history(eval_idx, t_idx) = 20;  % 초기온도
            end
        end
        
        % 진행상황 출력
        if mod(t_idx, 10) == 0
            fprintf('시간 진행: %.1f/%.1f 초 (%.0f%%)\n', current_time, total_time, t_idx/n_time_steps*100);
        end
    end
    
    return temperature_time_history;
end
```

**핵심 변수 관리**

```matlab
% Lines 416-420: 주요 변수들의 물리적 의미와 단위
key_variables = struct();

% 시간 관련
key_variables.time_current = 0;                    % 현재 시간 [s]
key_variables.time_step = 0.1;                     % 시간 간격 [s]
key_variables.total_simulation_time = 10.0;        % 총 시뮬레이션 시간 [s]

% 공간 관련  
key_variables.cutting_position_x = 0;              % 절삭 위치 [m]
key_variables.workpiece_length = 50e-3;            % 워크피스 길이 [m]
key_variables.workpiece_width = 20e-3;             % 워크피스 폭 [m]
key_variables.workpiece_height = 10e-3;            % 워크피스 높이 [m]

% 열원 관련
key_variables.heat_input_total = 0;                % 총 열입력 [W]
key_variables.heat_input_workpiece = 0;            % 워크피스 열입력 [W]
key_variables.cutting_velocity = 0;                % 절삭속도 [m/s]

% 재료 관련
key_variables.thermal_diffusivity = 0;             % 열확산계수 [m²/s]
key_variables.thermal_conductivity = 6.7;          % 열전도계수 [W/m·K]
key_variables.density = 4420;                      % 밀도 [kg/m³]
key_variables.specific_heat = 526;                 % 비열 [J/kg·K]

fprintf('핵심 변수 초기화 완료\n');
```

### 4.2.3 Performance Optimization Strategies

**행렬 연산 최적화**

```matlab
% Lines 421-450: 벡터화를 통한 성능 향상
function temperature_vectorized = calculate_temperature_vectorized(evaluation_points, source_positions, time, material_props)
    % 모든 평가점과 열원점의 조합을 한번에 계산 (벡터화)
    
    n_eval = size(evaluation_points, 1);
    n_sources = size(source_positions, 1);
    
    % 메시그리드로 모든 조합 생성
    [eval_idx_grid, source_idx_grid] = ndgrid(1:n_eval, 1:n_sources);
    
    % 거리 계산 (벡터화)
    eval_coords = evaluation_points(eval_idx_grid(:), :);
    source_coords = source_positions(source_idx_grid(:), :);
    
    % 상대 위치 벡터
    rel_positions = eval_coords - source_coords;
    
    % 거리 제곱 계산 (한번에)
    r_squared = sum(rel_positions.^2, 2);
    
    % 재료 상수
    alpha = material_props.thermal_conductivity / ...
            (material_props.density * material_props.specific_heat);
    
    % 지수함수 계산 (벡터화)
    exponential_terms = exp(-r_squared / (4 * alpha * time));
    
    % 분모 계산
    denominator = (4 * pi * alpha * time)^(3/2);
    
    % 온도 기여 계산
    temp_contributions = exponential_terms / denominator;
    
    % 각 평가점별로 모든 열원의 기여 합산
    temp_contributions_matrix = reshape(temp_contributions, n_eval, n_sources);
    temperature_vectorized = sum(temp_contributions_matrix, 2);
    
    fprintf('벡터화 계산 완료: %d 점 × %d 열원 = %d 계산\n', n_eval, n_sources, n_eval*n_sources);
end
```

**메모리 할당 패턴**

```matlab
% Lines 451-470: 효율적 메모리 관리
function optimized_memory_allocation()
    % 사전 할당으로 메모리 재할당 방지
    
    n_eval_points = 1000;
    n_time_steps = 100;
    
    % 나쁜 예: 동적 할당
    % temperature_bad = [];
    % for i = 1:n_time_steps
    %     temperature_bad = [temperature_bad; new_temp_data];  % 매번 재할당!
    % end
    
    % 좋은 예: 사전 할당
    temperature_good = zeros(n_eval_points, n_time_steps);  % 한번에 할당
    
    for t_idx = 1:n_time_steps
        % 계산 결과를 사전 할당된 공간에 저장
        temperature_good(:, t_idx) = calculate_temperature_at_time(t_idx);
    end
    
    % 중간 결과 메모리 정리
    clear temp_intermediate_variables;
    
    % 메모리 사용량 모니터링
    memory_info = memory;
    fprintf('메모리 사용량: %.1f MB\n', memory_info.MemUsedMATLAB/1e6);
end
```

**계산 복잡도 분석**

```matlab
% Lines 471-480: 알고리즘 복잡도 평가
function complexity_analysis(n_eval_points, n_sources, n_time_steps)
    % 시간 복잡도 분석
    
    % 기본 해석해 방법: O(N_eval × N_time × N_integration_points)
    basic_complexity = n_eval_points * n_time_steps * 1000;  % 적분점 1000개
    
    % 다중 점열원 방법: O(N_eval × N_sources × N_time)  
    multi_source_complexity = n_eval_points * n_sources * n_time_steps;
    
    % 벡터화 방법: O(N_eval × N_sources × N_time) but with better constants
    vectorized_complexity = multi_source_complexity / 10;  % 약 10배 빠름
    
    fprintf('계산 복잡도 비교:\n');
    fprintf('  - 기본 해석해: %.2e 연산\n', basic_complexity);
    fprintf('  - 다중 점열원: %.2e 연산\n', multi_source_complexity);
    fprintf('  - 벡터화: %.2e 연산 (추정)\n', vectorized_complexity);
    
    % 메모리 복잡도
    memory_basic = n_eval_points * n_time_steps * 8;  % double precision
    memory_vectorized = n_eval_points * n_sources * 8 + memory_basic;
    
    fprintf('메모리 사용량:\n');
    fprintf('  - 기본: %.1f MB\n', memory_basic/1e6);
    fprintf('  - 벡터화: %.1f MB\n', memory_vectorized/1e6);
end
```

## 4.3 `applyAdvancedThermalBoundaryConditions` Implementation

### 4.3.1 Boundary Condition Types and Implementation

**Dirichlet 경계조건 (고정 온도)**

```matlab
% Lines 481-500: 고정 온도 경계조건 구현
function apply_dirichlet_bc(node_indices, temperature_value, system_matrix, rhs_vector)
    % Dirichlet 조건: T = T_prescribed
    % 시스템 행렬 수정을 통한 구현
    
    for i = 1:length(node_indices)
        node_idx = node_indices(i);
        
        % 해당 행을 단위행렬로 만들기
        system_matrix(node_idx, :) = 0;
        system_matrix(node_idx, node_idx) = 1;
        
        % 우변 벡터에 경계값 설정
        rhs_vector(node_idx) = temperature_value;
    end
    
    fprintf('Dirichlet 경계조건 적용: %d 개 절점 = %.1f°C\n', length(node_indices), temperature_value);
end
```

**Neumann 경계조건 (열유속 지정)**

```matlab
% Lines 501-525: 열유속 경계조건
function apply_neumann_bc(boundary_faces, heat_flux_value, rhs_vector, mesh_info)
    % Neumann 조건: -k(∂T/∂n) = q_specified
    % 경계면 적분을 통한 구현
    
    total_heat_added = 0;
    
    for face_idx = 1:length(boundary_faces)
        face_nodes = boundary_faces{face_idx};
        face_area = calculate_face_area(face_nodes, mesh_info);
        
        % 면적분을 절점 기여로 분배
        nodes_per_face = length(face_nodes);
        heat_per_node = heat_flux_value * face_area / nodes_per_face;
        
        % 각 절점의 우변 벡터에 기여 추가
        for node_idx = face_nodes
            rhs_vector(node_idx) = rhs_vector(node_idx) + heat_per_node;
        end
        
        total_heat_added = total_heat_added + heat_flux_value * face_area;
    end
    
    fprintf('Neumann 경계조건 적용:\n');
    fprintf('  - 지정 열유속: %.1e W/m²\n', heat_flux_value);
    fprintf('  - 총 열입력: %.1f W\n', total_heat_added);
end
```

**Robin 경계조건 (대류 열전달)**

```matlab
% Lines 526-550: 대류 경계조건 구현  
function apply_robin_bc(boundary_faces, convection_coeff, ambient_temp, system_matrix, rhs_vector, mesh_info)
    % Robin 조건: -k(∂T/∂n) = h(T - T_ambient)
    % 경계면에서의 대류 열전달
    
    for face_idx = 1:length(boundary_faces)
        face_nodes = boundary_faces{face_idx};
        face_area = calculate_face_area(face_nodes, mesh_info);
        
        % 면적분 계산 (가우스 적분 사용)
        [gauss_points, weights] = get_face_gauss_points();
        
        for gp = 1:length(weights)
            % 가우스점에서 형상함수 계산
            [N, dN] = shape_functions_2d(gauss_points(gp, :));
            
            % Jacobian 계산
            face_jacobian = calculate_face_jacobian(face_nodes, dN, mesh_info);
            det_J = det(face_jacobian);
            
            % 대류 행렬 기여 계산
            convection_matrix = weights(gp) * det_J * convection_coeff * (N' * N);
            
            % 전체 시스템 행렬에 조립
            for i = 1:length(face_nodes)
                for j = 1:length(face_nodes)
                    system_matrix(face_nodes(i), face_nodes(j)) = ...
                        system_matrix(face_nodes(i), face_nodes(j)) + convection_matrix(i, j);
                end
            end
            
            % 대류 우변 벡터 기여
            convection_rhs = weights(gp) * det_J * convection_coeff * ambient_temp * N;
            for i = 1:length(face_nodes)
                rhs_vector(face_nodes(i)) = rhs_vector(face_nodes(i)) + convection_rhs(i);
            end
        end
    end
    
    fprintf('Robin 경계조건 적용:\n');
    fprintf('  - 대류계수: %.1f W/m²·K\n', convection_coeff);
    fprintf('  - 주변온도: %.1f°C\n', ambient_temp);
    fprintf('  - 경계면 수: %d\n', length(boundary_faces));
end
```

### 4.3.2 Tool-Workpiece Interface Modeling

**접촉 저항 모델**

```matlab
% Lines 551-580: 도구-워크피스 접촉 모델링
function contact_resistance = calculate_contact_resistance(contact_pressure, surface_roughness, material_props)
    % 다층 접촉 저항 모델 (Bahrami et al. 2004)
    
    % 고체-고체 접촉 저항
    solid_contact_resistance = calculate_solid_contact_resistance(contact_pressure, surface_roughness, material_props);
    
    % 가스 갭 저항 (미세한 공기층)
    gas_gap_resistance = calculate_gas_gap_resistance(surface_roughness, material_props);
    
    % 온도 의존성 모델링
    temperature_factor = calculate_temperature_dependence(contact_pressure, material_props);
    
    % 총 접촉 저항 (직렬 연결)
    contact_resistance = (solid_contact_resistance + gas_gap_resistance) * temperature_factor;
    
    % 물리적 한계 적용
    min_resistance = 1e-6;  % 1 μm²·K/W (거의 완전 접촉)
    max_resistance = 1e-2;  % 10 mm²·K/W (거의 완전 단열)
    contact_resistance = max(min(contact_resistance, max_resistance), min_resistance);
    
    fprintf('접촉 저항 계산:\n');
    fprintf('  - 고체 접촉: %.2e m²·K/W\n', solid_contact_resistance);
    fprintf('  - 가스 갭: %.2e m²·K/W\n', gas_gap_resistance);
    fprintf('  - 온도 보정: %.3f\n', temperature_factor);
    fprintf('  - 총 저항: %.2e m²·K/W\n', contact_resistance);
end

function solid_resistance = calculate_solid_contact_resistance(pressure, roughness, material_props)
    % 실제 접촉 면적 계산 (Greenwood & Williamson 모델)
    elastic_modulus = material_props.elastic_modulus;  % Pa
    hardness = material_props.hardness;                % Pa
    
    % 무차원 압력
    dimensionless_pressure = pressure / hardness;
    
    % 실제 접촉 면적 비율
    if dimensionless_pressure < 0.01
        % 탄성 접촉
        contact_area_ratio = 1.25 * dimensionless_pressure^0.94;
    else
        % 소성 접촉
        contact_area_ratio = min(1.0, dimensionless_pressure);
    end
    
    % 접촉 저항 (수축 저항)
    thermal_conductivity = material_props.thermal_conductivity;
    contact_spot_radius = sqrt(contact_area_ratio) * roughness;
    
    solid_resistance = 1 / (2 * thermal_conductivity * contact_spot_radius);
end
```

**열분배 계수 동적 계산**

```matlab
% Lines 581-600: 동적 열분배 모델
function [partition_workpiece, partition_tool] = calculate_dynamic_heat_partition(cutting_conditions, contact_resistance, material_props)
    % 열분배는 고정값이 아니라 절삭조건에 따라 변함
    
    % 기본 분배 (Komanduri & Hou 2000)
    base_workpiece = 0.8;
    base_tool = 0.15;
    base_chip = 0.05;
    
    % 절삭속도 영향 (고속일수록 도구로 더 많이)
    speed_factor = cutting_conditions.speed / 100;  % 100 m/min 기준 정규화
    tool_fraction_increase = 0.1 * (speed_factor - 1);  % 속도 증가시 도구 분배 증가
    
    % 접촉 저항 영향 (저항이 클수록 워크피스로 덜)
    resistance_factor = contact_resistance / 1e-4;  % 기준 저항으로 정규화
    workpiece_fraction_decrease = 0.05 * log(resistance_factor);
    
    % 재료 열전도도 영향
    conductivity_ratio = material_props.thermal_conductivity / 6.7;  % Ti-6Al-4V 기준
    conductivity_effect = 0.1 * (conductivity_ratio - 1);
    
    % 최종 분배 계수 계산
    partition_workpiece = base_workpiece - workpiece_fraction_decrease + conductivity_effect;
    partition_tool = base_tool + tool_fraction_increase;
    
    % 정규화 (총합 = 1)
    total_partition = partition_workpiece + partition_tool + base_chip;
    partition_workpiece = partition_workpiece / total_partition;
    partition_tool = partition_tool / total_partition;
    
    % 물리적 한계 적용
    partition_workpiece = max(0.5, min(0.95, partition_workpiece));
    partition_tool = max(0.05, min(0.4, partition_tool));
    
    fprintf('동적 열분배:\n');
    fprintf('  - 워크피스: %.1f%% (기준: %.1f%%)\n', partition_workpiece*100, base_workpiece*100);
    fprintf('  - 도구: %.1f%% (기준: %.1f%%)\n', partition_tool*100, base_tool*100);
    fprintf('  - 속도 영향: %.3f\n', speed_factor);
    fprintf('  - 저항 영향: %.3f\n', resistance_factor);
end
```

### 4.3.3 Coolant and Environmental Effects

**냉각제 효과 모델링**

```matlab
% Lines 601-650: 냉각제 효과 통합 모델
function effective_htc = calculate_coolant_enhanced_htc(base_htc, coolant_conditions, surface_conditions)
    % 냉각제가 있을 때의 향상된 열전달 계수
    
    if ~coolant_conditions.present
        effective_htc = base_htc;
        return;
    end
    
    % 냉각제 유형별 기본 계수
    switch coolant_conditions.type
        case 'water_based'
            coolant_base_htc = 5000;    % W/m²·K
            enhancement_factor = 3.0;
        case 'oil_based'  
            coolant_base_htc = 2000;    % W/m²·K
            enhancement_factor = 2.0;
        case 'air_mist'
            coolant_base_htc = 500;     % W/m²·K
            enhancement_factor = 1.5;
        otherwise
            coolant_base_htc = base_htc;
            enhancement_factor = 1.0;
    end
    
    % 압력 효과 (고압 쿨런트)
    pressure_bar = coolant_conditions.pressure / 1e5;  % Pa to bar
    if pressure_bar > 10
        pressure_enhancement = 1 + 0.1 * log10(pressure_bar/10);
    else
        pressure_enhancement = 1.0;
    end
    
    % 유량 효과
    flow_rate_lpm = coolant_conditions.flow_rate * 60000;  % m³/s to L/min
    if flow_rate_lpm > 1
        flow_enhancement = 1 + 0.05 * log10(flow_rate_lpm);
    else
        flow_enhancement = 1.0;
    end
    
    % 온도 효과 (냉각제와 표면 온도차)
    temp_difference = surface_conditions.temperature - coolant_conditions.temperature;
    if temp_difference > 50
        temp_enhancement = 1 + 0.002 * (temp_difference - 50);
    else
        temp_enhancement = 1.0;
    end
    
    % 표면 젖음성 효과
    wetting_factor = surface_conditions.wetting_angle / 90;  % 90도 기준 정규화
    wetting_enhancement = 2 - wetting_factor;  % 젖음성 좋을수록 향상
    
    % 종합 열전달 계수
    effective_htc = coolant_base_htc * enhancement_factor * pressure_enhancement * ...
                   flow_enhancement * temp_enhancement * wetting_enhancement;
    
    % 물리적 상한 적용
    max_htc = 50000;  % W/m²·K (비등 열전달 한계)
    effective_htc = min(effective_htc, max_htc);
    
    fprintf('냉각제 향상 열전달:\n');
    fprintf('  - 기본 HTC: %.0f W/m²·K\n', coolant_base_htc);
    fprintf('  - 압력 향상: %.2f (%.0f bar)\n', pressure_enhancement, pressure_bar);
    fprintf('  - 유량 향상: %.2f (%.1f L/min)\n', flow_enhancement, flow_rate_lpm);
    fprintf('  - 온도 향상: %.2f (ΔT=%.0f°C)\n', temp_enhancement, temp_difference);
    fprintf('  - 젖음성 향상: %.2f\n', wetting_enhancement);
    fprintf('  - 최종 HTC: %.0f W/m²·K\n', effective_htc);
end
```

**환경 조건 영향**

```matlab
% Lines 651-680: 환경 조건 통합 모델
function environmental_correction = calculate_environmental_effects(ambient_conditions, machine_conditions)
    % 주변 환경이 열전달에 미치는 영향
    
    correction_factors = struct();
    
    % 1. 대기압 효과 (고도의 영향)
    standard_pressure = 101325;  % Pa (해수면)
    pressure_ratio = ambient_conditions.atmospheric_pressure / standard_pressure;
    correction_factors.pressure = pressure_ratio^0.2;  % 약한 의존성
    
    % 2. 습도 효과 (공기 중 수분의 열전도도 향상)
    relative_humidity = ambient_conditions.relative_humidity;  % 0-1
    correction_factors.humidity = 1 + 0.1 * relative_humidity;
    
    % 3. 공기 유동 효과 (기계 주변 환기)
    air_velocity = machine_conditions.air_circulation_velocity;  % m/s
    if air_velocity > 0.5
        correction_factors.air_flow = 1 + 0.3 * log10(air_velocity/0.5);
    else
        correction_factors.air_flow = 1.0;
    end
    
    % 4. 주변 온도 효과 (복사 열전달)
    ambient_temp_C = ambient_conditions.temperature;
    if ambient_temp_C > 25
        correction_factors.ambient_temp = 1 + 0.01 * (ambient_temp_C - 25);
    else
        correction_factors.ambient_temp = 1.0;
    end
    
    % 5. 기계 열적 관성 (척, 스핀들 온도)
    machine_temp_C = machine_conditions.average_temperature;
    temp_uniformity = 1 - abs(machine_temp_C - ambient_temp_C) / 100;
    correction_factors.thermal_inertia = max(0.8, temp_uniformity);
    
    % 종합 보정 계수
    environmental_correction = correction_factors.pressure * correction_factors.humidity * ...
                              correction_factors.air_flow * correction_factors.ambient_temp * ...
                              correction_factors.thermal_inertia;
    
    fprintf('환경 보정 계수:\n');
    fprintf('  - 대기압: %.3f (%.0f Pa)\n', correction_factors.pressure, ambient_conditions.atmospheric_pressure);
    fprintf('  - 습도: %.3f (%.0f%%)\n', correction_factors.humidity, relative_humidity*100);
    fprintf('  - 공기 유동: %.3f (%.1f m/s)\n', correction_factors.air_flow, air_velocity);
    fprintf('  - 주변 온도: %.3f (%.1f°C)\n', correction_factors.ambient_temp, ambient_temp_C);
    fprintf('  - 열적 관성: %.3f\n', correction_factors.thermal_inertia);
    fprintf('  - 종합 보정: %.3f\n', environmental_correction);
end
```

---

*Chapter 4는 SFDP v17.3의 3D 열해석 엔진의 핵심 구현을 다룹니다. FEATool Multiphysics를 활용한 고급 FEM 해석과 Jaeger 이론 기반 해석적 방법을 모두 제공하여 다양한 계산 환경에서 유연하게 사용할 수 있습니다. 특히 움직이는 열원 모델링, 접촉 저항 계산, 환경 조건 반영 등 실제 가공 환경을 충실히 모사하는 물리 모델을 구현했습니다.*