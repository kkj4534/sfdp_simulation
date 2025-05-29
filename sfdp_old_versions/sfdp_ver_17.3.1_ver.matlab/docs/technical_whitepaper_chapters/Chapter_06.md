# Chapter 6: Surface Physics and Multi-Scale Modeling

## 6.1 Multi-Scale Roughness Analysis (`calculateMultiScaleRoughnessAdvanced`)

### 6.1.1 Fractal Surface Characterization (Lines 821-900)

```matlab
function [roughness_results, roughness_confidence] = calculateMultiScaleRoughnessAdvanced(cutting_speed, feed_rate, depth_of_cut, material_props, temperature_field, wear_results, simulation_state)
```

**프랙탈 차원 계산 구현**

```matlab
% Lines 825-850: Box-counting 방법으로 프랙탈 차원 계산
function fractal_dimension = calculate_surface_fractal_dimension(surface_profile)
    % 표면 프로파일로부터 프랙탈 차원 계산
    
    if isempty(surface_profile) || length(surface_profile) < 100
        fprintf('⚠️ 표면 프로파일이 부족합니다. 기본 프랙탈 차원을 사용합니다.\n');
        fractal_dimension = 2.2;  % Ti-6Al-4V 가공면의 일반적 값
        return;
    end
    
    % Box-counting 스케일 설정
    profile_length = length(surface_profile);
    max_scale = floor(profile_length / 4);
    min_scale = 2;
    
    % 로그 스케일로 박스 크기 설정
    num_scales = 20;
    scales = round(logspace(log10(min_scale), log10(max_scale), num_scales));
    scales = unique(scales);  % 중복 제거
    
    box_counts = zeros(size(scales));
    
    for i = 1:length(scales)
        scale = scales(i);
        
        % 각 스케일에서 박스 개수 계산
        num_boxes = 0;
        
        for start = 1:scale:profile_length-scale+1
            box_end = min(start + scale - 1, profile_length);
            box_data = surface_profile(start:box_end);
            
            % 박스 내에서 표면이 존재하는지 확인
            if range(box_data) > 1e-9  % 1 nm 이상의 변화가 있으면
                num_boxes = num_boxes + 1;
            end
        end
        
        box_counts(i) = num_boxes;
    end
    
    % 프랙탈 차원 계산 (로그-로그 회귀)
    valid_indices = box_counts > 0;
    if sum(valid_indices) < 3
        fractal_dimension = 2.2;  % 기본값
        return;
    end
    
    log_scales = log10(scales(valid_indices));
    log_counts = log10(box_counts(valid_indices));
    
    % 선형 회귀
    poly_coeffs = polyfit(log_scales, log_counts, 1);
    fractal_dimension = -poly_coeffs(1);  % 기울기의 음수
    
    % 물리적 타당성 검사
    if fractal_dimension < 2.0 || fractal_dimension > 3.0
        fprintf('⚠️ 비정상적인 프랙탈 차원 (%.3f). 기본값으로 설정합니다.\n', fractal_dimension);
        fractal_dimension = 2.2;
    end
    
    fprintf('표면 프랙탈 차원: %.3f\n', fractal_dimension);
end
```

**다중 스케일 거칠기 분해**

```matlab
% Lines 851-885: 스케일별 거칠기 분해
function multi_scale_roughness = decompose_surface_roughness(surface_profile, cutting_conditions)
    % 웨이블릿 변환을 이용한 다중 스케일 분해
    
    if length(surface_profile) < 64
        % 데이터가 부족한 경우 경험적 모델 사용
        multi_scale_roughness = empirical_roughness_model(cutting_conditions);
        return;
    end
    
    % 웨이블릿 분해 (Daubechies 4 웨이블릿 사용)
    max_level = floor(log2(length(surface_profile))) - 3;  % 최대 분해 레벨
    max_level = min(max_level, 6);  % 6레벨로 제한
    
    try
        [C, L] = wavedec(surface_profile, max_level, 'db4');
        
        % 각 스케일별 거칠기 계산
        roughness_by_scale = zeros(max_level + 1, 1);
        
        % 근사 계수 (가장 낮은 주파수)
        approx_coeffs = appcoef(C, L, 'db4', max_level);
        roughness_by_scale(1) = std(approx_coeffs);  % 거시적 거칠기
        
        % 상세 계수들 (높은 주파수부터 낮은 주파수)
        for level = 1:max_level
            detail_coeffs = detcoef(C, L, level);
            roughness_by_scale(level + 1) = std(detail_coeffs);
        end
        
    catch ME
        fprintf('웨이블릿 분해 실패: %s\n', ME.message);
        multi_scale_roughness = empirical_roughness_model(cutting_conditions);
        return;
    end
    
    % 스케일별 물리적 의미 부여
    scale_names = {'macro', 'meso6', 'meso5', 'meso4', 'micro3', 'micro2', 'micro1'};
    scale_lengths = [1e-2, 1e-3, 2e-4, 4e-5, 8e-6, 1.6e-6, 3.2e-7];  % m 단위
    
    multi_scale_roughness = struct();
    for i = 1:min(length(scale_names), length(roughness_by_scale))
        scale_name = scale_names{i};
        multi_scale_roughness.(scale_name) = roughness_by_scale(i);
        multi_scale_roughness.([scale_name '_length']) = scale_lengths(i);
    end
    
    % 총 거칠기 (RMS 합성)
    multi_scale_roughness.total_rms = sqrt(sum(roughness_by_scale.^2));
    
    fprintf('다중 스케일 거칠기 분해 완료:\n');
    for i = 1:min(length(scale_names), length(roughness_by_scale))
        fprintf('  - %s (%.1e m): %.3f μm\n', scale_names{i}, scale_lengths(i), roughness_by_scale(i)*1e6);
    end
end

function empirical_roughness = empirical_roughness_model(cutting_conditions)
    % 표면 프로파일이 없을 때 사용하는 경험적 모델
    
    feed_rate = cutting_conditions.feed;  % mm/rev
    cutting_speed = cutting_conditions.speed;  % m/min
    tool_nose_radius = 0.8e-3;  % 0.8mm 기본값
    
    % 이론적 거칠기 (기하학적)
    theoretical_Ra = feed_rate^2 / (32 * tool_nose_radius * 1000);  % mm to m
    
    % 절삭속도 효과 (높을수록 거칠기 개선)
    speed_factor = 100 / cutting_speed;  % 100 m/min 기준
    speed_factor = max(0.5, min(2.0, speed_factor));
    
    % 진동 효과 (경험적)
    vibration_factor = 1.5;  % Ti-6Al-4V는 진동하기 쉬움
    
    total_Ra = theoretical_Ra * speed_factor * vibration_factor;
    
    empirical_roughness = struct();
    empirical_roughness.macro = total_Ra * 0.6;
    empirical_roughness.meso6 = total_Ra * 0.3;
    empirical_roughness.meso5 = total_Ra * 0.15;
    empirical_roughness.micro3 = total_Ra * 0.1;
    empirical_roughness.total_rms = total_Ra;
    
    fprintf('경험적 거칠기 모델 사용: Ra = %.2f μm\n', total_Ra * 1e6);
end
```

### 6.1.2 Scale-Dependent Evolution Modeling

```matlab
% Lines 886-920: 시간에 따른 스케일별 거칠기 진화
function roughness_evolution = model_roughness_evolution(initial_roughness, wear_results, cutting_conditions, time_history)
    % 마모와 가공조건에 따른 거칠기 변화 모델링
    
    if isempty(wear_results)
        roughness_evolution = initial_roughness;
        return;
    end
    
    % 마모가 각 스케일에 미치는 영향
    wear_depth = wear_results.total_depth;  % 총 마모 깊이
    wear_rate = wear_results.total_rate;    % 마모율
    
    % 스케일별 영향 계수
    scale_sensitivity = struct();
    scale_sensitivity.macro = 0.1;    % 거시 스케일: 마모에 덜 민감
    scale_sensitivity.meso6 = 0.3;    % 중간 스케일: 중간 민감도
    scale_sensitivity.meso5 = 0.5;
    scale_sensitivity.micro3 = 0.8;   % 미시 스케일: 마모에 매우 민감
    
    % 시간 진화 모델
    roughness_evolution = initial_roughness;
    field_names = fieldnames(initial_roughness);
    
    for i = 1:length(field_names)
        field_name = field_names{i};
        
        if contains(field_name, '_length') || strcmp(field_name, 'total_rms')
            continue;  % 길이 스케일이나 총합은 건너뜀
        end
        
        if isfield(scale_sensitivity, field_name)
            sensitivity = scale_sensitivity.(field_name);
            
            % 마모에 의한 거칠기 변화
            wear_induced_change = sensitivity * wear_depth * 0.1;  % 10% 비례
            
            % 가공 진동에 의한 거칠기 증가
            vibration_increase = calculate_vibration_induced_roughness(cutting_conditions, field_name);
            
            % 총 변화
            total_change = wear_induced_change + vibration_increase;
            
            % 새로운 거칠기 값
            new_roughness = initial_roughness.(field_name) + total_change;
            
            % 물리적 제한 (너무 커지지 않도록)
            max_allowable = initial_roughness.(field_name) * 3;
            roughness_evolution.(field_name) = min(new_roughness, max_allowable);
        end
    end
    
    % 총 RMS 거칠기 재계산
    scale_values = [roughness_evolution.macro, roughness_evolution.meso6, ...
                   roughness_evolution.meso5, roughness_evolution.micro3];
    roughness_evolution.total_rms = sqrt(sum(scale_values.^2));
    
    fprintf('거칠기 진화 계산 완료:\n');
    fprintf('  - 초기 Ra: %.2f μm → 최종 Ra: %.2f μm\n', ...
        initial_roughness.total_rms*1e6, roughness_evolution.total_rms*1e6);
end

function vibration_roughness = calculate_vibration_induced_roughness(cutting_conditions, scale_name)
    % 절삭 진동이 각 스케일 거칠기에 미치는 영향
    
    spindle_speed = cutting_conditions.spindle_speed;  % rpm
    feed_rate = cutting_conditions.feed;               % mm/rev
    
    % 주요 진동 주파수 (tooth passing frequency)
    num_teeth = 4;  % 4날 엔드밀 가정
    tooth_frequency = spindle_speed * num_teeth / 60;  % Hz
    
    % 스케일별 진동 민감도
    switch scale_name
        case 'macro'
            vibration_amplitude = 1e-6;  % 1 μm (저주파 진동)
            sensitivity = 0.1;
        case {'meso6', 'meso5'}
            vibration_amplitude = 0.5e-6;  % 0.5 μm (중주파 진동)
            sensitivity = 0.3;
        case {'micro3', 'micro2', 'micro1'}
            vibration_amplitude = 0.2e-6;  % 0.2 μm (고주파 진동)
            sensitivity = 0.5;
        otherwise
            vibration_amplitude = 0.3e-6;
            sensitivity = 0.2;
    end
    
    % 진동에 의한 거칠기 증가 (시간과 조건에 비례)
    vibration_roughness = vibration_amplitude * sensitivity * feed_rate / 0.1;  % 0.1 mm/rev 기준 정규화
end
```

### 6.1.3 Contact Mechanics Integration

```matlab
% Lines 921-950: 거칠기와 접촉역학의 상호작용
function contact_roughness_interaction = integrate_roughness_contact_mechanics(roughness_results, contact_results, material_props)
    % 표면 거칠기가 접촉 특성에 미치는 영향
    
    if isempty(roughness_results) || isempty(contact_results)
        contact_roughness_interaction = struct();
        return;
    end
    
    % 실제 접촉 면적 계산 (Greenwood-Williamson 모델)
    total_roughness = roughness_results.total_rms;
    nominal_pressure = mean(contact_results.pressure_distribution);
    
    % 거칠기 매개변수
    beta = 0.05;  % 접촉점 밀도 매개변수
    sigma = total_roughness;  % RMS 거칠기
    
    % 무차원 압력
    hardness = material_props.hardness;
    dimensionless_pressure = nominal_pressure / hardness;
    
    % 실제 접촉 면적 비율
    if dimensionless_pressure < 0.01
        % 탄성 접촉 영역
        contact_area_ratio = 1.25 * beta * dimensionless_pressure^0.94;
    else
        % 소성 접촉 영역
        contact_area_ratio = min(1.0, beta * dimensionless_pressure);
    end
    
    % 접촉 저항 계산
    thermal_conductivity = material_props.thermal_conductivity;
    electrical_conductivity = material_props.electrical_conductivity;
    
    % 열 접촉 저항 (거칠기에 반비례)
    thermal_contact_resistance = sigma / (thermal_conductivity * contact_area_ratio);
    
    % 전기 접촉 저항
    electrical_contact_resistance = sigma / (electrical_conductivity * contact_area_ratio);
    
    % 마찰 계수 (거칠기 의존성)
    base_friction_coefficient = 0.3;  % Ti-6Al-4V vs 카바이드
    roughness_effect = 1 + 0.5 * log10(total_roughness / 1e-6);  % 1 μm 기준
    effective_friction_coefficient = base_friction_coefficient * roughness_effect;
    effective_friction_coefficient = min(effective_friction_coefficient, 0.8);  % 상한 제한
    
    contact_roughness_interaction = struct();
    contact_roughness_interaction.real_contact_area_ratio = contact_area_ratio;
    contact_roughness_interaction.thermal_contact_resistance = thermal_contact_resistance;
    contact_roughness_interaction.electrical_contact_resistance = electrical_contact_resistance;
    contact_roughness_interaction.effective_friction_coefficient = effective_friction_coefficient;
    contact_roughness_interaction.total_roughness = total_roughness;
    
    fprintf('거칠기-접촉 상호작용:\n');
    fprintf('  - 총 거칠기: %.2f μm\n', total_roughness * 1e6);
    fprintf('  - 실제 접촉면적 비율: %.3f\n', contact_area_ratio);
    fprintf('  - 열 접촉저항: %.2e m²·K/W\n', thermal_contact_resistance);
    fprintf('  - 유효 마찰계수: %.3f\n', effective_friction_coefficient);
end
```

## 6.2 Atomic-Scale Surface Phenomena

### 6.2.1 Surface Energy and Adhesion Modeling

**표면 에너지 계산 프레임워크**

```matlab
% Lines 951-980: 원자 스케일 표면 에너지 모델
function surface_energy_results = calculate_atomic_surface_phenomena(material_props, temperature_field, roughness_results)
    % 원자 스케일에서의 표면 에너지와 부착 현상 모델링
    
    % Ti-6Al-4V 표면 에너지 (실험값)
    surface_energy_Ti = 2.0;    % J/m² (Ti 표면 에너지)
    surface_energy_Al = 1.15;   % J/m² (Al 표면 에너지)
    surface_energy_V = 2.8;     % J/m² (V 표면 에너지)
    
    % 합금 조성 (Ti-6Al-4V)
    Ti_fraction = 0.90;  % 90% Ti
    Al_fraction = 0.06;  % 6% Al  
    V_fraction = 0.04;   % 4% V
    
    % 혼합 법칙으로 합금 표면 에너지 계산
    alloy_surface_energy = Ti_fraction * surface_energy_Ti + ...
                          Al_fraction * surface_energy_Al + ...
                          V_fraction * surface_energy_V;
    
    % 온도 의존성 (고온에서 표면 에너지 감소)
    if ~isempty(temperature_field)
        avg_temperature = mean(temperature_field.temperature);
    else
        avg_temperature = 20;  % 기본값
    end
    
    % 온도 보정 계수
    T_kelvin = avg_temperature + 273.15;
    T_ref = 293.15;  % 20°C
    
    temperature_correction = 1 - 0.0002 * (T_kelvin - T_ref);  % 온도 상승시 에너지 감소
    temperature_correction = max(temperature_correction, 0.5);  % 최소 50% 유지
    
    effective_surface_energy = alloy_surface_energy * temperature_correction;
    
    % 거칠기 효과 (표면적 증가)
    if ~isempty(roughness_results)
        roughness_factor = calculate_roughness_surface_area_factor(roughness_results);
        effective_surface_area = roughness_factor;  % 명목 면적 대비 실제 면적
    else
        effective_surface_area = 1.2;  % 기본 20% 증가
    end
    
    surface_energy_results = struct();
    surface_energy_results.base_surface_energy = alloy_surface_energy;
    surface_energy_results.temperature_corrected = effective_surface_energy;
    surface_energy_results.effective_surface_area = effective_surface_area;
    surface_energy_results.total_surface_energy = effective_surface_energy * effective_surface_area;
    surface_energy_results.temperature = avg_temperature;
    
    fprintf('원자 스케일 표면 에너지:\n');
    fprintf('  - 기본 표면 에너지: %.2f J/m²\n', alloy_surface_energy);
    fprintf('  - 온도 보정 (%.0f°C): %.2f J/m²\n', avg_temperature, effective_surface_energy);
    fprintf('  - 거칠기 면적 인자: %.2f\n', effective_surface_area);
    fprintf('  - 유효 표면 에너지: %.2f J/m²\n', effective_surface_energy * effective_surface_area);
end

function roughness_factor = calculate_roughness_surface_area_factor(roughness_results)
    % 거칠기로 인한 표면적 증가 계산
    
    % 프랙탈 표면 모델 사용
    if isfield(roughness_results, 'fractal_dimension')
        fractal_dim = roughness_results.fractal_dimension;
    else
        fractal_dim = 2.2;  % 기본값
    end
    
    % Mandelbrot의 프랙탈 표면적 공식
    % A_real / A_nominal = (L/l)^(D-2)
    % 여기서 L: 측정 스케일, l: 원자 스케일, D: 프랙탈 차원
    
    measurement_scale = 1e-6;  % 1 μm 측정 스케일
    atomic_scale = 3e-10;      % 3 Å 원자 스케일
    
    scale_ratio = measurement_scale / atomic_scale;
    roughness_factor = scale_ratio^(fractal_dim - 2);
    
    % 물리적 제한 (너무 커지지 않도록)
    roughness_factor = min(roughness_factor, 10);  % 최대 10배
    roughness_factor = max(roughness_factor, 1);   % 최소 1배
end
```

**부착 현상 모델링**

```matlab
% Lines 981-1010: 도구-워크피스 부착 모델
function adhesion_results = model_tool_workpiece_adhesion(surface_energy_results, contact_results, material_props)
    % 도구와 워크피스 간의 부착 현상 모델링
    
    if isempty(contact_results)
        adhesion_results = struct('adhesion_force', 0, 'sticking_probability', 0);
        return;
    end
    
    % 접촉 면적
    if isfield(contact_results, 'contact_area')
        contact_area = contact_results.contact_area;
    else
        contact_area = 1e-6;  % 1 mm² 기본값
    end
    
    % 접촉 압력
    if isfield(contact_results, 'pressure_distribution')
        contact_pressure = mean(contact_results.pressure_distribution);
    else
        contact_pressure = 100e6;  % 100 MPa 기본값
    end
    
    % Work of adhesion (Dupré 방정식)
    surface_energy_tool = 3.5;     % J/m² (WC-Co 카바이드)
    surface_energy_workpiece = surface_energy_results.temperature_corrected;
    interface_energy = 0.5;        % J/m² (Ti-WC 계면 에너지)
    
    work_of_adhesion = surface_energy_tool + surface_energy_workpiece - interface_energy;
    
    % 부착력 계산 (Johnson-Kendall-Roberts 모델)
    adhesion_force_per_area = work_of_adhesion;  % Pa
    total_adhesion_force = adhesion_force_per_area * contact_area;  % N
    
    % 분리에 필요한 힘 (접촉압력 + 부착력)
    separation_force = contact_pressure * contact_area + total_adhesion_force;
    
    % 달라붙을 확률 계산 (온도와 압력 의존)
    temperature = surface_energy_results.temperature;
    
    % 볼츠만 분포 기반
    k_boltzmann = 1.38e-23;  % J/K
    T_kelvin = temperature + 273.15;
    
    activation_energy = work_of_adhesion * 1e-20;  % J (원자당 부착 에너지)
    thermal_energy = k_boltzmann * T_kelvin;
    
    sticking_probability = exp(-activation_energy / thermal_energy);
    
    % 압력 효과 (높은 압력에서 부착 증가)
    pressure_factor = 1 + 0.1 * log10(contact_pressure / 1e6);  % MPa 기준
    sticking_probability = sticking_probability * pressure_factor;
    sticking_probability = min(sticking_probability, 1.0);
    
    adhesion_results = struct();
    adhesion_results.work_of_adhesion = work_of_adhesion;
    adhesion_results.adhesion_force = total_adhesion_force;
    adhesion_results.separation_force = separation_force;
    adhesion_results.sticking_probability = sticking_probability;
    adhesion_results.contact_area = contact_area;
    
    fprintf('부착 현상 분석:\n');
    fprintf('  - 부착 일: %.2f J/m²\n', work_of_adhesion);
    fprintf('  - 총 부착력: %.2f N\n', total_adhesion_force);
    fprintf('  - 분리 필요력: %.2f N\n', separation_force);
    fprintf('  - 달라붙을 확률: %.3f\n', sticking_probability);
end
```

### 6.2.2 Diffusion and Mass Transfer

**원자 확산 상세 모델**

```matlab
% Lines 1011-1050: 원자 확산 상세 모델링
function diffusion_detailed = model_atomic_diffusion_detailed(material_props, temperature_field, contact_results, time_params)
    % 원자 스케일 확산 현상의 상세 모델링
    
    if isempty(temperature_field)
        diffusion_detailed = struct('diffusion_flux', 0, 'concentration_profile', []);
        return;
    end
    
    % 확산 매개변수 (Ti-6Al-4V 시스템)
    diffusion_data = get_diffusion_parameters();
    
    % 접촉 영역 온도
    if ~isempty(contact_results)
        contact_temperature = calculate_average_contact_temperature(contact_results, temperature_field);
    else
        contact_temperature = max(temperature_field.temperature);
    end
    
    T_kelvin = contact_temperature + 273.15;
    R = 8.314;  % J/mol·K
    
    % 다중 원소 확산 계수 계산
    diffusion_coefficients = struct();
    
    for element = {'Ti', 'Al', 'V', 'C'}
        element_name = element{1};
        if isfield(diffusion_data, element_name)
            data = diffusion_data.(element_name);
            D = data.D0 * exp(-data.Q / (R * T_kelvin));
            diffusion_coefficients.(element_name) = D;
        end
    end
    
    % 농도 구배 설정 (도구-워크피스 계면)
    interface_thickness = 1e-9;  % 1 nm
    concentration_gradients = calculate_concentration_gradients(interface_thickness);
    
    % 확산 플럭스 계산 (Fick의 제1법칙)
    diffusion_fluxes = struct();
    
    element_names = fieldnames(diffusion_coefficients);
    for i = 1:length(element_names)
        element_name = element_names{i};
        D = diffusion_coefficients.(element_name);
        grad_C = concentration_gradients.(element_name);
        
        flux = -D * grad_C;  % kg/m²·s
        diffusion_fluxes.(element_name) = flux;
    end
    
    % 시간 진화 계산 (Fick의 제2법칙)
    if isfield(time_params, 'time_steps') && ~isempty(time_params.time_steps)
        concentration_profiles = solve_diffusion_equation(diffusion_coefficients, ...
            concentration_gradients, time_params, interface_thickness);
    else
        concentration_profiles = [];
    end
    
    diffusion_detailed = struct();
    diffusion_detailed.temperature = contact_temperature;
    diffusion_detailed.diffusion_coefficients = diffusion_coefficients;
    diffusion_detailed.concentration_gradients = concentration_gradients;
    diffusion_detailed.diffusion_fluxes = diffusion_fluxes;
    diffusion_detailed.concentration_profiles = concentration_profiles;
    
    fprintf('원자 확산 상세 분석:\n');
    fprintf('  - 접촉 온도: %.1f°C\n', contact_temperature);
    element_names = fieldnames(diffusion_coefficients);
    for i = 1:length(element_names)
        element_name = element_names{i};
        D = diffusion_coefficients.(element_name);
        flux = diffusion_fluxes.(element_name);
        fprintf('  - %s 확산계수: %.2e m²/s, 플럭스: %.2e kg/m²·s\n', element_name, D, abs(flux));
    end
end

function diffusion_data = get_diffusion_parameters()
    % 각 원소의 확산 매개변수 데이터베이스
    
    diffusion_data = struct();
    
    % 티타늄 자체 확산
    diffusion_data.Ti.D0 = 2.5e-4;   % m²/s
    diffusion_data.Ti.Q = 153000;    % J/mol
    
    % 알루미늄 확산 (Ti 내에서)
    diffusion_data.Al.D0 = 1.8e-5;   % m²/s
    diffusion_data.Al.Q = 142000;    % J/mol
    
    % 바나듐 확산 (Ti 내에서)  
    diffusion_data.V.D0 = 3.2e-5;    % m²/s
    diffusion_data.V.Q = 145000;     % J/mol
    
    % 탄소 확산 (Ti 내에서) - 도구에서 오는 원소
    diffusion_data.C.D0 = 2.3e-4;    % m²/s
    diffusion_data.C.Q = 150000;     % J/mol
end

function gradients = calculate_concentration_gradients(interface_thickness)
    % 도구-워크피스 계면에서의 농도 구배 계산
    
    gradients = struct();
    
    % 도구(WC-Co) 조성
    tool_composition.C = 0.85;   % 85% 탄소 (카바이드)
    tool_composition.W = 0.12;   % 12% 텅스텐
    tool_composition.Co = 0.03;  % 3% 코발트
    
    % 워크피스(Ti-6Al-4V) 조성
    workpiece_composition.Ti = 0.90;  % 90% 티타늄
    workpiece_composition.Al = 0.06;  % 6% 알루미늄
    workpiece_composition.V = 0.04;   % 4% 바나듐
    workpiece_composition.C = 0.0008; % 0.08% 탄소
    
    % 농도 구배 계산 (단위: kg/m⁴)
    gradients.C = (tool_composition.C - workpiece_composition.C) / interface_thickness;
    gradients.Ti = (0 - workpiece_composition.Ti) / interface_thickness;  % 도구에는 Ti 없음
    gradients.Al = (0 - workpiece_composition.Al) / interface_thickness;  % 도구에는 Al 없음
    gradients.V = (0 - workpiece_composition.V) / interface_thickness;    % 도구에는 V 없음
end
```

### 6.2.3 Crystal Structure and Phase Transformations

**상변태 모델링**

```matlab
% Lines 1051-1090: 고온 상변태 모델
function phase_transformation = model_phase_transformations(material_props, temperature_field, stress_results)
    % Ti-6Al-4V의 고온 상변태 모델링
    
    if isempty(temperature_field)
        phase_transformation = struct('alpha_fraction', 1.0, 'beta_fraction', 0.0);
        return;
    end
    
    % Ti-6Al-4V 상변태 온도
    T_alpha_beta = 995;      % °C (α+β → β 변태 시작 온도)
    T_complete_beta = 1050;  % °C (완전 β상 변태 온도)
    
    % 온도 분포
    temperatures = temperature_field.temperature;
    n_points = length(temperatures);
    
    % 각 점에서의 상분율 계산
    alpha_fractions = zeros(n_points, 1);
    beta_fractions = zeros(n_points, 1);
    
    for i = 1:n_points
        T_local = temperatures(i);
        
        if T_local < T_alpha_beta
            % 완전 α+β 상 (평형상태)
            alpha_fractions(i) = 0.85;  % 85% α상
            beta_fractions(i) = 0.15;   % 15% β상
            
        elseif T_local < T_complete_beta
            % 부분 β 변태 (선형 보간)
            transformation_progress = (T_local - T_alpha_beta) / (T_complete_beta - T_alpha_beta);
            
            % Johnson-Mehl-Avrami 변태 속도론 적용
            % 시간 의존성 고려 (빠른 가열 가정)
            heating_rate = 1000;  % K/s (빠른 가열)
            time_at_temp = 0.001; % 1 ms (짧은 체류시간)
            
            avrami_exponent = 2.5;  % Ti 합금의 경험값
            kinetic_factor = 1 - exp(-(time_at_temp * heating_rate * transformation_progress)^avrami_exponent);
            
            beta_fractions(i) = 0.15 + (1.0 - 0.15) * kinetic_factor;
            alpha_fractions(i) = 1.0 - beta_fractions(i);
            
        else
            % 완전 β상
            alpha_fractions(i) = 0.0;
            beta_fractions(i) = 1.0;
        end
    end
    
    % 응력 효과 (응력이 상변태에 미치는 영향)
    if ~isempty(stress_results) && isfield(stress_results, 'von_mises_stress')
        stress_effects = calculate_stress_induced_transformation(stress_results, temperatures);
        
        % 응력은 β상 형성을 촉진
        beta_fractions = beta_fractions + stress_effects.beta_enhancement;
        alpha_fractions = 1.0 - beta_fractions;
        
        % 물리적 제한
        beta_fractions = max(0, min(1, beta_fractions));
        alpha_fractions = max(0, min(1, alpha_fractions));
    end
    
    % 평균 상분율
    avg_alpha_fraction = mean(alpha_fractions);
    avg_beta_fraction = mean(beta_fractions);
    
    % 상변태가 물성에 미치는 영향
    property_changes = calculate_phase_property_changes(avg_alpha_fraction, avg_beta_fraction);
    
    phase_transformation = struct();
    phase_transformation.alpha_fractions = alpha_fractions;
    phase_transformation.beta_fractions = beta_fractions;
    phase_transformation.avg_alpha_fraction = avg_alpha_fraction;
    phase_transformation.avg_beta_fraction = avg_beta_fraction;
    phase_transformation.property_changes = property_changes;
    phase_transformation.transformation_temperatures = [T_alpha_beta, T_complete_beta];
    
    fprintf('상변태 분석:\n');
    fprintf('  - 평균 α상 분율: %.1f%%\n', avg_alpha_fraction * 100);
    fprintf('  - 평균 β상 분율: %.1f%%\n', avg_beta_fraction * 100);
    fprintf('  - 최고 온도: %.1f°C\n', max(temperatures));
    
    if avg_beta_fraction > 0.5
        fprintf('  ⚠️ 상당한 β상 변태 발생 (%.1f%%) - 기계적 성질 변화 예상\n', avg_beta_fraction * 100);
    end
end

function stress_effects = calculate_stress_induced_transformation(stress_results, temperatures)
    % 응력이 유도하는 상변태 효과
    
    von_mises_stress = stress_results.von_mises_stress;
    
    % 임계 응력 (상변태를 유도하는 최소 응력)
    critical_stress = 200e6;  % 200 MPa
    
    % 온도에 따른 임계 응력 변화
    stress_temperature_factor = zeros(size(temperatures));
    for i = 1:length(temperatures)
        T = temperatures(i);
        if T > 800
            stress_temperature_factor(i) = 0.5;  % 고온에서 임계응력 감소
        else
            stress_temperature_factor(i) = 1.0;
        end
    end
    
    effective_critical_stress = critical_stress * stress_temperature_factor;
    
    % 응력 유도 β상 증가
    beta_enhancement = zeros(size(von_mises_stress));
    
    for i = 1:length(von_mises_stress)
        if von_mises_stress(i) > effective_critical_stress(i)
            stress_ratio = von_mises_stress(i) / effective_critical_stress(i);
            beta_enhancement(i) = 0.1 * log10(stress_ratio);  % 로그 의존성
            beta_enhancement(i) = min(beta_enhancement(i), 0.3);  % 최대 30% 증가
        end
    end
    
    stress_effects = struct();
    stress_effects.beta_enhancement = beta_enhancement;
    stress_effects.critical_stress = effective_critical_stress;
end

function property_changes = calculate_phase_property_changes(alpha_fraction, beta_fraction)
    % 상분율 변화가 기계적 성질에 미치는 영향
    
    % 기준 물성 (α+β 상의 평형 조성)
    base_yield_strength = 830e6;      % Pa
    base_elastic_modulus = 113e9;     % Pa
    base_density = 4420;              % kg/m³
    
    % 상별 물성 차이
    % β상은 α상보다 부드럽고 밀도가 낮음
    beta_yield_factor = 0.7;      % β상의 항복강도는 α상의 70%
    beta_modulus_factor = 0.8;    % β상의 탄성계수는 α상의 80%
    beta_density_factor = 0.98;   % β상의 밀도는 α상의 98%
    
    % 혼합 법칙 적용
    effective_yield_strength = base_yield_strength * ...
        (alpha_fraction + beta_fraction * beta_yield_factor);
    
    effective_elastic_modulus = base_elastic_modulus * ...
        (alpha_fraction + beta_fraction * beta_modulus_factor);
    
    effective_density = base_density * ...
        (alpha_fraction + beta_fraction * beta_density_factor);
    
    property_changes = struct();
    property_changes.yield_strength = effective_yield_strength;
    property_changes.elastic_modulus = effective_elastic_modulus;
    property_changes.density = effective_density;
    property_changes.yield_strength_change = (effective_yield_strength - base_yield_strength) / base_yield_strength;
    property_changes.modulus_change = (effective_elastic_modulus - base_elastic_modulus) / base_elastic_modulus;
    property_changes.density_change = (effective_density - base_density) / base_density;
end
```

---

*Chapter 6은 SFDP v17.3의 표면 물리학 및 다중 스케일 모델링 엔진의 핵심 구현을 다룹니다. 프랙탈 표면 특성화, 다중 스케일 거칠기 분해, 원자 스케일 표면 현상 등을 통해 가공 표면의 복잡한 물리적 특성을 정확하게 모사합니다. 특히 Box-counting 방법을 이용한 프랙탈 차원 계산, 웨이블릿 변환을 통한 스케일별 거칠기 분해, 원자 확산 및 상변태 모델링을 통해 표면 품질 예측의 정확도를 크게 향상시켰습니다.*