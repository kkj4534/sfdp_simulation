# Chapter 10: Kalman Filter Architecture

## 10.1 Kalman Filter Theoretical Foundation

### 10.1.1 Introduction to Kalman Filtering in Machining Systems

**칼만 필터의 기본 원리와 가공 시스템 적용**

칼만 필터(Kalman Filter)는 1960년 Rudolf Kalman이 개발한 **최적 상태 추정 알고리즘**으로, 불확실성이 있는 동적 시스템에서 노이즈가 포함된 측정값들로부터 시스템의 실제 상태를 추정하는 데 사용됩니다. 

**가공 시스템에서 칼만 필터가 필요한 이유:**

1. **다중 물리량의 동시 추정**: 온도, 마모, 표면조도 등을 동시에 추정
2. **측정 노이즈 처리**: 센서 노이즈와 모델 불확실성 보상
3. **실시간 상태 추정**: 연속적인 가공 과정에서의 실시간 모니터링
4. **물리적 연관성 활용**: 온도와 마모, 마모와 표면조도 간의 물리적 관계 활용

**SFDP에서의 칼만 필터 설계 철학:**

기존의 단일 변수 칼만 필터와 달리, SFDP는 **다중 물리량과 그 불확실성을 동시에 추정**하는 확장된 칼만 필터를 구현합니다. 각 물리량에 대해 평균값과 분산을 모두 상태 변수로 포함하여, **추정의 신뢰도까지 정량화**합니다.

### 10.1.2 칼만 필터의 쉬운 이해 (공대 2학년 버전)

**칼만 필터를 일상적인 예시로 이해하기**

GPS 네비게이션을 생각해보세요. GPS 신호는 가끔 튀거나 건물에 막혀서 부정확할 때가 있죠? 그런데도 네비게이션은 여러분의 위치를 꽤 정확하게 추정합니다. 이게 바로 칼만 필터의 마법입니다!

**칼만 필터의 기본 아이디어:**

1. **예측하기**: "지금 속도로 1초 더 가면 어디쯤 있을까?"
2. **측정하기**: "GPS가 말하는 위치는 여기인데..."
3. **믿을만한 정도 판단**: "GPS 신호가 약하니까 70%만 믿자"
4. **최적 추정**: "예측값 30% + 측정값 70% = 최종 위치"

**SFDP에서 칼만 필터가 하는 일:**

```
🔮 예측 단계: "이전 온도가 500°C였으니, 0.1초 후에는 520°C일 거야"
📏 측정 단계: "센서가 측정한 온도는 515°C네?"
⚖️ 가중치 계산: "센서가 꽤 정확하니까 80% 정도 믿자"
✅ 최종 추정: 520°C × 0.2 + 515°C × 0.8 = 516°C
```

**수식을 단계별로 이해하기:**

예측 단계는 이렇게 생각하세요:
```
다음_상태 = 현재_상태 × 변화율 + 외부_입력
```

갱신 단계는 이렇게 생각하세요:
```
칼만_이득 = 예측_신뢰도 / (예측_신뢰도 + 측정_신뢰도)
최종_추정 = 예측값 + 칼만_이득 × (측정값 - 예측값)
```

### 10.1.3 Mathematical Framework (For Advanced Readers)

**기본 칼만 필터 방정식:**

**예측 단계 (Prediction Step):**
```
x̂(k|k-1) = F(k)x̂(k-1|k-1) + B(k)u(k)
P(k|k-1) = F(k)P(k-1|k-1)F(k)ᵀ + Q(k)
```

**갱신 단계 (Update Step):**
```
K(k) = P(k|k-1)H(k)ᵀ[H(k)P(k|k-1)H(k)ᵀ + R(k)]⁻¹
x̂(k|k) = x̂(k|k-1) + K(k)[z(k) - H(k)x̂(k|k-1)]
P(k|k) = [I - K(k)H(k)]P(k|k-1)
```

변수 설명:
- **x̂(k|k)**: 시점 k에서의 상태 추정값
- **K(k)**: 칼만 이득 (얼마나 측정값을 믿을지 결정)

**SFDP 다중 물리량 확장:**

SFDP에서는 각 물리량 i에 대해 평균 μᵢ와 분산 σᵢ²를 동시에 추정하므로:

```
x = [μ₁, σ₁², μ₂, σ₂², ..., μₙ, σₙ², t, E, C]ᵀ
```

이는 **불확실성까지 정량화**하는 고도화된 접근법으로, 기존 연구에서는 찾아보기 어려운 독창적인 설계입니다.

### 10.1.3 Physical Coupling Models in State Transition Matrix

**물리적 연관성 모델링의 필요성:**

가공 과정에서 각 물리량들은 독립적이지 않고 강한 연관성을 가집니다:

1. **열-기계적 커플링**: 온도 상승 → 재료 연화 → 마모 증가
2. **마모-표면 커플링**: 공구 마모 → 표면 거칠기 증가  
3. **동역학적 커플링**: 절삭력 변화 → 진동 발생 → 치수 정도 저하

**아레니우스 기반 열-마모 커플링:**

온도와 마모의 관계는 확산 마모 메커니즘을 기반으로 한 아레니우스 방정식으로 모델링됩니다:

```
마모율 = A × exp(-Eₐ/RT)
```

여기서 Eₐ는 활성화 에너지(Ti-6Al-4V: 45,000 J/mol)입니다.

**시간 상수 기반 동적 응답:**

각 물리량은 고유한 시간 상수를 가지며, 이는 시스템의 응답 특성을 결정합니다:
- **열적 응답**: τ_thermal = 30초 (열 확산 지배)
- **기계적 응답**: τ_force = 5초 (구조 동역학 지배)  
- **진동 응답**: τ_vibration = 2초 (모달 감쇠 지배)

## 10.2 Advanced Kalman Filter Implementation

### 10.1.1 Multi-Variable State Vector Design

**SFDP Kalman 상태 벡터의 설계 철학**

SFDP v17.3에서는 다음과 같은 15차원 상태 벡터를 사용합니다:

```matlab
% SFDP_kalman_fusion_suite.m:125-140에서 정의
state_vector = [
    temperature_mean;           % 1: 평균 온도 [°C]
    temperature_variance;       % 2: 온도 분산 [°C²]
    tool_wear_mean;            % 3: 평균 공구마모 [mm]
    tool_wear_variance;        % 4: 공구마모 분산 [mm²]
    surface_roughness_mean;    % 5: 평균 표면조도 [μm]
    surface_roughness_variance; % 6: 표면조도 분산 [μm²]
    cutting_force_mean;        % 7: 평균 절삭력 [N]
    cutting_force_variance;    % 8: 절삭력 분산 [N²]
    vibration_mean;           % 9: 평균 진동 [m/s²]
    vibration_variance;       % 10: 진동 분산 [m²/s⁴]
    dimensional_accuracy_mean; % 11: 평균 치수정도 [μm]
    dimensional_accuracy_variance; % 12: 치수정도 분산 [μm²]
    process_time;             % 13: 공정시간 [s]
    energy_consumption;       % 14: 에너지 소비 [J]
    overall_confidence       % 15: 전체 신뢰도 [0-1]
];
```

**각 변수의 물리적 의미와 모델링**

1. **온도 변수 (Temperature)**
   - 절삭영역의 평균 온도와 그 분산
   - 열역학 법칙에 기반한 진화 모델
   - Ti-6Al-4V의 경우 400-1200°C 범위

2. **공구마모 변수 (Tool Wear)**
   - VB (flank wear) 기준 마모량
   - Archard 법칙과 다중 마모 메커니즘 적용
   - 0-0.3mm 범위에서 모델링

3. **표면조도 변수 (Surface Roughness)**
   - Ra 기준 산술평균 거칠기
   - 다중스케일 프랙탈 모델 적용
   - 0.1-5.0μm 범위

**상태 전이 행렬의 물리적 설계**

```matlab
% initializeAdaptiveKalmanFilter 함수에서 (Lines 45-120)
function [kalman_state, A_matrix] = design_state_transition_matrix(cutting_conditions, material_props, dt)
    
    % 기본 단위행렬 (관성 효과)
    A_matrix = eye(15);
    
    % 물리적 연관성 모델링
    
    % 1. 온도 → 마모 커플링 (아레니우스 관계)
    activation_energy = 45000; % J/mol (Ti-6Al-4V 확산)
    R = 8.314; % 기체상수
    T_ref = 600 + 273.15; % 참조온도 [K]
    
    thermal_wear_coupling = dt * exp(-activation_energy/(R*T_ref));
    A_matrix(3,1) = thermal_wear_coupling; % 온도 → 마모 평균
    A_matrix(4,2) = thermal_wear_coupling * 0.1; % 온도분산 → 마모분산
    
    % 2. 마모 → 표면조도 커플링
    wear_roughness_coupling = dt * 0.02; % mm → μm 변환 포함
    A_matrix(5,3) = wear_roughness_coupling;
    A_matrix(6,4) = wear_roughness_coupling * 0.5;
    
    % 3. 온도 → 절삭력 커플링 (열연화 효과)
    thermal_softening = -dt * 0.15; % N/°C
    A_matrix(7,1) = thermal_softening;
    
    % 4. 절삭력 → 진동 커플링
    force_vibration_coupling = dt * 0.001; % N → m/s²
    A_matrix(9,7) = force_vibration_coupling;
    A_matrix(10,8) = force_vibration_coupling * 0.5;
    
    % 5. 진동 → 치수정도 커플링
    vibration_accuracy_coupling = dt * 0.1; % m/s² → μm
    A_matrix(11,9) = vibration_accuracy_coupling;
    A_matrix(12,10) = vibration_accuracy_coupling * 0.3;
    
    % 6. 시간 진화 (감쇠 효과)
    thermal_time_constant = 30; % 초
    force_time_constant = 5; % 초
    vibration_time_constant = 2; % 초
    
    A_matrix(1,1) = exp(-dt/thermal_time_constant);
    A_matrix(7,7) = exp(-dt/force_time_constant);
    A_matrix(9,9) = exp(-dt/vibration_time_constant);
    
    % 7. 신뢰도 진화 (경험 기반)
    confidence_decay = exp(-dt/100); % 100초 시상수
    A_matrix(15,15) = confidence_decay;
end
```

### 10.1.2 Adaptive Noise Covariance Tuning

**프로세스 노이즈 행렬의 적응적 조정**

```matlab
% calculateAdaptiveNoiseCovariance 함수에서 (Lines 180-250)
function [Q_adaptive, R_adaptive] = calculate_adaptive_noise_covariance(cutting_conditions, ...
    temperature_level, wear_level, measurement_history)
    
    % 기본 프로세스 노이즈 (15x15 대각행렬)
    Q_base = diag([
        25,     % 온도 평균 노이즈 [°C²]
        100,    % 온도 분산 노이즈 [°C⁴]
        0.001,  % 마모 평균 노이즈 [mm²]
        0.0001, % 마모 분산 노이즈 [mm⁴]
        0.01,   % 조도 평균 노이즈 [μm²]
        0.001,  % 조도 분산 노이즈 [μm⁴]
        100,    % 력 평균 노이즈 [N²]
        1000,   % 력 분산 노이즈 [N⁴]
        0.1,    % 진동 평균 노이즈 [m²/s⁴]
        0.01,   % 진동 분산 노이즈 [m⁴/s⁸]
        0.01,   % 정도 평균 노이즈 [μm²]
        0.001,  % 정도 분산 노이즈 [μm⁴]
        1,      % 시간 노이즈 [s²]
        100,    % 에너지 노이즈 [J²]
        0.01    % 신뢰도 노이즈 [1]
    ]);
    
    % 조건별 적응적 조정
    
    % 1. 절삭속도에 따른 조정
    speed_factor = cutting_conditions.speed / 100; % 100 m/min 기준 정규화
    Q_adaptive = Q_base * (1 + 0.5 * (speed_factor - 1));
    
    % 2. 온도 수준에 따른 조정
    if temperature_level > 800 % 고온에서 불확실성 증가
        thermal_multiplier = 1 + 0.3 * (temperature_level - 800) / 200;
        Q_adaptive(1,1) = Q_adaptive(1,1) * thermal_multiplier;
        Q_adaptive(2,2) = Q_adaptive(2,2) * thermal_multiplier^2;
    end
    
    % 3. 마모 수준에 따른 조정
    if wear_level > 0.15 % 심한 마모시 불확실성 증가
        wear_multiplier = 1 + 2 * (wear_level - 0.15) / 0.15;
        Q_adaptive(3,3) = Q_adaptive(3,3) * wear_multiplier;
        Q_adaptive(5,5) = Q_adaptive(5,5) * wear_multiplier; % 조도도 영향
    end
    
    % 측정 노이즈 적응적 조정
    R_base = diag([
        16,    % 온도 측정 노이즈 [°C²]
        0.0001, % 마모 측정 노이즈 [mm²]
        0.01,   % 조도 측정 노이즈 [μm²]
        25,     % 력 측정 노이즈 [N²]
        0.04    % 진동 측정 노이즈 [m²/s⁴]
    ]);
    
    % 측정 이력 기반 노이즈 추정
    if length(measurement_history) > 10
        measurement_residuals = diff(measurement_history, 1, 2);
        empirical_noise = var(measurement_residuals, 0, 2);
        
        % 경험적 노이즈와 이론적 노이즈의 가중 평균
        alpha = 0.3; % 가중치
        R_adaptive = (1-alpha) * R_base + alpha * diag(empirical_noise);
    else
        R_adaptive = R_base;
    end
end
```

### 10.1.3 Multi-Layer Kalman Fusion Strategy

**계층별 칼먼 융합 전략**

```matlab
% performKalmanMultiLayerFusion 함수에서 (Lines 300-450)
function [kalman_fused, fusion_confidence] = perform_kalman_multi_layer_fusion(...
    layer1_results, layer2_results, layer3_results, layer4_results, kalman_state, simulation_state)
    
    fprintf('🔀 다층 칼만 융합 시작\n');
    
    % 각 층의 예측 결과를 상태 벡터 형태로 변환
    layer_predictions = cell(4,1);
    layer_confidences = zeros(4,1);
    
    % Layer 1: 고급 물리 해석 결과 변환
    if ~isempty(layer1_results) && isfield(layer1_results, 'thermal')
        layer_predictions{1} = convert_to_state_vector(layer1_results, 'advanced_physics');
        layer_confidences(1) = layer1_results.overall_confidence;
        fprintf('  📊 Layer 1 변환 완료 (신뢰도: %.3f)\n', layer_confidences(1));
    end
    
    % Layer 2: 간소화 물리 해석 결과 변환
    if ~isempty(layer2_results) && isfield(layer2_results, 'thermal')
        layer_predictions{2} = convert_to_state_vector(layer2_results, 'simplified_physics');
        layer_confidences(2) = layer2_results.overall_confidence;
        fprintf('  📊 Layer 2 변환 완료 (신뢰도: %.3f)\n', layer_confidences(2));
    end
    
    % Layer 3: 경험적 평가 결과 변환
    if ~isempty(layer3_results) && isfield(layer3_results, 'tool_life')
        layer_predictions{3} = convert_to_state_vector(layer3_results, 'empirical');
        layer_confidences(3) = layer3_results.overall_confidence;
        fprintf('  📊 Layer 3 변환 완료 (신뢰도: %.3f)\n', layer_confidences(3));
    end
    
    % Layer 4: 데이터 보정 결과 변환
    if ~isempty(layer4_results) && isfield(layer4_results, 'corrected')
        layer_predictions{4} = convert_to_state_vector(layer4_results, 'corrected');
        layer_confidences(4) = layer4_results.correction_confidence;
        fprintf('  📊 Layer 4 변환 완료 (신뢰도: %.3f)\n', layer_confidences(4));
    end
    
    % 유효한 층 식별
    valid_layers = find(layer_confidences > 0);
    num_valid_layers = length(valid_layers);
    
    if num_valid_layers == 0
        warning('유효한 층이 없습니다');
        kalman_fused = [];
        fusion_confidence = 0;
        return;
    end
    
    fprintf('  ✅ %d개 유효 층 확인\n', num_valid_layers);
    
    % 층별 가중 칼먼 융합
    fused_state = zeros(15, 1);
    fused_covariance = zeros(15, 15);
    
    % 신뢰도 기반 가중치 계산
    weights = layer_confidences(valid_layers);
    weights = weights ./ sum(weights); % 정규화
    
    % 각 층의 기여도에 따른 융합
    for i = 1:num_valid_layers
        layer_idx = valid_layers(i);
        weight = weights(i);
        
        % 해당 층의 상태 예측
        layer_state = layer_predictions{layer_idx};
        
        if ~isempty(layer_state)
            % 칼먼 필터 업데이트 수행
            [updated_state, updated_covariance] = kalman_update_step(...
                kalman_state.current_state, kalman_state.covariance, ...
                layer_state, kalman_state.measurement_noise);
            
            % 가중 평균으로 융합
            fused_state = fused_state + weight * updated_state;
            fused_covariance = fused_covariance + weight^2 * updated_covariance;
            
            fprintf('    🔄 Layer %d 융합 (가중치: %.3f)\n', layer_idx, weight);
        end
    end
    
    % 융합 결과 품질 평가
    fusion_confidence = calculate_fusion_confidence(fused_state, fused_covariance, ...
        layer_confidences(valid_layers));
    
    % 최종 융합 결과 구성
    kalman_fused = struct();
    kalman_fused.state_vector = fused_state;
    kalman_fused.covariance_matrix = fused_covariance;
    kalman_fused.contributing_layers = valid_layers;
    kalman_fused.layer_weights = weights;
    kalman_fused.fusion_confidence = fusion_confidence;
    
    % 물리 변수별 추출
    kalman_fused.temperature = struct('mean', fused_state(1), 'variance', fused_state(2));
    kalman_fused.tool_wear = struct('mean', fused_state(3), 'variance', fused_state(4));
    kalman_fused.surface_roughness = struct('mean', fused_state(5), 'variance', fused_state(6));
    kalman_fused.cutting_force = struct('mean', fused_state(7), 'variance', fused_state(8));
    kalman_fused.vibration = struct('mean', fused_state(9), 'variance', fused_state(10));
    kalman_fused.dimensional_accuracy = struct('mean', fused_state(11), 'variance', fused_state(12));
    kalman_fused.process_time = fused_state(13);
    kalman_fused.energy_consumption = fused_state(14);
    kalman_fused.overall_confidence = fused_state(15);
    
    fprintf('🎯 칼먼 융합 완료: 전체 신뢰도 %.3f\n', fusion_confidence);
end
```

## 10.2 Variable-Specific Kalman Dynamics

### 10.2.1 Temperature Dynamics Modeling

**온도 칼먼 필터의 전용 동역학**

```matlab
% Temperature-specific Kalman dynamics
function [temp_kalman_params] = setup_temperature_kalman_dynamics()
    
    temp_kalman_params = struct();
    
    % 온도 상태 벡터: [T_avg, T_gradient, T_max, T_rate]
    temp_kalman_params.state_dimension = 4;
    
    % 상태 전이 행렬 (물리 기반)
    dt = 0.1; % 0.1초 간격
    thermal_diffusivity = 2.87e-6; % Ti-6Al-4V [m²/s]
    
    % 열확산 방정식 기반 상태 전이
    A_temp = [
        1, dt, 0, dt;                    % 평균온도
        0, exp(-dt*0.1), 0, 0;          % 온도구배 (감쇠)
        0, 0.5*dt, 0.9, 0;              % 최대온도
        0, 0, 0, exp(-dt*0.2)           % 온도변화율
    ];
    
    temp_kalman_params.state_transition = A_temp;
    
    % 프로세스 노이즈 (물리적 불확실성)
    Q_temp = diag([
        16,    % 평균온도 노이즈 [°C²]
        4,     % 구배 노이즈 [°C²/mm²]
        25,    % 최대온도 노이즈 [°C²]
        9      % 변화율 노이즈 [°C²/s²]
    ]);
    
    temp_kalman_params.process_noise = Q_temp;
    
    % 측정 행렬
    H_temp = [
        1, 0, 0, 0;    % 평균온도 측정
        0, 0, 1, 0     % 최대온도 측정
    ];
    
    temp_kalman_params.measurement_matrix = H_temp;
    
    % 측정 노이즈
    R_temp = diag([
        16,    % 평균온도 측정 오차 [°C²]
        25     % 최대온도 측정 오차 [°C²]
    ]);
    
    temp_kalman_params.measurement_noise = R_temp;
    
    % 적응형 조정 매개변수
    temp_kalman_params.adaptation = struct();
    temp_kalman_params.adaptation.range = 0.10; % ±10% 조정
    temp_kalman_params.adaptation.learning_rate = 0.05;
    temp_kalman_params.adaptation.forgetting_factor = 0.95;
end
```

### 10.2.2 Tool Wear Dynamics Modeling

**공구마모 전용 칼먼 동역학**

```matlab
% Tool wear specific Kalman dynamics  
function [wear_kalman_params] = setup_wear_kalman_dynamics()
    
    wear_kalman_params = struct();
    
    % 마모 상태 벡터: [VB_flank, VB_crater, wear_rate, accumulated_distance]
    wear_kalman_params.state_dimension = 4;
    
    dt = 0.1; % 시간 간격
    
    % Archard 마모 법칙 기반 상태 전이
    A_wear = [
        1, 0, dt, 0;                     % 플랭크 마모
        0, 1, 0.7*dt, 0;                 % 크레이터 마모 (상관관계)
        0, 0, 0.98, 0;                   % 마모율 (천천히 변화)
        0, 0, 0, 1                       % 누적 거리 (단조증가)
    ];
    
    wear_kalman_params.state_transition = A_wear;
    
    % 프로세스 노이즈 (마모의 확률적 특성)
    Q_wear = diag([
        1e-6,   % 플랭크 마모 노이즈 [mm²]
        5e-7,   % 크레이터 마모 노이즈 [mm²]
        1e-8,   % 마모율 노이즈 [mm²/s²]
        1e-4    % 거리 노이즈 [m²]
    ]);
    
    wear_kalman_params.process_noise = Q_wear;
    
    % 측정 행렬 (현미경 측정)
    H_wear = [
        1, 0, 0, 0;    % 플랭크 마모 직접 측정
        0, 1, 0, 0     % 크레이터 마모 직접 측정
    ];
    
    wear_kalman_params.measurement_matrix = H_wear;
    
    % 측정 노이즈 (측정 장비 정밀도)
    R_wear = diag([
        4e-6,   % 플랭크 마모 측정 오차 [mm²] (±2μm)
        9e-6    % 크레이터 마모 측정 오차 [mm²] (±3μm)
    ]);
    
    wear_kalman_params.measurement_noise = R_wear;
    
    % 적응형 조정 (온도 의존성)
    wear_kalman_params.adaptation = struct();
    wear_kalman_params.adaptation.range = 0.08; % ±8% 조정
    wear_kalman_params.adaptation.temperature_coupling = true;
    wear_kalman_params.adaptation.arrhenius_activation = 45000; % J/mol
end
```

### 10.2.3 Surface Roughness Dynamics Modeling

**표면조도 전용 칼먼 동역학**

```matlab
% Surface roughness specific Kalman dynamics
function [roughness_kalman_params] = setup_roughness_kalman_dynamics()
    
    roughness_kalman_params = struct();
    
    % 조도 상태 벡터: [Ra, Rz, fractal_dimension, waviness]
    roughness_kalman_params.state_dimension = 4;
    
    dt = 0.1;
    
    % 표면 형성 물리학 기반 상태 전이
    A_roughness = [
        0.95, 0.1*dt, 0, 0.05*dt;       % Ra (주변 인자들의 영향)
        0.05*dt, 0.92, 0, 0.1*dt;       % Rz 
        0, 0, 0.99, 0;                  % 프랙탈 차원 (천천히 변화)
        0, 0, 0, 0.88                   % 웨이브니스 (빠른 변화)
    ];
    
    roughness_kalman_params.state_transition = A_roughness;
    
    % 프로세스 노이즈 (표면 형성의 확률적 특성)
    Q_roughness = diag([
        0.01,   % Ra 노이즈 [μm²]
        0.04,   % Rz 노이즈 [μm²]
        0.001,  % 프랙탈 차원 노이즈
        0.0025  % 웨이브니스 노이즈 [μm²]
    ]);
    
    roughness_kalman_params.process_noise = Q_roughness;
    
    % 측정 행렬 (표면 거칠기 측정기)
    H_roughness = [
        1, 0, 0, 0;    % Ra 직접 측정
        0, 1, 0, 0;    % Rz 직접 측정
        0, 0, 0, 1     % 웨이브니스 측정
    ];
    
    roughness_kalman_params.measurement_matrix = H_roughness;
    
    % 측정 노이즈
    R_roughness = diag([
        0.0025,  % Ra 측정 오차 [μm²] (±0.05μm)
        0.01,    % Rz 측정 오차 [μm²] (±0.1μm)
        0.0004   % 웨이브니스 측정 오차 [μm²]
    ]);
    
    roughness_kalman_params.measurement_noise = R_roughness;
    
    % 적응형 조정
    roughness_kalman_params.adaptation = struct();
    roughness_kalman_params.adaptation.range = 0.12; % ±12% 조정
    roughness_kalman_params.adaptation.wear_dependency = true;
    roughness_kalman_params.adaptation.feed_rate_coupling = 2.5; % mm/rev → μm
end
```

## 10.3 Real-Time Kalman Update Strategies

### 10.3.1 Sequential Update Implementation

**순차적 측정 업데이트**

```matlab
% performSequentialKalmanUpdate 함수에서 (Lines 500-650)
function [updated_state, updated_covariance] = perform_sequential_kalman_update(...
    prior_state, prior_covariance, measurements, measurement_times, kalman_params)
    
    fprintf('🔄 순차적 칼먼 업데이트 시작\n');
    
    % 초기값 설정
    current_state = prior_state;
    current_covariance = prior_covariance;
    
    num_measurements = length(measurements);
    update_history = cell(num_measurements, 1);
    
    for i = 1:num_measurements
        measurement_time = measurement_times(i);
        measurement_value = measurements{i};
        
        fprintf('  📊 측정 %d/%d 처리 중 (시간: %.2fs)\n', i, num_measurements, measurement_time);
        
        % 1. 시간 진행에 따른 예측 단계
        if i > 1
            dt = measurement_time - measurement_times(i-1);
            
            % 적응형 상태 전이 행렬 계산
            A_adaptive = update_state_transition_matrix(kalman_params.state_transition, dt);
            
            % 예측 단계
            predicted_state = A_adaptive * current_state;
            predicted_covariance = A_adaptive * current_covariance * A_adaptive' + kalman_params.process_noise;
        else
            predicted_state = current_state;
            predicted_covariance = current_covariance;
        end
        
        % 2. 측정값 타입 식별 및 처리
        measurement_type = identify_measurement_type(measurement_value);
        
        switch measurement_type
            case 'temperature'
                [H, R] = get_temperature_measurement_model(kalman_params);
                measured_values = [measurement_value.mean; measurement_value.max];
                
            case 'wear'
                [H, R] = get_wear_measurement_model(kalman_params);
                measured_values = [measurement_value.flank; measurement_value.crater];
                
            case 'roughness'
                [H, R] = get_roughness_measurement_model(kalman_params);
                measured_values = [measurement_value.Ra; measurement_value.Rz; measurement_value.waviness];
                
            otherwise
                warning('알 수 없는 측정 타입: %s', measurement_type);
                continue;
        end
        
        % 3. 칼먼 이득 계산
        innovation_covariance = H * predicted_covariance * H' + R;
        kalman_gain = predicted_covariance * H' / innovation_covariance;
        
        % 4. 상태 업데이트
        innovation = measured_values - H * predicted_state;
        current_state = predicted_state + kalman_gain * innovation;
        current_covariance = (eye(size(kalman_gain, 1)) - kalman_gain * H) * predicted_covariance;
        
        % 5. 업데이트 품질 평가
        innovation_normalized = innovation' / innovation_covariance * innovation;
        update_quality = exp(-0.5 * innovation_normalized); % Chi-square 기반
        
        % 6. 이력 저장
        update_history{i} = struct();
        update_history{i}.measurement_type = measurement_type;
        update_history{i}.innovation = innovation;
        update_history{i}.kalman_gain = kalman_gain;
        update_history{i}.update_quality = update_quality;
        update_history{i}.state_after_update = current_state;
        
        fprintf('    ✅ %s 측정 업데이트 완료 (품질: %.3f)\n', measurement_type, update_quality);
    end
    
    % 최종 결과
    updated_state = current_state;
    updated_covariance = current_covariance;
    
    fprintf('🎯 순차적 업데이트 완료: %d개 측정값 처리\n', num_measurements);
end
```

### 10.3.2 Parallel Kalman Processing

**병렬 칼먼 처리**

```matlab
% performParallelKalmanProcessing 함수에서 (Lines 700-850)
function [parallel_results] = perform_parallel_kalman_processing(...
    state_vector, covariance_matrix, measurement_batch, kalman_configs)
    
    fprintf('⚡ 병렬 칼먼 처리 시작\n');
    
    num_variables = length(kalman_configs);
    parallel_results = cell(num_variables, 1);
    
    % MATLAB Parallel Computing Toolbox 사용
    if license('test', 'Distrib_Computing_Toolbox')
        
        % 병렬 풀 시작
        if isempty(gcp('nocreate'))
            parpool('local', min(4, num_variables));
        end
        
        % 변수별 병렬 처리
        parfor var_idx = 1:num_variables
            var_name = kalman_configs(var_idx).variable_name;
            
            fprintf('  🔄 변수 %s 병렬 처리 중...\n', var_name);
            
            % 해당 변수의 상태와 측정값 추출
            var_state = extract_variable_state(state_vector, var_name);
            var_measurements = extract_variable_measurements(measurement_batch, var_name);
            var_config = kalman_configs(var_idx);
            
            % 변수별 칼먼 필터 실행
            [var_updated_state, var_confidence] = run_variable_kalman_filter(...
                var_state, var_measurements, var_config);
            
            % 결과 저장
            parallel_results{var_idx} = struct();
            parallel_results{var_idx}.variable_name = var_name;
            parallel_results{var_idx}.updated_state = var_updated_state;
            parallel_results{var_idx}.confidence = var_confidence;
            parallel_results{var_idx}.processing_time = toc;
            
            fprintf('  ✅ 변수 %s 처리 완료\n', var_name);
        end
        
    else
        % 순차 처리 (Parallel Toolbox 없는 경우)
        fprintf('  ⚠️ Parallel Toolbox 없음 - 순차 처리\n');
        
        for var_idx = 1:num_variables
            var_name = kalman_configs(var_idx).variable_name;
            
            var_state = extract_variable_state(state_vector, var_name);
            var_measurements = extract_variable_measurements(measurement_batch, var_name);
            var_config = kalman_configs(var_idx);
            
            [var_updated_state, var_confidence] = run_variable_kalman_filter(...
                var_state, var_measurements, var_config);
            
            parallel_results{var_idx} = struct();
            parallel_results{var_idx}.variable_name = var_name;
            parallel_results{var_idx}.updated_state = var_updated_state;
            parallel_results{var_idx}.confidence = var_confidence;
        end
    end
    
    fprintf('⚡ 병렬 처리 완료: %d개 변수\n', num_variables);
end
```

### 10.3.3 Adaptive Learning Rate Control

**적응형 학습률 제어**

```matlab
% controlAdaptiveLearningRate 함수에서 (Lines 900-1050)
function [updated_learning_rates] = control_adaptive_learning_rate(...
    kalman_history, performance_metrics, adaptation_config)
    
    fprintf('📈 적응형 학습률 제어 시작\n');
    
    num_variables = length(adaptation_config.variables);
    updated_learning_rates = struct();
    
    for var_idx = 1:num_variables
        var_name = adaptation_config.variables{var_idx};
        
        % 최근 성능 이력 분석
        recent_history = get_recent_performance_history(kalman_history, var_name, 20);
        
        if length(recent_history) < 5
            % 충분한 이력이 없으면 기본값 사용
            updated_learning_rates.(var_name) = adaptation_config.default_learning_rate;
            continue;
        end
        
        % 성능 지표 계산
        prediction_errors = [recent_history.prediction_error];
        innovation_magnitudes = [recent_history.innovation_magnitude];
        confidence_levels = [recent_history.confidence];
        
        % 1. 예측 오차 추세 분석
        error_trend = calculate_trend(prediction_errors);
        
        % 2. 혁신 크기 변화 분석
        innovation_trend = calculate_trend(innovation_magnitudes);
        
        % 3. 신뢰도 변화 분석
        confidence_trend = calculate_trend(confidence_levels);
        
        % 현재 학습률 가져오기
        current_lr = get_current_learning_rate(kalman_history, var_name);
        
        % 적응형 조정 로직
        lr_adjustment_factor = 1.0;
        
        % 오차가 증가하는 경우 → 학습률 증가
        if error_trend > 0.1
            lr_adjustment_factor = lr_adjustment_factor * 1.2;
            fprintf('  📈 %s: 오차 증가 → 학습률 증가\n', var_name);
        end
        
        % 오차가 감소하는 경우 → 학습률 유지 또는 약간 감소
        if error_trend < -0.05
            lr_adjustment_factor = lr_adjustment_factor * 0.95;
            fprintf('  📉 %s: 오차 감소 → 학습률 안정화\n', var_name);
        end
        
        % 혁신이 너무 큰 경우 → 학습률 감소 (과적응 방지)
        if mean(innovation_magnitudes(end-5:end)) > adaptation_config.innovation_threshold
            lr_adjustment_factor = lr_adjustment_factor * 0.8;
            fprintf('  🚫 %s: 혁신 과대 → 학습률 감소\n', var_name);
        end
        
        % 신뢰도가 낮은 경우 → 학습률 증가 (더 적극적 적응)
        if mean(confidence_levels(end-5:end)) < 0.7
            lr_adjustment_factor = lr_adjustment_factor * 1.1;
            fprintf('  ⬆️ %s: 낮은 신뢰도 → 학습률 증가\n', var_name);
        end
        
        % 조정 범위 제한
        lr_adjustment_factor = max(0.5, min(2.0, lr_adjustment_factor));
        
        % 새로운 학습률 계산
        new_learning_rate = current_lr * lr_adjustment_factor;
        new_learning_rate = max(adaptation_config.min_learning_rate, ...
                               min(adaptation_config.max_learning_rate, new_learning_rate));
        
        updated_learning_rates.(var_name) = new_learning_rate;
        
        fprintf('  🎯 %s: %.4f → %.4f (조정: %.2f배)\n', ...
               var_name, current_lr, new_learning_rate, lr_adjustment_factor);
    end
    
    fprintf('📈 학습률 제어 완료\n');
end
```