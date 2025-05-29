# Chapter 3: Mathematical Methods and Numerical Techniques

## 3.1 Finite Element Method (FEM) Basics for Engineers

### 3.1.1 Discretization and Shape Functions

**왜 FEM이 필요한가?**

앞서 Chapter 2에서 본 물리 방정식들은 대부분 편미분방정식(PDE)입니다:

```
∇·(k∇T) + Q = ρcp(∂T/∂t)    (열전도)
∇·σ + ρg = 0                 (역학 평형)
```

이런 방정식들은 복잡한 형상과 경계조건에서는 해석적으로 풀 수 없습니다. 따라서 수치해법이 필요합니다.

**FEM의 기본 아이디어**

복잡한 영역을 작은 조각(요소, Element)으로 나누어서 문제를 단순화합니다:

1. **영역 분할**: 복잡한 3D 형상을 사면체, 육면체 등으로 분할
2. **근사화**: 각 요소 내에서 해를 간단한 함수로 근사
3. **조립**: 모든 요소의 방정식을 연결하여 전체 방정식 구성
4. **해법**: 대규모 연립방정식을 수치적으로 해결

**형상함수 (Shape Functions)의 개념**

각 요소 내에서 온도나 변위 같은 물리량을 다음과 같이 근사합니다:

```
T(x,y,z) ≈ Σ Ni(x,y,z) × Ti
```

여기서:
- Ni: i번째 절점의 형상함수
- Ti: i번째 절점에서의 온도값

**1차 사면체 요소의 형상함수**

가장 간단한 3D 요소인 1차 사면체의 형상함수는:

```
N1 = (a1 + b1x + c1y + d1z) / (6V)
N2 = (a2 + b2x + c2y + d2z) / (6V)
N3 = (a3 + b3x + c3y + d3z) / (6V)
N4 = (a4 + b4x + c4y + d4z) / (6V)
```

여기서 V는 사면체의 체적이고, ai, bi, ci, di는 절점 좌표로부터 계산되는 계수입니다.

**SFDP에서의 FEM 구현**

```matlab
% SFDP_physics_suite.m:162에서 실제 메시 생성
fea = meshgeom(fea, 'hmax', mesh_size_far_field, 'hgrad', 1.3);

% Lines 172-173에서 절삭 영역 메시 세분화
cutting_zone_elements = find(fea.grid.p(3, fea.grid.c(1:4, :)) > workpiece_height - 2e-3);
fea = refine_mesh_elements(fea, cutting_zone_elements, 2);
```

메시 생성 전략:
- 절삭 영역: 0.2mm 크기 (온도 구배가 큰 영역)
- 원거리 영역: 1.0mm 크기 (계산 효율성 고려)
- 구배 제한: 1.3 (인접 요소 크기 비율 제한)

### 3.1.2 Assembly Process and System Matrices

**요소 행렬에서 전체 행렬로**

각 요소에서 계산된 요소 행렬들을 전체 행렬로 조립하는 과정입니다.

**열전도 문제의 요소 행렬**

각 요소 e에서의 열전도 방정식:

```
[Ke]{Te} + [Ce]{dTe/dt} = {fe}
```

여기서:
- [Ke]: 요소 전도 행렬 (4×4, 사면체의 경우)
- [Ce]: 요소 열용량 행렬 (4×4)
- {fe}: 요소 열생성 벡터 (4×1)

**전도 행렬 계산**

```
Ke_ij = ∫∫∫ k(∇Ni·∇Nj) dV
```

이 적분을 수치적으로 계산하려면 Gauss 적분을 사용합니다:

```matlab
% 가우스 적분점과 가중치 (사면체용)
gauss_points = [0.25, 0.25, 0.25, 0.25;    % ξ 좌표
                0.25, 0.25, 0.25, 0.25;    % η 좌표  
                0.25, 0.25, 0.25, 0.25];   % ζ 좌표
weights = [1/24, 1/24, 1/24, 1/24];        % 가중치

% 요소 행렬 계산
for gp = 1:4  % 4개 적분점
    [N, dN] = shape_functions(gauss_points(:,gp));
    J = jacobian_matrix(element_coords, dN);
    B = inv(J) * dN;  % 형상함수 기울기
    
    Ke = Ke + weights(gp) * B' * k * B * det(J);
    Ce = Ce + weights(gp) * rho_cp * N' * N * det(J);
end
```

**전체 행렬 조립**

```matlab
% 전체 행렬 초기화
global_K = sparse(total_nodes, total_nodes);
global_C = sparse(total_nodes, total_nodes);

% 각 요소에서 전체 행렬로 기여
for e = 1:num_elements
    element_nodes = connectivity(e,:);  % 요소의 절점 번호
    
    % 요소 행렬을 전체 행렬에 더함
    global_K(element_nodes, element_nodes) = ...
        global_K(element_nodes, element_nodes) + Ke;
    global_C(element_nodes, element_nodes) = ...
        global_C(element_nodes, element_nodes) + Ce;
end
```

### 3.1.3 Time Integration: Implicit vs Explicit Methods

**시간 적분의 필요성**

가공 문제는 시간에 따라 변하는 현상입니다:
- 도구가 움직임에 따른 온도 변화
- 마모가 진행되는 과정
- 표면 거칠기의 점진적 변화

따라서 시간 영역에서도 수치 적분이 필요합니다.

**Explicit 방법 (Forward Euler)**

현재 시점의 정보만으로 다음 시점을 계산:

```
[C]{T^(n+1)} = ([C] - Δt[K]){T^n} + Δt{f^n}
```

장점:
- 계산이 간단하고 빠름
- 각 시간 단계에서 연립방정식을 풀 필요 없음

단점:
- 시간 간격이 너무 크면 불안정
- 안정성 조건: Δt < 2/(최대 고유값)

**Implicit 방법 (Backward Euler)**

다음 시점의 미지수로 방정식을 구성:

```
([C] + Δt[K]){T^(n+1)} = [C]{T^n} + Δt{f^(n+1)}
```

장점:
- 무조건 안정 (Unconditionally Stable)
- 큰 시간 간격 사용 가능

단점:
- 매 시간 단계마다 연립방정식을 풀어야 함
- 계산량이 많음

**SFDP에서의 시간 적분 구현**

```matlab
% FEATool에서는 자동으로 Implicit 방법 사용
% 시간 스텝 설정
time_steps = 0:0.1:total_time;  % 0.1초 간격

% 각 시간 단계에서 해 계산
for t_step = 2:length(time_steps)
    current_time = time_steps(t_step);
    
    % 움직이는 열원 위치 업데이트
    cutting_position = cutting_speed * current_time / 60;  % m/min to m/s
    
    % 열원 위치에 따른 우변 벡터 업데이트
    heat_source_vector = update_heat_source(cutting_position);
    
    % 연립방정식 해결: ([C] + Δt[K]){T} = {RHS}
    temperature_field = solve_linear_system(system_matrix, rhs_vector);
end
```

시간 간격 선택 기준:
- 열확산 시간: Δt ≈ h²/(6α) (여기서 h는 요소 크기, α는 열확산계수)
- Ti-6Al-4V의 경우: α ≈ 3×10⁻⁶ m²/s
- 0.2mm 요소: Δt ≈ 0.007초 권장

## 3.2 Kalman Filtering Theory

### 3.2.1 State Space Representation: x(k+1) = A·x(k) + B·u(k) + w(k)

**칼먼 필터가 왜 필요한가?**

가공 시뮬레이션에서는 여러 정보원이 있습니다:
1. **물리학 모델**: 이론적으로 정확하지만 모든 현상을 다 포함하지 못함
2. **실험 데이터**: 실제 현상을 반영하지만 노이즈가 있고 제한적
3. **경험적 모델**: 실용적이지만 적용 범위가 제한적

칼먼 필터는 이런 여러 정보를 **최적으로 결합**하는 방법입니다.

**상태공간 모델의 기본 개념**

시스템의 상태를 벡터로 표현하고, 시간에 따른 변화를 수학적으로 모델링합니다.

**SFDP의 상태 벡터 정의**

```matlab
% 상태 벡터: [온도, 도구마모, 표면거칠기, 절삭력, 압력]
x = [T; VB; Ra; F; P];
```

각 상태의 물리적 의미:
- T: 절삭 영역 평균 온도 [°C]
- VB: 도구 마모량 [mm]
- Ra: 표면 거칠기 [μm]
- F: 절삭력 [N]
- P: 접촉 압력 [MPa]

**상태 전이 모델 (Process Model)**

```
x(k+1) = A·x(k) + B·u(k) + w(k)
```

여기서:
- A: 상태 전이 행렬 (5×5)
- B: 입력 행렬 (5×3)
- u(k): 입력 벡터 [절삭속도, 이송속도, 절삭깊이]
- w(k): 프로세스 노이즈 (모델 불확실성)

**SFDP에서의 상태 전이 행렬 구성**

```matlab
% calculateKalmanFusion 함수에서 (Lines 1-120)
function A = construct_state_transition_matrix(dt, cutting_conditions)
    % dt: 시간 간격
    % 상태 간의 물리적 연관성을 반영
    
    A = eye(5);  % 기본적으로 단위행렬
    
    % 온도와 마모의 관계 (온도가 높으면 마모 가속)
    A(2,1) = dt * wear_temperature_coupling;
    
    % 마모와 표면거칠기의 관계 (마모가 진행되면 거칠기 증가)
    A(3,2) = dt * wear_roughness_coupling;
    
    % 온도와 절삭력의 관계 (온도가 높으면 재료가 연화되어 힘 감소)
    A(4,1) = -dt * thermal_softening_effect;
    
    % 시간에 따른 자연적 변화 (관성 효과)
    A(1,1) = exp(-dt/thermal_time_constant);
    A(4,4) = exp(-dt/force_time_constant);
end
```

### 3.2.2 Prediction and Update Steps

**칼먼 필터의 두 단계**

칼먼 필터는 **예측(Prediction)**과 **업데이트(Update)** 두 단계를 반복합니다.

**1단계: 예측 (Prediction)**

이전 상태를 바탕으로 현재 상태를 예측:

```
x̂⁻(k) = A·x̂(k-1) + B·u(k)      % 상태 예측
P⁻(k) = A·P(k-1)·Aᵀ + Q         % 오차 공분산 예측
```

여기서:
- x̂⁻(k): 예측된 상태
- P⁻(k): 예측 오차 공분산 행렬
- Q: 프로세스 노이즈 공분산

**2단계: 업데이트 (Update)**

측정값을 이용해서 예측을 보정:

```
K(k) = P⁻(k)·Hᵀ·(H·P⁻(k)·Hᵀ + R)⁻¹   % 칼먼 이득
x̂(k) = x̂⁻(k) + K(k)·(z(k) - H·x̂⁻(k))  % 상태 업데이트
P(k) = (I - K(k)·H)·P⁻(k)             % 오차 공분산 업데이트
```

여기서:
- K(k): 칼먼 이득 (물리모델과 측정값의 가중치)
- z(k): 측정값 벡터
- H: 측정 행렬 (상태를 측정값으로 변환)
- R: 측정 노이즈 공분산

**SFDP에서의 실제 구현**

```matlab
% SFDP_kalman_fusion_suite.m에서 실제 칼먼 필터 구현
function [state_updated, covariance_updated] = kalman_prediction_update(state_prev, ...
    covariance_prev, measurement, process_noise, measurement_noise)

% 1단계: 예측
state_predicted = A * state_prev + B * input_vector;
covariance_predicted = A * covariance_prev * A' + process_noise;

% 2단계: 업데이트
innovation = measurement - H * state_predicted;  % 혁신 (예측 오차)
innovation_covariance = H * covariance_predicted * H' + measurement_noise;
kalman_gain = covariance_predicted * H' / innovation_covariance;

% 최종 업데이트
state_updated = state_predicted + kalman_gain * innovation;
covariance_updated = (eye(5) - kalman_gain * H) * covariance_predicted;
```

**측정 모델 설계**

```matlab
% 측정값과 상태의 관계
% z = [T_measured; VB_measured; Ra_measured]
H = [1, 0, 0, 0, 0;    % 온도 측정
     0, 1, 0, 0, 0;    % 마모 측정  
     0, 0, 1, 0, 0];   % 거칠기 측정
```

### 3.2.3 Extended and Unscented Kalman Filters

**비선형성의 문제**

가공 현상은 본질적으로 비선형입니다:
- 온도와 마모의 관계: 지수함수적
- 마모와 거칠기의 관계: 비선형
- 열전도계수의 온도 의존성

따라서 선형 칼먼 필터로는 한계가 있습니다.

**확장 칼먼 필터 (Extended Kalman Filter, EKF)**

비선형 함수를 1차 Taylor 전개로 선형화:

```
x(k+1) = f(x(k), u(k)) + w(k)      % 비선형 상태방정식
z(k) = h(x(k)) + v(k)              % 비선형 측정방정식
```

Jacobian 행렬 계산:
```
F = ∂f/∂x |_(x̂(k),u(k))           % 상태 Jacobian
H = ∂h/∂x |_(x̂⁻(k))               % 측정 Jacobian
```

**SFDP에서의 EKF 구현**

```matlab
% calculateExtendedKalman 함수에서 비선형 모델 처리
function [F_jacobian] = calculate_state_jacobian(state, input)
    T = state(1); VB = state(2); Ra = state(3);
    
    F_jacobian = eye(5);
    
    % ∂(마모)/∂(온도): 아레니우스 관계
    F_jacobian(2,1) = wear_coefficient * exp(-activation_energy/(R*T)) * ...
                      activation_energy/(R*T^2);
    
    % ∂(거칠기)/∂(마모): 제곱근 관계
    F_jacobian(3,2) = 0.5 / sqrt(VB + small_number);
    
    % ∂(절삭력)/∂(온도): 열연화 효과
    F_jacobian(4,1) = -thermal_softening_coefficient * exp(-T/softening_temperature);
end
```

**무향 칼먼 필터 (Unscented Kalman Filter, UKF)**

선형화 대신 **sigma point**를 사용하여 비선형성을 더 정확히 처리:

```matlab
% calculateUnscentedKalman 함수에서 sigma point 생성
function sigma_points = generate_sigma_points(mean_state, covariance, alpha, beta, kappa)
    n = length(mean_state);
    lambda = alpha^2 * (n + kappa) - n;
    
    % Cholesky 분해로 제곱근 행렬 계산
    sqrt_matrix = chol((n + lambda) * covariance)';
    
    % Sigma points 생성
    sigma_points(:,1) = mean_state;  % 중심점
    
    for i = 1:n
        sigma_points(:,i+1) = mean_state + sqrt_matrix(:,i);      % +방향
        sigma_points(:,i+1+n) = mean_state - sqrt_matrix(:,i);    % -방향
    end
end
```

UKF의 장점:
- 2차 정확도 (EKF는 1차)
- Jacobian 계산 불필요
- 강한 비선형성에서도 안정적

## 3.3 Machine Learning Integration in Physics-Based Models

### 3.3.1 Supervised Learning: Regression Trees, SVMs, Neural Networks

**물리학 모델의 한계와 ML의 역할**

아무리 정교한 물리학 모델도 모든 현상을 다 포함할 수는 없습니다:
- 복잡한 재료 거동
- 미세한 기계 진동
- 환경 조건의 영향
- 도구 마모의 복잡성

기계학습은 이런 **모델 오차**를 데이터 기반으로 보정하는 역할을 합니다.

**Random Forest 회귀의 구현**

```matlab
% SFDP_empirical_ml_suite.m의 calculateEmpiricalML 함수에서
function [ml_prediction, prediction_confidence] = random_forest_regression(features, target_data)
    
    % 여러 개의 결정트리 학습
    num_trees = 100;
    predictions = zeros(size(features,1), num_trees);
    
    for tree = 1:num_trees
        % 부트스트랩 샘플링
        bootstrap_idx = randsample(size(target_data,1), size(target_data,1), true);
        train_features = features(bootstrap_idx,:);
        train_targets = target_data(bootstrap_idx);
        
        % 결정트리 학습
        tree_model = fitrtree(train_features, train_targets, 'MaxNumSplits', 50);
        
        % 예측
        predictions(:,tree) = predict(tree_model, features);
    end
    
    % 앙상블 예측 (평균)
    ml_prediction = mean(predictions, 2);
    
    % 예측 신뢰도 (표준편차 기반)
    prediction_confidence = 1 ./ (1 + std(predictions, 0, 2));
end
```

**Support Vector Machine (SVM) 구현**

고차원에서 비선형 관계를 학습:

```matlab
% SVM with RBF kernel for nonlinear regression
function svm_model = train_svm_regression(features, targets)
    % 특성 정규화
    [features_normalized, norm_params] = normalize_features(features);
    
    % SVM 학습 (RBF 커널)
    svm_model = fitrsvm(features_normalized, targets, ...
        'KernelFunction', 'rbf', 'Epsilon', 0.01, 'Standardize', true);
    
    % 정규화 파라미터 저장
    svm_model.normalization_params = norm_params;
end
```

**Neural Network 구현**

```matlab
% 간단한 feedforward 신경망
function nn_model = train_neural_network(features, targets, hidden_layers)
    % 네트워크 구조 정의
    net = feedforwardnet(hidden_layers);
    
    % 학습 파라미터 설정
    net.trainParam.epochs = 1000;
    net.trainParam.lr = 0.01;
    net.trainParam.goal = 1e-6;
    
    % 데이터 분할 (학습:검증:테스트 = 70:15:15)
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % 신경망 학습
    [nn_model, train_record] = train(net, features', targets');
end
```

### 3.3.2 Feature Engineering for Physics-Informed ML

**물리학 기반 특성 추출**

단순히 원시 데이터를 넣는 것이 아니라, 물리학적 의미가 있는 특성을 만들어야 합니다.

**온도 관련 특성**

```matlab
% performFeatureEngineering 함수에서 물리 기반 특성 생성
function physics_features = create_thermal_features(cutting_conditions, material_props)
    
    % 기본 특성
    cutting_speed = cutting_conditions.speed;
    feed_rate = cutting_conditions.feed;
    depth_of_cut = cutting_conditions.depth;
    
    % 물리학 기반 특성들
    features = [];
    
    % 1. Peclet 수: 대류/확산 비율
    peclet_number = cutting_speed * depth_of_cut / material_props.thermal_diffusivity;
    features = [features, peclet_number];
    
    % 2. 비절삭에너지
    specific_cutting_energy = calculate_specific_energy(cutting_conditions);
    features = [features, specific_cutting_energy];
    
    % 3. 열발생률
    heat_generation_rate = specific_cutting_energy * cutting_speed * feed_rate * depth_of_cut;
    features = [features, heat_generation_rate];
    
    % 4. 무차원 온도
    dimensionless_temp = heat_generation_rate / (material_props.conductivity * cutting_speed);
    features = [features, dimensionless_temp];
    
    physics_features = features;
end
```

**마모 관련 특성**

```matlab
function wear_features = create_wear_features(cutting_conditions, temperature_field)
    
    features = [];
    
    % 1. 아레니우스 인자: exp(-Q/RT)
    activation_energy = 50000;  % J/mol (확산 활성화 에너지)
    gas_constant = 8.314;
    max_temp = max(temperature_field.temperature);
    arrhenius_factor = exp(-activation_energy / (gas_constant * (max_temp + 273.15)));
    features = [features, arrhenius_factor];
    
    % 2. 누적 슬라이딩 거리
    sliding_distance = cutting_conditions.speed * cutting_conditions.time / 60;
    features = [features, sliding_distance];
    
    % 3. 접촉 압력 추정
    contact_pressure = estimate_contact_pressure(cutting_conditions);
    features = [features, contact_pressure];
    
    % 4. 마모 수명 예측 (Taylor 기반)
    taylor_life = (cutting_conditions.taylor_constant / cutting_conditions.speed)^(1/cutting_conditions.taylor_exponent);
    wear_rate_indicator = cutting_conditions.time / taylor_life;
    features = [features, wear_rate_indicator];
    
    wear_features = features;
end
```

### 3.3.3 Uncertainty Quantification and Model Confidence

**불확실성이 왜 중요한가?**

가공 시뮬레이션에서는 **결과의 신뢰도**를 아는 것이 중요합니다:
- 예측이 정확한지 판단
- 추가 실험의 필요성 결정
- 안전 여유 설정

**앙상블 기반 불확실성 정량화**

```matlab
% 여러 모델의 예측을 종합하여 불확실성 계산
function [prediction, uncertainty] = ensemble_prediction_with_uncertainty(models, features)
    
    num_models = length(models);
    predictions = zeros(size(features,1), num_models);
    
    % 각 모델에서 예측
    for i = 1:num_models
        predictions(:,i) = predict(models{i}, features);
    end
    
    % 평균 예측
    prediction = mean(predictions, 2);
    
    % 불확실성 추정 (표준편차)
    uncertainty = std(predictions, 0, 2);
    
    % 신뢰구간 계산 (95%)
    confidence_interval = 1.96 * uncertainty;
    
    % 신뢰도 점수 (0-1)
    confidence_score = 1 ./ (1 + uncertainty/abs(prediction));
end
```

**Bootstrap 기반 불확실성**

```matlab
function [prediction_mean, prediction_std] = bootstrap_uncertainty(model, features, num_bootstrap)
    
    bootstrap_predictions = zeros(size(features,1), num_bootstrap);
    
    for i = 1:num_bootstrap
        % 부트스트랩 샘플로 모델 재학습
        bootstrap_model = retrain_with_bootstrap(model);
        
        % 예측
        bootstrap_predictions(:,i) = predict(bootstrap_model, features);
    end
    
    % 통계량 계산
    prediction_mean = mean(bootstrap_predictions, 2);
    prediction_std = std(bootstrap_predictions, 0, 2);
end
```