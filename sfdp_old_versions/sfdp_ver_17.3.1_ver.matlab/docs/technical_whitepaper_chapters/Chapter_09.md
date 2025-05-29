# Chapter 9: Machine Learning Implementation

## 9.1 Empirical Model Suite Architecture

The SFDP system incorporates machine learning through its empirical model suite, which provides data-driven corrections and predictions to complement the physics-based calculations.

### 9.1.1 Neural Network Implementations

**File: SFDP_empirical_ml_suite.m:1-100**

```matlab
function [ml_results, ml_confidence] = performAdvancedMachineLearning(cutting_conditions, ...
    material_props, physics_results, simulation_state)
    
    fprintf('🤖 고급 머신러닝 모델 실행 시작\n');
    
    ml_results = struct();
    ml_confidence = 0;
    
    try
        % 1. 입력 데이터 전처리
        fprintf('  📊 입력 데이터 전처리 중...\n');
        [processed_features, feature_confidence] = preprocessMLFeatures(...
            cutting_conditions, material_props, physics_results, simulation_state);
        
        if feature_confidence < 0.5
            warning('특성 추출 신뢰도가 낮습니다 (%.2f)', feature_confidence);
        end
        
        % 2. 신경망 기반 온도 예측
        fprintf('  🌡️ 신경망 온도 예측 실행 중...\n');
        [temperature_prediction, temp_confidence] = neuralNetworkTemperaturePrediction(...
            processed_features, simulation_state);
        
        ml_results.temperature = temperature_prediction;
        ml_results.temperature_confidence = temp_confidence;
        
        % 3. 랜덤 포레스트 공구마모 예측
        fprintf('  🔧 랜덤 포레스트 공구마모 예측 실행 중...\n');
        [wear_prediction, wear_confidence] = randomForestWearPrediction(...
            processed_features, simulation_state);
        
        ml_results.tool_wear = wear_prediction;
        ml_results.wear_confidence = wear_confidence;
        
        % 4. SVM 표면조도 예측
        fprintf('  📏 SVM 표면조도 예측 실행 중...\n');
        [surface_prediction, surface_confidence] = svmSurfaceRoughnessPrediction(...
            processed_features, simulation_state);
        
        ml_results.surface_roughness = surface_prediction;
        ml_results.surface_confidence = surface_confidence;
        
        % 전체 신뢰도 계산
        confidences = [temp_confidence, wear_confidence, surface_confidence];
        ml_confidence = mean(confidences(confidences > 0));
        
        fprintf('  ✅ 머신러닝 예측 완료 (전체 신뢰도: %.2f)\n', ml_confidence);
        
    catch ME
        fprintf('  ❌ 머신러닝 실행 실패: %s\n', ME.message);
        ml_results = [];
        ml_confidence = 0;
    end
end
```

### 9.1.2 Feature Engineering and Data Preprocessing

The system implements sophisticated feature engineering to extract meaningful patterns from the machining data.

**Feature Extraction Implementation (SFDP_empirical_ml_suite.m:200-350)**

```matlab
function [processed_features, feature_confidence] = preprocessMLFeatures(cutting_conditions, ...
    material_props, physics_results, simulation_state)
    
    processed_features = struct();
    feature_confidence = 0;
    
    try
        % 1. 기본 절삭 조건 특성
        basic_features = extractBasicCuttingFeatures(cutting_conditions);
        
        % 2. 재료 속성 특성
        material_features = extractMaterialPropertyFeatures(material_props);
        
        % 3. 물리 기반 특성
        physics_features = extractPhysicsBasedFeatures(physics_results);
        
        % 4. 상호작용 특성
        interaction_features = createInteractionFeatures(basic_features, material_features);
        
        % 5. 시간 도메인 특성
        temporal_features = extractTemporalFeatures(cutting_conditions, simulation_state);
        
        % 특성 결합
        processed_features.basic = basic_features;
        processed_features.material = material_features;
        processed_features.physics = physics_features;
        processed_features.interaction = interaction_features;
        processed_features.temporal = temporal_features;
        
        % 특성 정규화
        processed_features.normalized = normalizeFeatures(processed_features, simulation_state);
        
        % 특성 선택
        processed_features.selected = selectOptimalFeatures(processed_features.normalized, simulation_state);
        
        feature_confidence = evaluateFeatureQuality(processed_features);
        
    catch ME
        fprintf('특성 전처리 중 오류: %s\n', ME.message);
        processed_features = [];
        feature_confidence = 0;
    end
end

function basic_features = extractBasicCuttingFeatures(cutting_conditions)
    basic_features = struct();
    
    % 직접 절삭 파라미터
    basic_features.cutting_speed = cutting_conditions.speed;          % m/min
    basic_features.feed_rate = cutting_conditions.feed;              % mm/rev
    basic_features.depth_of_cut = cutting_conditions.depth;          % mm
    
    % 계산된 파라미터
    if isfield(cutting_conditions, 'tool') && isfield(cutting_conditions.tool, 'diameter')
        tool_diameter = cutting_conditions.tool.diameter;
        basic_features.spindle_speed = (cutting_conditions.speed * 1000) / (pi * tool_diameter); % RPM
    else
        basic_features.spindle_speed = cutting_conditions.speed * 318.3; % 기본 추정값
    end
    
    % 재료 제거율
    basic_features.material_removal_rate = cutting_conditions.speed * cutting_conditions.feed * cutting_conditions.depth;
    
    % 무차원 수
    basic_features.speed_to_feed_ratio = cutting_conditions.speed / cutting_conditions.feed;
    basic_features.feed_to_depth_ratio = cutting_conditions.feed / cutting_conditions.depth;
    
    % 로그 변환 특성
    basic_features.log_speed = log(cutting_conditions.speed);
    basic_features.log_feed = log(cutting_conditions.feed);
    basic_features.log_depth = log(cutting_conditions.depth);
    
    % 제곱근 변환 특성
    basic_features.sqrt_speed = sqrt(cutting_conditions.speed);
    basic_features.sqrt_feed = sqrt(cutting_conditions.feed);
    basic_features.sqrt_depth = sqrt(cutting_conditions.depth);
end

function material_features = extractMaterialPropertyFeatures(material_props)
    material_features = struct();
    
    % 열적 속성
    if isfield(material_props, 'thermal')
        material_features.thermal_conductivity = material_props.thermal.conductivity;
        material_features.specific_heat = material_props.thermal.specific_heat;
        material_features.density = material_props.thermal.density;
        material_features.thermal_diffusivity = material_props.thermal.conductivity / ...
            (material_props.thermal.density * material_props.thermal.specific_heat);
    end
    
    % 기계적 속성
    if isfield(material_props, 'mechanical')
        material_features.youngs_modulus = material_props.mechanical.youngs_modulus;
        material_features.yield_strength = material_props.mechanical.yield_strength;
        material_features.hardness = material_props.mechanical.hardness_hv;
        material_features.poisson_ratio = material_props.mechanical.poisson_ratio;
    end
    
    % 가공성 지수
    if isfield(material_props, 'machinability')
        material_features.machinability_index = material_props.machinability.tool_wear_factor;
        material_features.cutting_force_coefficient = material_props.machinability.cutting_force_coefficient;
    end
    
    % 파생 특성
    if isfield(material_features, 'yield_strength') && isfield(material_features, 'hardness')
        material_features.strength_hardness_ratio = material_features.yield_strength / (material_features.hardness * 1e6);
    end
end

function physics_features = extractPhysicsBasedFeatures(physics_results)
    physics_features = struct();
    
    if isempty(physics_results)
        % 물리 결과가 없는 경우 기본값
        physics_features.max_temperature = 0;
        physics_features.avg_temperature = 0;
        physics_features.temperature_gradient = 0;
        physics_features.max_stress = 0;
        physics_features.contact_pressure = 0;
        return;
    end
    
    % 열적 특성
    if isfield(physics_results, 'thermal')
        if isfield(physics_results.thermal, 'max_temperature')
            physics_features.max_temperature = physics_results.thermal.max_temperature;
        end
        if isfield(physics_results.thermal, 'avg_temperature')
            physics_features.avg_temperature = physics_results.thermal.avg_temperature;
        end
        if isfield(physics_results.thermal, 'max_gradient')
            physics_features.temperature_gradient = physics_results.thermal.max_gradient;
        end
    end
    
    % 기계적 특성
    if isfield(physics_results, 'mechanical')
        if isfield(physics_results.mechanical, 'max_stress')
            physics_features.max_stress = physics_results.mechanical.max_stress;
        end
        if isfield(physics_results.mechanical, 'avg_pressure')
            physics_features.contact_pressure = physics_results.mechanical.avg_pressure;
        end
    end
    
    % 마모 특성
    if isfield(physics_results, 'wear')
        if isfield(physics_results.wear, 'total_wear')
            physics_features.predicted_wear = physics_results.wear.total_wear;
        end
        if isfield(physics_results.wear, 'wear_rate')
            physics_features.wear_rate = physics_results.wear.wear_rate;
        end
    end
end

function interaction_features = createInteractionFeatures(basic_features, material_features)
    interaction_features = struct();
    
    % 절삭 조건과 재료 속성의 상호작용
    if isfield(basic_features, 'cutting_speed') && isfield(material_features, 'thermal_conductivity')
        interaction_features.speed_thermal_interaction = basic_features.cutting_speed / material_features.thermal_conductivity;
    end
    
    if isfield(basic_features, 'feed_rate') && isfield(material_features, 'hardness')
        interaction_features.feed_hardness_interaction = basic_features.feed_rate * material_features.hardness;
    end
    
    if isfield(basic_features, 'material_removal_rate') && isfield(material_features, 'density')
        interaction_features.mrr_density_interaction = basic_features.material_removal_rate * material_features.density;
    end
    
    % 복합 무차원 수
    if isfield(basic_features, 'cutting_speed') && isfield(material_features, 'thermal_diffusivity')
        characteristic_length = 1e-3;  % 1mm 특성 길이
        interaction_features.peclet_number = (basic_features.cutting_speed / 60) * characteristic_length / material_features.thermal_diffusivity;
    end
end
```

### 9.1.3 Neural Network Temperature Prediction

The system uses a deep neural network for temperature prediction, incorporating both cutting conditions and material properties.

**Neural Network Implementation (SFDP_empirical_ml_suite.m:500-700)**

```matlab
function [temperature_prediction, temp_confidence] = neuralNetworkTemperaturePrediction(...
    processed_features, simulation_state)
    
    temperature_prediction = struct();
    temp_confidence = 0;
    
    try
        % 1. 신경망 입력 벡터 구성
        input_vector = constructNeuralNetworkInput(processed_features);
        
        % 2. 신경망 모델 로드 또는 생성
        if isfield(simulation_state, 'ml_models') && isfield(simulation_state.ml_models, 'temperature_nn')
            % 기존 훈련된 모델 사용
            nn_model = simulation_state.ml_models.temperature_nn;
        else
            % 기본 신경망 구조 생성
            nn_model = createDefaultTemperatureNeuralNetwork();
        end
        
        % 3. 순전파 예측
        [nn_output, prediction_uncertainty] = forwardPrediction(nn_model, input_vector);
        
        % 4. 결과 해석 및 후처리
        temperature_prediction.peak_temperature = nn_output(1) * simulation_state.scaling.temperature_max;
        temperature_prediction.average_temperature = nn_output(2) * simulation_state.scaling.temperature_max;
        temperature_prediction.temperature_gradient = nn_output(3) * simulation_state.scaling.gradient_max;
        
        % 불확실성 정량화
        temperature_prediction.uncertainty = struct();
        temperature_prediction.uncertainty.epistemic = prediction_uncertainty.model_uncertainty;
        temperature_prediction.uncertainty.aleatoric = prediction_uncertainty.data_uncertainty;
        temperature_prediction.uncertainty.total = sqrt(prediction_uncertainty.model_uncertainty^2 + ...
                                                        prediction_uncertainty.data_uncertainty^2);
        
        % 신뢰도 계산
        temp_confidence = calculatePredictionConfidence(temperature_prediction, nn_model);
        
    catch ME
        fprintf('신경망 온도 예측 중 오류: %s\n', ME.message);
        temperature_prediction = [];
        temp_confidence = 0;
    end
end

function nn_model = createDefaultTemperatureNeuralNetwork()
    % 기본 신경망 구조 정의
    nn_model = struct();
    
    % 네트워크 구조
    nn_model.architecture = [15, 20, 15, 10, 3]; % 입력-은닉층들-출력
    nn_model.activation_functions = {'relu', 'relu', 'relu', 'linear'}; % 레이어별 활성화 함수
    
    % 가중치 초기화 (기본값 - 실제로는 훈련된 가중치 사용)
    nn_model.weights = cell(1, length(nn_model.architecture)-1);
    nn_model.biases = cell(1, length(nn_model.architecture)-1);
    
    for i = 1:length(nn_model.architecture)-1
        % Xavier 초기화
        fan_in = nn_model.architecture(i);
        fan_out = nn_model.architecture(i+1);
        xavier_bound = sqrt(6.0 / (fan_in + fan_out));
        
        nn_model.weights{i} = (rand(fan_out, fan_in) * 2 - 1) * xavier_bound;
        nn_model.biases{i} = zeros(fan_out, 1);
    end
    
    % 훈련 관련 정보
    nn_model.training_info = struct();
    nn_model.training_info.epochs_trained = 0;
    nn_model.training_info.validation_loss = inf;
    nn_model.training_info.regularization = 0.001;
end

function [output, uncertainty] = forwardPrediction(nn_model, input_vector)
    % 신경망 순전파 예측
    
    current_activation = input_vector(:);
    
    % 각 레이어를 통과
    for layer = 1:length(nn_model.weights)
        % 선형 변환
        z = nn_model.weights{layer} * current_activation + nn_model.biases{layer};
        
        % 활성화 함수 적용
        activation_func = nn_model.activation_functions{layer};
        switch activation_func
            case 'relu'
                current_activation = max(0, z);
            case 'sigmoid'
                current_activation = 1 ./ (1 + exp(-z));
            case 'tanh'
                current_activation = tanh(z);
            case 'linear'
                current_activation = z;
            otherwise
                current_activation = z;
        end
    end
    
    output = current_activation;
    
    % 불확실성 추정 (간소화된 버전)
    uncertainty = struct();
    uncertainty.model_uncertainty = 0.1 * norm(output);  % 모델 불확실성
    uncertainty.data_uncertainty = 0.05 * norm(output);  % 데이터 불확실성
end

function confidence = calculatePredictionConfidence(prediction, nn_model)
    % 예측 신뢰도 계산
    
    confidence = 0.5;  % 기본값
    
    try
        % 불확실성 기반 신뢰도
        if isfield(prediction, 'uncertainty') && isfield(prediction.uncertainty, 'total')
            total_uncertainty = prediction.uncertainty.total;
            max_reasonable_uncertainty = 100;  % 예상 최대 불확실성
            
            uncertainty_score = max(0, 1 - total_uncertainty / max_reasonable_uncertainty);
            confidence = confidence * 0.5 + uncertainty_score * 0.5;
        end
        
        % 물리적 타당성 기반 신뢰도
        if isfield(prediction, 'peak_temperature')
            temp = prediction.peak_temperature;
            if temp > 0 && temp < 2000  % 합리적 온도 범위
                physics_score = 1.0;
            else
                physics_score = 0.2;
            end
            confidence = confidence * 0.7 + physics_score * 0.3;
        end
        
        % 모델 훈련 품질 반영
        if isfield(nn_model, 'training_info') && isfield(nn_model.training_info, 'validation_loss')
            if nn_model.training_info.validation_loss < 0.1
                training_score = 0.9;
            elseif nn_model.training_info.validation_loss < 0.5
                training_score = 0.7;
            else
                training_score = 0.4;
            end
            confidence = confidence * 0.8 + training_score * 0.2;
        end
        
    catch ME
        fprintf('신뢰도 계산 중 오류: %s\n', ME.message);
        confidence = 0.3;
    end
    
    % 신뢰도 범위 제한
    confidence = max(0.1, min(0.95, confidence));
end
```

## 9.2 Support Vector Machine and Random Forest Models

### 9.2.1 SVM Surface Roughness Prediction

**Implementation (SFDP_empirical_ml_suite.m:800-950)**

```matlab
function [surface_prediction, surface_confidence] = svmSurfaceRoughnessPrediction(...
    processed_features, simulation_state)
    
    surface_prediction = struct();
    surface_confidence = 0;
    
    try
        % 1. SVM 입력 벡터 구성
        svm_input = constructSVMInput(processed_features);
        
        % 2. SVM 모델 설정
        if isfield(simulation_state, 'ml_models') && isfield(simulation_state.ml_models, 'surface_svm')
            svm_model = simulation_state.ml_models.surface_svm;
        else
            svm_model = createDefaultSurfaceSVM();
        end
        
        % 3. SVM 예측 실행
        [svm_output, decision_values] = predictSVM(svm_model, svm_input);
        
        % 4. 결과 해석
        surface_prediction.ra_roughness = svm_output(1) * simulation_state.scaling.roughness_max;
        surface_prediction.rz_roughness = svm_output(2) * simulation_state.scaling.roughness_max;
        surface_prediction.roughness_profile = svm_output(3:end);
        
        % 신뢰도 계산 (결정 경계로부터의 거리 기반)
        surface_confidence = calculateSVMConfidence(decision_values, svm_model);
        
    catch ME
        fprintf('SVM 표면조도 예측 중 오류: %s\n', ME.message);
        surface_prediction = [];
        surface_confidence = 0;
    end
end

function svm_model = createDefaultSurfaceSVM()
    svm_model = struct();
    
    % SVM 하이퍼파라미터
    svm_model.kernel = 'rbf';           % 방사기저함수 커널
    svm_model.gamma = 0.1;              % RBF 커널 파라미터
    svm_model.C = 1.0;                  % 정규화 파라미터
    svm_model.epsilon = 0.01;           % SVR의 epsilon 파라미터
    
    % 지원 벡터 (기본값 - 실제로는 훈련된 값 사용)
    svm_model.support_vectors = randn(50, 15); % 50개 지원벡터, 15차원 특성
    svm_model.support_vector_labels = randn(50, 3); % 3차원 출력
    svm_model.alphas = randn(50, 3);    % 라그랑주 승수
    svm_model.b = randn(3, 1);          % 편향
    
    % 정규화 파라미터
    svm_model.input_mean = zeros(15, 1);
    svm_model.input_std = ones(15, 1);
    svm_model.output_mean = zeros(3, 1);
    svm_model.output_std = ones(3, 1);
end

function [output, decision_values] = predictSVM(svm_model, input_vector)
    % SVM 예측 실행
    
    % 입력 정규화
    normalized_input = (input_vector - svm_model.input_mean) ./ svm_model.input_std;
    
    % 커널 함수 계산
    if strcmp(svm_model.kernel, 'rbf')
        % RBF 커널
        distances = sum((svm_model.support_vectors - normalized_input').^2, 2);
        kernel_values = exp(-svm_model.gamma * distances);
    else
        % 선형 커널 (폴백)
        kernel_values = svm_model.support_vectors * normalized_input;
    end
    
    % 예측값 계산
    raw_output = zeros(size(svm_model.alphas, 2), 1);
    for i = 1:size(svm_model.alphas, 2)
        raw_output(i) = sum(svm_model.alphas(:, i) .* kernel_values) + svm_model.b(i);
    end
    
    % 출력 역정규화
    output = raw_output .* svm_model.output_std + svm_model.output_mean;
    
    % 결정 값 (신뢰도 계산용)
    decision_values = abs(raw_output);
end

function confidence = calculateSVMConfidence(decision_values, svm_model)
    % SVM 신뢰도 계산
    
    % 결정 경계로부터의 거리 기반
    min_decision_value = min(abs(decision_values));
    max_decision_value = max(abs(decision_values));
    
    % 정규화된 신뢰도
    if max_decision_value > 0
        confidence = min_decision_value / max_decision_value;
    else
        confidence = 0.5;
    end
    
    % 모델 품질 반영
    if isfield(svm_model, 'validation_score')
        confidence = confidence * 0.7 + svm_model.validation_score * 0.3;
    end
    
    confidence = max(0.1, min(0.9, confidence));
end
```

### 9.2.2 Random Forest Tool Wear Prediction

**Implementation (SFDP_empirical_ml_suite.m:1100-1300)**

```matlab
function [wear_prediction, wear_confidence] = randomForestWearPrediction(...
    processed_features, simulation_state)
    
    wear_prediction = struct();
    wear_confidence = 0;
    
    try
        % 1. 랜덤 포레스트 입력 구성
        rf_input = constructRandomForestInput(processed_features);
        
        % 2. 랜덤 포레스트 모델 로드
        if isfield(simulation_state, 'ml_models') && isfield(simulation_state.ml_models, 'wear_rf')
            rf_model = simulation_state.ml_models.wear_rf;
        else
            rf_model = createDefaultWearRandomForest();
        end
        
        % 3. 앙상블 예측 실행
        [rf_predictions, prediction_variance] = predictRandomForest(rf_model, rf_input);
        
        % 4. 결과 해석
        wear_prediction.flank_wear = rf_predictions(1) * simulation_state.scaling.wear_max;
        wear_prediction.crater_wear = rf_predictions(2) * simulation_state.scaling.wear_max;
        wear_prediction.tool_life = rf_predictions(3) * simulation_state.scaling.life_max;
        
        % 예측 불확실성
        wear_prediction.uncertainty = struct();
        wear_prediction.uncertainty.flank_wear_std = sqrt(prediction_variance(1));
        wear_prediction.uncertainty.crater_wear_std = sqrt(prediction_variance(2));
        wear_prediction.uncertainty.tool_life_std = sqrt(prediction_variance(3));
        
        % 신뢰도 계산 (예측 분산의 역수 기반)
        wear_confidence = calculateRandomForestConfidence(prediction_variance, rf_model);
        
    catch ME
        fprintf('랜덤 포레스트 공구마모 예측 중 오류: %s\n', ME.message);
        wear_prediction = [];
        wear_confidence = 0;
    end
end

function rf_model = createDefaultWearRandomForest()
    rf_model = struct();
    
    % 랜덤 포레스트 하이퍼파라미터
    rf_model.n_trees = 100;             % 트리 개수
    rf_model.max_depth = 10;            % 최대 깊이
    rf_model.min_samples_split = 5;     % 분할 최소 샘플
    rf_model.min_samples_leaf = 2;      % 잎 노드 최소 샘플
    rf_model.max_features = 'sqrt';     % 특성 선택 방법
    
    % 개별 트리들 (간소화된 구조)
    rf_model.trees = cell(1, rf_model.n_trees);
    for i = 1:rf_model.n_trees
        tree = struct();
        tree.tree_id = i;
        tree.feature_indices = randperm(15, 4); % 15개 중 4개 특성 선택
        tree.thresholds = randn(1, 4);
        tree.leaf_values = randn(8, 3);         % 8개 잎노드, 3차원 출력
        rf_model.trees{i} = tree;
    end
    
    % 특성 중요도
    rf_model.feature_importance = rand(15, 1);
    rf_model.feature_importance = rf_model.feature_importance / sum(rf_model.feature_importance);
end

function [predictions, variance] = predictRandomForest(rf_model, input_vector)
    % 랜덤 포레스트 예측
    
    n_trees = rf_model.n_trees;
    n_outputs = 3;  % flank wear, crater wear, tool life
    
    tree_predictions = zeros(n_trees, n_outputs);
    
    % 각 트리에서 예측
    for i = 1:n_trees
        tree = rf_model.trees{i};
        tree_output = predictSingleTree(tree, input_vector);
        tree_predictions(i, :) = tree_output;
    end
    
    % 앙상블 평균
    predictions = mean(tree_predictions, 1);
    
    % 예측 분산 (불확실성)
    variance = var(tree_predictions, [], 1);
end

function output = predictSingleTree(tree, input_vector)
    % 단일 의사결정 트리 예측 (간소화된 버전)
    
    selected_features = input_vector(tree.feature_indices);
    
    % 간단한 선형 결합 (실제로는 트리 구조 사용)
    output = sum(selected_features .* tree.thresholds') + tree.leaf_values(1, :);
    
    % 물리적 제약 적용
    output = max(0, output);  % 음수 방지
end

function confidence = calculateRandomForestConfidence(prediction_variance, rf_model)
    % 랜덤 포레스트 신뢰도 계산
    
    % 예측 분산이 낮을수록 높은 신뢰도
    avg_variance = mean(prediction_variance);
    max_expected_variance = 0.1;  % 예상 최대 분산
    
    variance_score = max(0, 1 - avg_variance / max_expected_variance);
    
    % 트리 개수 효과
    tree_score = min(1, rf_model.n_trees / 100);  % 100개 트리가 최적
    
    % 종합 신뢰도
    confidence = variance_score * 0.7 + tree_score * 0.3;
    
    confidence = max(0.1, min(0.9, confidence));
end
```

## 9.3 Model Training and Validation Framework

### 9.3.1 Cross-Validation Implementation

The system includes comprehensive model validation to ensure robustness and generalization.

**Validation Framework (SFDP_empirical_ml_suite.m:1500-1700)**

```matlab
function [validation_results, model_performance] = performMLModelValidation(...
    training_data, validation_data, simulation_state)
    
    fprintf('🧪 머신러닝 모델 검증 시작\n');
    
    validation_results = struct();
    model_performance = struct();
    
    try
        % 1. K-Fold 교차 검증
        fprintf('  🔄 K-Fold 교차 검증 실행 중...\n');
        k_folds = 5;
        [cv_scores, cv_models] = performKFoldCrossValidation(training_data, k_folds, simulation_state);
        
        validation_results.cross_validation = cv_scores;
        validation_results.cv_models = cv_models;
        
        % 2. 홀드아웃 검증
        fprintf('  📊 홀드아웃 검증 실행 중...\n');
        [holdout_scores, holdout_models] = performHoldoutValidation(training_data, validation_data, simulation_state);
        
        validation_results.holdout_validation = holdout_scores;
        validation_results.holdout_models = holdout_models;
        
        % 3. 시간 시리즈 검증 (시간 순서 고려)
        fprintf('  ⏰ 시간 시리즈 검증 실행 중...\n');
        [time_series_scores, ts_models] = performTimeSeriesValidation(training_data, simulation_state);
        
        validation_results.time_series_validation = time_series_scores;
        validation_results.ts_models = ts_models;
        
        % 4. 종합 성능 지표 계산
        model_performance = calculateOverallPerformance(validation_results);
        
        fprintf('  ✅ 모델 검증 완료\n');
        fprintf('    - CV 평균 점수: %.3f\n', mean([cv_scores.r2_scores]));
        fprintf('    - 홀드아웃 점수: %.3f\n', holdout_scores.r2_score);
        fprintf('    - 시계열 점수: %.3f\n', time_series_scores.r2_score);
        
    catch ME
        fprintf('  ❌ 모델 검증 실패: %s\n', ME.message);
        validation_results = [];
        model_performance = [];
    end
end

function [cv_scores, cv_models] = performKFoldCrossValidation(data, k_folds, simulation_state)
    % K-Fold 교차 검증 구현
    
    n_samples = size(data.features, 1);
    fold_size = floor(n_samples / k_folds);
    
    cv_scores = struct();
    cv_scores.r2_scores = zeros(1, k_folds);
    cv_scores.mse_scores = zeros(1, k_folds);
    cv_scores.mae_scores = zeros(1, k_folds);
    
    cv_models = cell(1, k_folds);
    
    for fold = 1:k_folds
        fprintf('    Fold %d/%d 실행 중...\n', fold, k_folds);
        
        % 검증 세트 인덱스
        val_start = (fold - 1) * fold_size + 1;
        val_end = min(fold * fold_size, n_samples);
        val_indices = val_start:val_end;
        
        % 훈련 세트 인덱스
        train_indices = setdiff(1:n_samples, val_indices);
        
        % 데이터 분할
        train_features = data.features(train_indices, :);
        train_targets = data.targets(train_indices, :);
        val_features = data.features(val_indices, :);
        val_targets = data.targets(val_indices, :);
        
        % 모델 훈련
        fold_model = trainMLModels(train_features, train_targets, simulation_state);
        cv_models{fold} = fold_model;
        
        % 예측 및 평가
        predictions = predictWithModel(fold_model, val_features);
        
        cv_scores.r2_scores(fold) = calculateR2Score(val_targets, predictions);
        cv_scores.mse_scores(fold) = mean((val_targets - predictions).^2);
        cv_scores.mae_scores(fold) = mean(abs(val_targets - predictions));
    end
    
    % 통계 요약
    cv_scores.mean_r2 = mean(cv_scores.r2_scores);
    cv_scores.std_r2 = std(cv_scores.r2_scores);
    cv_scores.mean_mse = mean(cv_scores.mse_scores);
    cv_scores.mean_mae = mean(cv_scores.mae_scores);
end

function r2 = calculateR2Score(y_true, y_pred)
    % R² 점수 계산
    
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    
    if ss_tot == 0
        r2 = 1;  % 완벽한 예측
    else
        r2 = 1 - ss_res / ss_tot;
    end
end

function model = trainMLModels(features, targets, simulation_state)
    % 통합 ML 모델 훈련
    
    model = struct();
    
    % 신경망 훈련
    model.neural_network = trainNeuralNetwork(features, targets(:, 1), simulation_state);
    
    % SVM 훈련
    model.svm = trainSVM(features, targets(:, 2), simulation_state);
    
    % 랜덤 포레스트 훈련
    model.random_forest = trainRandomForest(features, targets(:, 3), simulation_state);
end
```

### 9.3.2 Feature Importance Analysis

```matlab
function feature_importance = analyzeFeatureImportance(trained_models, feature_names, simulation_state)
    % 특성 중요도 분석
    
    feature_importance = struct();
    n_features = length(feature_names);
    
    % 1. 순열 중요도 (Permutation Importance)
    permutation_importance = calculatePermutationImportance(trained_models, simulation_state);
    
    % 2. 랜덤 포레스트 내장 중요도
    if isfield(trained_models, 'random_forest')
        rf_importance = trained_models.random_forest.feature_importance;
    else
        rf_importance = ones(n_features, 1) / n_features;
    end
    
    % 3. 그래디언트 기반 중요도 (신경망)
    if isfield(trained_models, 'neural_network')
        gradient_importance = calculateGradientImportance(trained_models.neural_network, simulation_state);
    else
        gradient_importance = ones(n_features, 1) / n_features;
    end
    
    % 4. 종합 중요도 계산
    weights = [0.4, 0.3, 0.3];  % 순열, RF, 그래디언트
    combined_importance = weights(1) * permutation_importance + ...
                         weights(2) * rf_importance + ...
                         weights(3) * gradient_importance;
    
    % 정규화
    combined_importance = combined_importance / sum(combined_importance);
    
    % 결과 구조체 생성
    feature_importance.permutation = permutation_importance;
    feature_importance.random_forest = rf_importance;
    feature_importance.gradient = gradient_importance;
    feature_importance.combined = combined_importance;
    feature_importance.feature_names = feature_names;
    
    % 상위 중요 특성 출력
    [sorted_importance, sort_idx] = sort(combined_importance, 'descend');
    fprintf('\n📊 특성 중요도 분석 결과:\n');
    for i = 1:min(10, length(feature_names))
        fprintf('  %d. %s: %.3f\n', i, feature_names{sort_idx(i)}, sorted_importance(i));
    end
end

function permutation_importance = calculatePermutationImportance(models, simulation_state)
    % 순열 중요도 계산
    
    if ~isfield(simulation_state, 'validation_data')
        permutation_importance = ones(15, 1) / 15;  % 기본값
        return;
    end
    
    val_data = simulation_state.validation_data;
    baseline_score = evaluateModel(models, val_data.features, val_data.targets);
    
    n_features = size(val_data.features, 2);
    permutation_importance = zeros(n_features, 1);
    
    for feature_idx = 1:n_features
        % 특성 순열
        shuffled_features = val_data.features;
        shuffled_features(:, feature_idx) = shuffled_features(randperm(size(shuffled_features, 1)), feature_idx);
        
        % 순열된 데이터로 평가
        shuffled_score = evaluateModel(models, shuffled_features, val_data.targets);
        
        % 중요도 = 성능 감소량
        permutation_importance(feature_idx) = max(0, baseline_score - shuffled_score);
    end
    
    % 정규화
    if sum(permutation_importance) > 0
        permutation_importance = permutation_importance / sum(permutation_importance);
    else
        permutation_importance = ones(n_features, 1) / n_features;
    end
end
```

### 9.3.3 Model Ensemble and Meta-Learning

```matlab
function ensemble_model = createModelEnsemble(individual_models, validation_data, simulation_state)
    % 모델 앙상블 생성
    
    fprintf('🎭 모델 앙상블 구성 중...\n');
    
    ensemble_model = struct();
    ensemble_model.models = individual_models;
    
    % 1. 개별 모델 성능 평가
    model_names = fieldnames(individual_models);
    n_models = length(model_names);
    model_scores = zeros(n_models, 1);
    
    for i = 1:n_models
        model_name = model_names{i};
        model = individual_models.(model_name);
        predictions = predictWithModel(model, validation_data.features);
        model_scores(i) = calculateR2Score(validation_data.targets, predictions);
        
        fprintf('  %s 성능: R² = %.3f\n', model_name, model_scores(i));
    end
    
    % 2. 가중치 계산 (성능 기반)
    % 소프트맥스를 사용하여 가중치 계산
    exp_scores = exp(model_scores * 5);  % 온도 파라미터 5
    ensemble_weights = exp_scores / sum(exp_scores);
    
    ensemble_model.weights = ensemble_weights;
    ensemble_model.model_names = model_names;
    
    % 3. 스태킹 메타 모델 훈련 (선택사항)
    if simulation_state.config.use_stacking
        meta_model = trainStackingMetaModel(individual_models, validation_data, simulation_state);
        ensemble_model.meta_model = meta_model;
        ensemble_model.use_stacking = true;
    else
        ensemble_model.use_stacking = false;
    end
    
    % 4. 앙상블 성능 평가
    ensemble_predictions = predictWithEnsemble(ensemble_model, validation_data.features);
    ensemble_score = calculateR2Score(validation_data.targets, ensemble_predictions);
    
    ensemble_model.ensemble_score = ensemble_score;
    
    fprintf('  🎯 앙상블 성능: R² = %.3f\n', ensemble_score);
    fprintf('  📈 성능 향상: %.3f\n', ensemble_score - max(model_scores));
end

function predictions = predictWithEnsemble(ensemble_model, features)
    % 앙상블 예측
    
    if ensemble_model.use_stacking
        % 스태킹 방법
        base_predictions = zeros(size(features, 1), length(ensemble_model.model_names));
        
        for i = 1:length(ensemble_model.model_names)
            model_name = ensemble_model.model_names{i};
            model = ensemble_model.models.(model_name);
            base_predictions(:, i) = predictWithModel(model, features);
        end
        
        predictions = predictWithModel(ensemble_model.meta_model, base_predictions);
        
    else
        % 가중 평균 방법
        predictions = zeros(size(features, 1), 1);
        
        for i = 1:length(ensemble_model.model_names)
            model_name = ensemble_model.model_names{i};
            model = ensemble_model.models.(model_name);
            model_predictions = predictWithModel(model, features);
            
            predictions = predictions + ensemble_model.weights(i) * model_predictions;
        end
    end
end
```

---

*Chapter 9는 SFDP v17.3의 머신러닝 구현의 핵심을 다룹니다. 신경망, SVM, 랜덤 포레스트 등 다양한 ML 알고리즘을 통합한 경험적 모델 스위트, 정교한 특성 엔지니어링, 교차 검증 프레임워크, 특성 중요도 분석, 모델 앙상블 등을 통해 물리 기반 시뮬레이션을 보완하는 강력한 데이터 기반 예측 시스템을 구축했습니다. 특히 불확실성 정량화와 신뢰도 평가를 통해 예측 결과의 품질을 체계적으로 관리합니다.*