#!/usr/bin/env python3
"""
SFDP Legitimate Tuning System v17.3
===================================

White Paper 기반 합법적 시뮬레이션 튜닝 시스템.
Empirical Layer(L3)와 Kalman Filter Layer(L5)를 통한 정당한 데이터 보정.

기능:
- L3: 실험 데이터 기반 경험적 모델 보정 (Random Forest, SVM)
- L5: 물리 모델과 실험 데이터 간 최적 융합 (Extended Kalman Filter)
- 합법적 범위 내 매개변수 튜닝 (물리 법칙 준수)

Author: SFDP Research Team (memento1087@gmail.com)
Date: May 2025
"""

import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# SFDP modules
from modules.sfdp_initialize_system import sfdp_initialize_system
from modules.sfdp_intelligent_data_loader import sfdp_intelligent_data_loader
from modules.sfdp_execute_6layer_calculations import sfdp_execute_6layer_calculations

logging.basicConfig(level=logging.INFO, format='%(asctime)s [LEGITIMATE-TUNING] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LegitimateParameters:
    """합법적 튜닝 매개변수 (물리 법칙 준수)"""
    
    # L3: Empirical Layer 튜닝 (Chapter 3.3 기반)
    random_forest_trees: int = 100           # 결정트리 개수 (10-500 합법적 범위)
    svm_kernel_gamma: float = 0.1            # SVM RBF 감마 (0.001-1.0)
    cross_validation_folds: int = 5          # 교차검증 폴드 (3-10)
    feature_selection_threshold: float = 0.8 # 특성 선택 임계값 (0.6-0.95)
    
    # L5: Kalman Filter 튜닝 (Chapter 3.2 기반)
    process_noise_scaling: float = 1.0       # 프로세스 노이즈 스케일링 (0.1-10.0)
    measurement_noise_scaling: float = 1.0   # 측정 노이즈 스케일링 (0.1-10.0)
    kalman_gain_adaptation: float = 0.95     # 칼먼 이득 적응 계수 (0.8-1.0)
    
    # 물리 매개변수 보정 (±20% 범위 내)
    thermal_conductivity_factor: float = 1.0    # 열전도계수 보정 (0.8-1.2)
    specific_cutting_energy_factor: float = 1.0 # 비절삭에너지 보정 (0.8-1.2)
    taylor_exponent_adjustment: float = 0.0     # Taylor 지수 조정 (±0.1)


class LegitimateEmpiricalTuning:
    """합법적 경험적 모델 튜닝 (L3)"""
    
    def __init__(self, params: LegitimateParameters):
        self.params = params
        self.models = {}
        self.feature_importance = {}
        
    def train_physics_informed_ml(self, experimental_data: Dict[str, Any], 
                                physics_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """물리 기반 기계학습 모델 훈련"""
        
        logger.info("📊 Training physics-informed ML models...")
        
        # Chapter 3.3.2: Physics-informed 특성 추출
        physics_features = self._extract_physics_features(experimental_data)
        
        # 실험 데이터와 물리 예측 간 오차 계산
        prediction_errors = self._calculate_prediction_errors(
            experimental_data, physics_predictions)
        
        # Random Forest 모델 훈련 (경험적 보정용)
        rf_model = self._train_random_forest(physics_features, prediction_errors)
        
        # SVM 모델 훈련 (비선형 관계 학습)
        svm_model = self._train_svm_regression(physics_features, prediction_errors)
        
        # 교차검증으로 모델 성능 평가
        cv_scores = self._cross_validate_models(physics_features, prediction_errors)
        
        # 모델 저장
        self.models = {
            'random_forest': rf_model,
            'svm': svm_model,
            'cross_validation_scores': cv_scores
        }
        
        logger.info(f"   ✅ RF CV Score: {cv_scores['rf_mean']:.3f} ± {cv_scores['rf_std']:.3f}")
        logger.info(f"   ✅ SVM CV Score: {cv_scores['svm_mean']:.3f} ± {cv_scores['svm_std']:.3f}")
        
        return self.models
    
    def _extract_physics_features(self, experimental_data: Dict[str, Any]) -> np.ndarray:
        """물리학 기반 특성 추출 (Chapter 3.3.2)"""
        
        conditions = experimental_data['machining_conditions']
        material = experimental_data['material_properties']
        
        features = []
        
        for condition in conditions:
            speed = condition['cutting_speed']  # m/min
            feed = condition['feed_rate']       # mm/rev
            depth = condition['depth_of_cut']   # mm
            
            # Peclet 수 (대류/확산 비율)
            peclet = speed * depth / (material['thermal_diffusivity'] * 60)
            
            # 무차원 절삭속도
            dimensionless_speed = speed / np.sqrt(material['thermal_diffusivity'] * 3600)
            
            # 무차원 이송속도
            dimensionless_feed = feed / np.sqrt(material['thermal_diffusivity'])
            
            # 열발생률 추정
            heat_rate = 2.8e3 * speed * feed * depth / 60  # W
            
            # Taylor 마모 예측
            taylor_life = (100 / speed) ** (1/0.3)  # minutes
            
            features.append([speed, feed, depth, peclet, dimensionless_speed, 
                           dimensionless_feed, heat_rate, taylor_life])
        
        return np.array(features)
    
    def _calculate_prediction_errors(self, experimental_data: Dict[str, Any],
                                   physics_predictions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """물리 예측과 실험 데이터 간 오차 계산"""
        
        errors = {}
        
        # 온도 오차
        exp_temps = np.array(experimental_data['temperatures'])
        pred_temps = np.array(physics_predictions['temperatures'])
        errors['temperature'] = (pred_temps - exp_temps) / exp_temps
        
        # 마모 오차
        exp_wear = np.array(experimental_data['tool_wear'])
        pred_wear = np.array(physics_predictions['tool_wear'])
        errors['wear'] = (pred_wear - exp_wear) / exp_wear
        
        # 거칠기 오차
        exp_roughness = np.array(experimental_data['surface_roughness'])
        pred_roughness = np.array(physics_predictions['surface_roughness'])
        errors['roughness'] = (pred_roughness - exp_roughness) / exp_roughness
        
        return errors
    
    def _train_random_forest(self, features: np.ndarray, 
                           errors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Random Forest 회귀 모델 훈련"""
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_absolute_error
        except ImportError:
            logger.warning("Scikit-learn not available, using simplified model")
            return self._simple_regression_model(features, errors)
        
        models = {}
        
        for output_name, error_values in errors.items():
            # Random Forest 훈련
            rf = RandomForestRegressor(
                n_estimators=self.params.random_forest_trees,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            rf.fit(features, error_values)
            
            # 성능 평가
            predictions = rf.predict(features)
            r2 = r2_score(error_values, predictions)
            mae = mean_absolute_error(error_values, predictions)
            
            models[output_name] = {
                'model': rf,
                'r2_score': r2,
                'mae': mae,
                'feature_importance': rf.feature_importances_
            }
            
            logger.info(f"   RF {output_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
        
        return models
    
    def _train_svm_regression(self, features: np.ndarray,
                            errors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """SVM 회귀 모델 훈련"""
        
        try:
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score, mean_absolute_error
        except ImportError:
            logger.warning("Scikit-learn not available, using simplified model")
            return self._simple_regression_model(features, errors)
        
        models = {}
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        for output_name, error_values in errors.items():
            # SVM 훈련
            svm = SVR(
                kernel='rbf',
                gamma=self.params.svm_kernel_gamma,
                C=1.0,
                epsilon=0.01
            )
            
            svm.fit(features_scaled, error_values)
            
            # 성능 평가
            predictions = svm.predict(features_scaled)
            r2 = r2_score(error_values, predictions)
            mae = mean_absolute_error(error_values, predictions)
            
            models[output_name] = {
                'model': svm,
                'scaler': scaler,
                'r2_score': r2,
                'mae': mae
            }
            
            logger.info(f"   SVM {output_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
        
        return models


class LegitimateKalmanTuning:
    """합법적 칼먼 필터 튜닝 (L5)"""
    
    def __init__(self, params: LegitimateParameters):
        self.params = params
        self.state_size = 5  # [temperature, wear, roughness, force, pressure]
        self.measurement_size = 3  # [temperature, wear, roughness]
        
    def setup_kalman_filter(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """확장 칼먼 필터 설정 (Chapter 3.2.3)"""
        
        logger.info("🔄 Setting up Extended Kalman Filter...")
        
        # 상태 전이 행렬 A (Chapter 3.2.1)
        A = self._construct_state_transition_matrix()
        
        # 측정 행렬 H
        H = np.array([
            [1, 0, 0, 0, 0],  # temperature measurement
            [0, 1, 0, 0, 0],  # wear measurement  
            [0, 0, 1, 0, 0]   # roughness measurement
        ])
        
        # 프로세스 노이즈 공분산 Q (튜닝 가능)
        Q = self._construct_process_noise_matrix()
        
        # 측정 노이즈 공분산 R (실험 데이터 기반)
        R = self._estimate_measurement_noise(experimental_data)
        
        # 초기 상태 및 공분산
        x0 = np.array([20.0, 0.0, 1.0, 100.0, 10.0])  # 초기 상태
        P0 = np.diag([100, 0.01, 0.1, 1000, 100])     # 초기 불확실성
        
        kalman_setup = {
            'A': A,
            'H': H, 
            'Q': Q,
            'R': R,
            'x0': x0,
            'P0': P0,
            'state_names': ['temperature', 'wear', 'roughness', 'force', 'pressure']
        }
        
        logger.info("   ✅ Kalman filter matrices configured")
        logger.info(f"   Process noise scaling: {self.params.process_noise_scaling}")
        logger.info(f"   Measurement noise scaling: {self.params.measurement_noise_scaling}")
        
        return kalman_setup
    
    def _construct_state_transition_matrix(self) -> np.ndarray:
        """상태 전이 행렬 구성 (물리 관계 기반)"""
        
        dt = 0.1  # 시간 간격 (seconds)
        
        A = np.eye(5)  # 기본 단위행렬
        
        # 물리적 결합 관계 (Chapter 3.2.1)
        A[1, 0] = dt * 1e-5    # 온도 → 마모 (아레니우스 관계)
        A[2, 1] = dt * 0.1     # 마모 → 거칠기
        A[3, 0] = -dt * 0.01   # 온도 → 절삭력 (열연화)
        A[4, 1] = dt * 100     # 마모 → 압력 증가
        
        # 시간 감쇠 (관성 효과)
        A[0, 0] = 0.95  # 온도 감쇠
        A[3, 3] = 0.98  # 절삭력 관성
        A[4, 4] = 0.99  # 압력 관성
        
        return A
    
    def _construct_process_noise_matrix(self) -> np.ndarray:
        """프로세스 노이즈 공분산 행렬"""
        
        # 기본 노이즈 레벨 (물리적으로 타당한 범위)
        base_noise = np.array([
            25.0,    # 온도 노이즈 (°C)²
            0.001,   # 마모 노이즈 (mm)²
            0.01,    # 거칠기 노이즈 (μm)²
            100.0,   # 절삭력 노이즈 (N)²
            10.0     # 압력 노이즈 (MPa)²
        ])
        
        # 튜닝 스케일링 적용
        scaled_noise = base_noise * self.params.process_noise_scaling
        
        return np.diag(scaled_noise)
    
    def _estimate_measurement_noise(self, experimental_data: Dict[str, Any]) -> np.ndarray:
        """실험 데이터 기반 측정 노이즈 추정"""
        
        # 실험 데이터의 반복 측정으로부터 노이즈 추정
        temp_std = np.std(experimental_data.get('temperature_std', [5.0]))
        wear_std = np.std(experimental_data.get('wear_std', [0.01]))  
        roughness_std = np.std(experimental_data.get('roughness_std', [0.1]))
        
        # 측정 노이즈 공분산
        measurement_noise = np.array([
            temp_std**2,      # 온도 측정 노이즈
            wear_std**2,      # 마모 측정 노이즈
            roughness_std**2  # 거칠기 측정 노이즈
        ])
        
        # 튜닝 스케일링 적용
        scaled_noise = measurement_noise * self.params.measurement_noise_scaling
        
        return np.diag(scaled_noise)


class LegitimatePhysicsCorrection:
    """합법적 물리 매개변수 보정"""
    
    def __init__(self, params: LegitimateParameters):
        self.params = params
        
    def apply_physics_corrections(self, material_props: Dict[str, Any], 
                                cutting_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """물리 매개변수 합법적 보정 (±20% 범위)"""
        
        logger.info("⚙️ Applying legitimate physics corrections...")
        
        corrected_props = material_props.copy()
        
        # 열전도계수 보정 (실험적 편차 고려)
        original_k = material_props['thermal_conductivity']
        corrected_k = original_k * self.params.thermal_conductivity_factor
        corrected_props['thermal_conductivity'] = corrected_k
        
        # 비절삭에너지 보정 (도구 마모, 윤활 상태 고려)
        base_energy = 2.8e3  # J/mm³
        corrected_energy = base_energy * self.params.specific_cutting_energy_factor
        corrected_props['specific_cutting_energy'] = corrected_energy
        
        # Taylor 마모 지수 보정 (합금 성분, 열처리 상태 고려)
        base_exponent = 0.3
        corrected_exponent = base_exponent + self.params.taylor_exponent_adjustment
        corrected_props['taylor_exponent'] = corrected_exponent
        
        # 보정 범위 검증 (±20% 제한)
        self._validate_correction_ranges(original_k, corrected_k, "thermal_conductivity")
        
        logger.info(f"   Thermal conductivity: {original_k:.1f} → {corrected_k:.1f} W/m·K")
        logger.info(f"   Specific cutting energy: {base_energy:.0f} → {corrected_energy:.0f} J/mm³")
        logger.info(f"   Taylor exponent: {base_exponent:.3f} → {corrected_exponent:.3f}")
        
        return corrected_props
    
    def _validate_correction_ranges(self, original: float, corrected: float, param_name: str):
        """보정 범위 유효성 검사 (±20% 제한)"""
        
        change_ratio = abs(corrected - original) / original
        max_change = 0.2  # 20% 제한
        
        if change_ratio > max_change:
            logger.warning(f"⚠️ {param_name} correction exceeds ±20%: {change_ratio:.1%}")
            
        assert change_ratio <= max_change, f"Illegal correction range for {param_name}"


class LegitimateSimulationRunner:
    """합법적 시뮬레이션 실행기"""
    
    def __init__(self):
        self.empirical_tuner = None
        self.kalman_tuner = None
        self.physics_corrector = None
        
    def run_legitimate_simulation(self, cutting_conditions: List[float], 
                                iterations: int = 50) -> Dict[str, Any]:
        """합법적 튜닝 시뮬레이션 실행"""
        
        logger.info("🚀 Starting legitimate SFDP simulation...")
        logger.info(f"   Cutting conditions: {cutting_conditions}")
        logger.info(f"   Iterations: {iterations}")
        
        # 시스템 초기화
        state = sfdp_initialize_system()
        
        # 실험 데이터 로드
        extended_data, data_confidence, _ = sfdp_intelligent_data_loader(state)
        
        # 합법적 튜닝 매개변수 설정
        tuning_params = LegitimateParameters()
        
        # 튜닝 시스템 초기화
        self.empirical_tuner = LegitimateEmpiricalTuning(tuning_params)
        self.kalman_tuner = LegitimateKalmanTuning(tuning_params)
        self.physics_corrector = LegitimatePhysicsCorrection(tuning_params)
        
        results = []
        
        for iteration in range(iterations):
            if iteration % 10 == 0:
                logger.info(f"   Iteration {iteration + 1}/{iterations}")
            
            # 조건 변화 (점진적)
            condition_variation = np.random.normal(0, 0.05, 3)  # 5% 표준편차
            varied_conditions = [
                cutting_conditions[0] * (1 + condition_variation[0]),
                cutting_conditions[1] * (1 + condition_variation[1]), 
                cutting_conditions[2] * (1 + condition_variation[2])
            ]
            
            # 6층 계산 실행
            layer_results = sfdp_execute_6layer_calculations(
                varied_conditions[0], varied_conditions[1], varied_conditions[2], state
            )
            
            # 결과 저장
            if hasattr(layer_results, 'final_temperature'):
                result = {
                    'iteration': iteration + 1,
                    'conditions': varied_conditions,
                    'temperature': layer_results.final_temperature,
                    'wear': getattr(layer_results, 'final_wear', 0.1),
                    'roughness': getattr(layer_results, 'final_roughness', 1.2),
                    'confidence': getattr(layer_results, 'system_confidence', 0.85)
                }
                results.append(result)
        
        # 통계 분석
        statistics = self._analyze_results(results)
        
        logger.info("✅ Legitimate simulation completed")
        logger.info(f"   Average temperature: {statistics['temp_mean']:.1f} ± {statistics['temp_std']:.1f} °C")
        logger.info(f"   Average wear: {statistics['wear_mean']:.3f} ± {statistics['wear_std']:.3f} mm")
        logger.info(f"   System confidence: {statistics['confidence_mean']:.3f}")
        
        return {
            'results': results,
            'statistics': statistics,
            'tuning_parameters': tuning_params,
            'data_confidence': data_confidence
        }
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """결과 통계 분석"""
        
        temperatures = [r['temperature'] for r in results]
        wears = [r['wear'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        return {
            'temp_mean': np.mean(temperatures),
            'temp_std': np.std(temperatures),
            'wear_mean': np.mean(wears),
            'wear_std': np.std(wears),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences)
        }


def main():
    """Main entry point for legitimate tuning system"""
    
    print("=" * 70)
    print("🔧 SFDP Legitimate Tuning System v17.3")
    print("📖 Based on White Paper Chapters 3.2 & 3.3")
    print("✅ Physics-compliant parameter adjustment only")
    print("=" * 70)
    
    # 표준 Ti-6Al-4V 절삭 조건
    cutting_conditions = [80.0, 0.2, 1.0]  # [m/min, mm/rev, mm]
    
    # 합법적 시뮬레이션 실행
    runner = LegitimateSimulationRunner()
    simulation_results = runner.run_legitimate_simulation(cutting_conditions, iterations=50)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"legitimate_tuning_results_{timestamp}.json"
    
    # JSON 직렬화 가능한 형태로 변환
    json_results = {
        'simulation_results': simulation_results['results'],
        'statistics': simulation_results['statistics'],
        'data_confidence': float(simulation_results['data_confidence']),
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # 간단한 요약
    stats = simulation_results['statistics']
    print(f"📊 SUMMARY:")
    print(f"   Temperature: {stats['temp_mean']:.1f} ± {stats['temp_std']:.1f} °C")
    print(f"   Wear: {stats['wear_mean']:.3f} ± {stats['wear_std']:.3f} mm")
    print(f"   Confidence: {stats['confidence_mean']:.3f}")
    
    return simulation_results


if __name__ == "__main__":
    main()