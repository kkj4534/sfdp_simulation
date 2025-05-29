# SFDP v17.3 사용자 매뉴얼
**6-Layer Hierarchical Multi-Physics Simulator for Ti-6Al-4V Machining**

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [설치 및 환경 설정](#설치-및-환경-설정)
3. [빠른 시작 가이드](#빠른-시작-가이드)
4. [상세 사용법](#상세-사용법)
5. [설정 및 커스터마이징](#설정-및-커스터마이징)
6. [결과 해석](#결과-해석)
7. [문제 해결](#문제-해결)
8. [고급 사용법](#고급-사용법)

---

## 시스템 개요

### **SFDP v17.3란?**
SFDP(Smart Framework for Data-driven Physics)는 Ti-6Al-4V 가공 공정을 위한 6-layer 계층형 멀티피직스 시뮬레이터입니다.

### **특징**
- 6-Layer 계층 구조: Physics → Empirical → Kalman → Validation
- 변수별 적응형 칼먼 필터: 온도(±10-15%), 마모(±8-12%), 조도(±12-18%)
- 42개 함수 구현: 물리학, 경험식, 머신러닝, 검증, 유틸리티
- 상대경로 시스템으로 이동 가능한 구조
- ASME V&V 10-2006 표준 참조

### **지원 가공 조건**
- **재료**: Ti-6Al-4V (primary), Al-7075, SS-316L (limited)
- **공구**: Carbide, TiAlN Coated, CBN, PCD
- **절삭속도**: 50-500 m/min
- **이송속도**: 0.05-0.5 mm/rev
- **절삭깊이**: 0.2-5.0 mm

---

## 설치 및 환경 설정

### **시스템 요구사항**
- **MATLAB**: R2020a 이상 (권장: R2023a)
- **메모리**: 최소 8GB RAM (권장: 16GB)
- **저장공간**: 2GB 이상
- **OS**: Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+

### **필수 MATLAB 툴박스**
- Statistics and Machine Learning Toolbox (권장)
- Curve Fitting Toolbox (권장)
- Symbolic Math Toolbox (선택)

### **외부 툴박스 (선택사항)**
- FEATool Multiphysics v1.17+ (고급 물리학 해석)
- GIBBON v3.5+ (접촉역학 해석)
- CFDTool v1.10+ (냉각제 유동 해석)

### **설치 과정**

#### **방법 1: USB/폴더 복사 (권장)**
1. `write_here` 폴더를 원하는 위치에 복사
2. 폴더명을 `SFDP_v17_3`로 변경 (선택)
3. MATLAB에서 해당 폴더로 이동
4. `SFDP_v17_3_main.m` 실행

#### **방법 2: GitHub Clone**
```bash
git clone [repository_url]
cd SFDP_v17_3
```

### **초기 설정 확인**
```matlab
% MATLAB에서 실행
cd 'C:\your_path\SFDP_v17_3'  % 실제 경로로 변경
SFDP_v17_3_main()
```

---

## 빠른 시작 가이드

### **Step 1: 프로그램 실행**
```matlab
% MATLAB 명령창에서
SFDP_v17_3_main()
```

### **Step 2: 공구 선택**
프로그램이 시작되면 공구 선택 화면이 나타납니다:
```
Available tool options for Ti-6Al-4V machining:
  1. Carbide Insert (CNMG120408, Uncoated) - Standard Choice
  2. TiAlN Coated Carbide (CNMG120408, TiAlN) - High Performance  
  3. CBN Insert (CNGA120408, CBN) - Ultra High Speed
  4. PCD Insert (CCMG120408, PCD) - Ultra Precision

GWO automatic optimization enabled. Running tool optimization...
```

**자동 공구 최적화 기능**
- Grey Wolf Optimizer (GWO)를 사용한 공구 선택
- 다기준 최적화: 공구 수명(40%), 표면 품질(25%), 비용(20%), 생산성(15%)
- 수동 선택 가능: GWO가 비활성화되면 직접 선택
```
Please select tool number (1-4): 
```

**추천 선택:**
- **일반 가공**: 1번 (Carbide Insert)
- **고속 가공**: 2번 (TiAlN Coated)
- **정밀 가공**: 4번 (PCD Insert)

### **Step 3: 결과 확인**
프로그램이 자동으로 6-layer 계산을 수행하고 결과를 표시합니다.

### **전체 실행 시간**
- **일반 조건**: 약 60-90초
- **고급 물리학 포함**: 약 2-3분

---

## 상세 사용법

### **6-Layer 계산 과정**

#### **Layer 1: Advanced Physics (고급 물리학)**
```
Advanced Physics Layer (3D FEM-level)
  ├─ 3D thermal analysis with FEATool
  ├─ GIBBON contact mechanics  
  ├─ Multi-scale surface roughness
  └─ 6-mechanism wear physics
```

#### **Layer 2: Simplified Physics (간소화 물리학)**
```
Simplified Physics Layer (Classical solutions)
  ├─ Enhanced Jaeger moving source
  ├─ Taylor tool life enhancement
  ├─ Classical roughness models
  └─ Merchant cutting force analysis
```

#### **Layer 3: Empirical Assessment (경험적 평가)**
```
Empirical Assessment Layer (Data-driven)
  ├─ ML ensemble prediction
  ├─ Traditional correlations
  ├─ Built-in database lookup
  └─ Statistical analysis
```

#### **Layer 4: Data Correction (데이터 보정)**
```
Data Correction Layer (Intelligent fusion)
  ├─ Physics-empirical fusion
  ├─ Bias correction
  ├─ Quality improvement
  └─ Consistency checking
```

#### **Layer 5: Adaptive Kalman Filter (적응형 칼먼 필터)**
```
Adaptive Kalman Layer (Variable-specific dynamics)
  ├─ Temperature: ±10-15% correction
  ├─ Tool wear: ±8-12% correction
  ├─ Surface roughness: ±12-18% correction
  └─ Dynamic weight adjustment
```

#### **Layer 6: Final Validation (최종 검증)**
```
Final Validation Layer (Quality assurance)
  ├─ Physics consistency checking
  ├─ Statistical validation
  ├─ Bounds verification
  └─ ASME V&V standards
```

### **출력 결과 구조**
```matlab
final_results = struct(
    'temperature',        % 온도 [°C]
    'tool_wear',         % 공구 마모 [mm]
    'surface_roughness', % 표면 조도 [μm]
    'cutting_force',     % 절삭력 [N]
    'confidence',        % 예측 신뢰도 [0-1]
    'layer_results',     % 각 layer별 상세 결과
    'validation_results' % 검증 결과
);
```

---

## 설정 및 커스터마이징

### **기본 설정 변경**

#### **재료 물성 변경**
```matlab
% config/SFDP_constants_tables.m 편집
constants.material.ti6al4v.density = 4430;           % kg/m³
constants.material.ti6al4v.thermal_conductivity = 7.2; % W/m·K
constants.material.ti6al4v.specific_heat = 526;      % J/kg·K
```

#### **칼먼 필터 dynamics 조정**
```matlab
% 온도 correction 범위 변경
constants.kalman.temperature.correction_range = [0.08, 0.12]; % ±8-12%로 변경

% 마모 correction 범위 변경  
constants.kalman.tool_wear.correction_range = [0.10, 0.15]; % ±10-15%로 변경
```

#### **병렬 처리 설정**
```matlab
% 병렬 처리 활성화/비활성화
constants.computational.parallel.enabled = true;
constants.computational.parallel.max_workers = 4;
```

### **고급 설정**

#### **로깅 레벨 조정**
```matlab
% config/SFDP_user_config.m
config.data_locations.logs_directory = 'adaptive_logs';  % 로그 디렉토리 설정

% config/SFDP_constants_tables.m에서 로깅 설정
constants.computational.logging.default_log_level = 'INFO';  % 'DEBUG', 'WARNING', 'ERROR'
constants.computational.logging.max_log_file_size_mb = 50;   % 로그 파일 최대 크기
constants.computational.logging.log_rotation_count = 5;      % 로그 파일 회전 개수
constants.computational.logging.console_output = true;       % 콘솔 출력
constants.computational.logging.file_output = true;          % 파일 저장
```

로깅 시스템 기능
- 설정 가능한 로그 경로
- 자동 로그 파일 회전
- JSON 형식 지원

#### **물리학 모듈 활성화/비활성화**
```matlab
user_config.physics.enable_fem = true;     % FEATool 사용
user_config.physics.enable_gibbon = false; % GIBBON 비활성화
user_config.physics.fallback_mode = true;  % 대체 모드 활성화
```

---

## 결과 해석

### **주요 출력 변수**

#### **온도 (Temperature)**
```
예측 온도: 650°C
신뢰도: 0.85
물리학적 의미: 
  - 350°C 이하: 안전한 가공 온도
  - 350-600°C: 주의 필요 (tool wear 가속)
  - 600°C 이상: 위험 (급속 공구 마모)
```

#### **공구 마모 (Tool Wear)**
```
예측 마모: 0.25mm
신뢰도: 0.78
물리학적 의미:
  - 0.1mm 이하: 초기 마모 (excellent)
  - 0.1-0.3mm: 정상 마모 (good)
  - 0.3mm 이상: 공구 교체 필요 (poor)
```

#### **표면 조도 (Surface Roughness)**
```
예측 조도: Ra 1.2μm
신뢰도: 0.82
가공 품질 평가:
  - Ra < 0.8μm: 정밀 가공 (precision)
  - Ra 0.8-1.6μm: 일반 가공 (standard)
  - Ra > 1.6μm: 거친 가공 (rough)
```

### **신뢰도 해석**
- **0.9-1.0**: Excellent (매우 높은 신뢰도)
- **0.8-0.9**: Good (높은 신뢰도)
- **0.7-0.8**: Fair (보통 신뢰도)
- **0.6-0.7**: Poor (낮은 신뢰도)
- **< 0.6**: Very Poor (매우 낮은 신뢰도)

### **검증 결과 해석**
```
VALIDATION SUMMARY
=====================
Overall Status: PASSED
Overall Confidence: 0.85/1.00
Critical Issues: 0
Warnings: 1
Validation Time: 15.2 seconds

VALIDATION PASSED - Results are reliable
```

**상태 해석:**
- **PASSED**: 모든 검증 통과, 결과 신뢰 가능
- **CONDITIONAL**: 일부 경고 있음, 주의하여 사용
- **FAILED**: 심각한 문제 있음, 결과 신뢰 불가

---

## 문제 해결

### **일반적인 오류**

#### **1. 경로 오류**
```
Error: Cannot find function 'SFDP_initialize_system'
```
**해결책:**
```matlab
% 현재 폴더가 SFDP_v17_3인지 확인
pwd
% 올바른 폴더로 이동
cd 'C:\path\to\SFDP_v17_3'
```

#### **2. 메모리 부족**
```
Error: Out of memory
```
**해결책:**
- MATLAB 재시작
- 다른 프로그램 종료
- `constants.computational.memory.max_usage_mb` 값 감소

#### **3. 툴박스 누락**
```
Warning: Statistics Toolbox not found, using fallback methods
```
**해결책:**
- Statistics Toolbox 설치 (권장)
- 또는 계속 진행 (대체 방법 사용)

#### **4. 데이터 로딩 실패**
```
Warning: No materials data available, using default coefficients
```
**해결책:**
- `data` 폴더에 CSV 파일 확인
- 파일 권한 확인
- 파일 형식 확인 (UTF-8 인코딩)

### **성능 최적화**

#### **실행 속도 향상**
```matlab
% 병렬 처리 활성화
constants.computational.parallel.enabled = true;

% 메모리 캐싱 활성화
constants.computational.caching.enabled = true;

% 고급 물리학 모듈 비활성화 (속도 우선시)
user_config.physics.enable_fem = false;
```

#### **메모리 사용량 감소**
```matlab
% 최대 메모리 사용량 제한
constants.computational.memory.max_usage_mb = 2048; % 2GB

% 데이터 압축 활성화
constants.computational.compression.enabled = true;
```

---

## 고급 사용법

### **배치 실행**

#### **여러 조건 자동 계산**
```matlab
% 배치 실행 스크립트 예제
cutting_speeds = [100, 150, 200, 250];
feed_rates = [0.1, 0.2, 0.3];

results_matrix = [];
for i = 1:length(cutting_speeds)
    for j = 1:length(feed_rates)
        % 조건 설정
        simulation_state.cutting_speed = cutting_speeds(i);
        simulation_state.feed_rate = feed_rates(j);
        
        % 시뮬레이션 실행
        [final_results] = SFDP_execute_6layer_calculations(...);
        
        % 결과 저장
        results_matrix(i,j,:) = [final_results.temperature, ...
                                final_results.tool_wear, ...
                                final_results.surface_roughness];
    end
end
```

### **결과 내보내기**

#### **CSV 파일로 저장**
```matlab
% 결과를 테이블로 변환
results_table = table(cutting_speeds', temperatures', tool_wears', ...
                     'VariableNames', {'Speed_mmin', 'Temp_C', 'Wear_mm'});

% CSV 파일로 저장
writetable(results_table, 'simulation_results.csv');
```

#### **그래프 생성**
```matlab
% 온도 분포 그래프
figure('Name', 'Temperature vs Cutting Speed');
plot(cutting_speeds, temperatures, 'bo-', 'LineWidth', 2);
xlabel('Cutting Speed (m/min)');
ylabel('Temperature (°C)');
title('SFDP v17.3 Simulation Results');
grid on;

% 저장
saveas(gcf, 'temperature_results.png');
```

### **커스텀 재료 추가**

#### **새로운 재료 정의**
```matlab
% config/SFDP_constants_tables.m에 추가
constants.material.al7075.name = 'Al-7075';
constants.material.al7075.density = 2810;           % kg/m³
constants.material.al7075.thermal_conductivity = 130; % W/m·K
constants.material.al7075.specific_heat = 960;      % J/kg·K
constants.material.al7075.melting_point = 635;      % °C
constants.material.al7075.hardness = 150;           % HV
```

### **API 사용법**

#### **개별 함수 호출**
```matlab
% 물리학 계산만 실행
physics_results = SFDP_physics_suite.calculate3DThermalFEATool(...);

% 칼먼 필터만 실행
kalman_results = SFDP_kalman_fusion_suite.applyEnhancedAdaptiveKalman(...);

% 검증만 실행
validation_results = SFDP_validation_qa_suite.performComprehensiveValidation(...);
```

---

## 지원 및 문의

### **문서 및 자료**
- **사용자 매뉴얼**: 현재 문서
- **기술 문서**: `docs/SFDP_v17_3_Code_Evaluation_Report.md`
- **로드맵**: `docs/SFDP_v17_3_Roadmap.md`

### **로그 파일 위치**
- **시뮬레이션 로그**: `logs/simulation_YYYYMMDD_HHMMSS.log`
- **오류 로그**: `logs/error_YYYYMMDD_HHMMSS.log`
- **성능 로그**: `logs/performance_YYYYMMDD_HHMMSS.log`

### **문제 리포트**
문제 발생 시 다음 정보를 포함하여 문의:
1. MATLAB 버전
2. 운영체제
3. 오류 메시지 전문
4. 재현 단계
5. 로그 파일

---

## 참고 자료

### **이론적 배경**
- Taylor (1907) "On the art of cutting metals"
- Kalman (1960) "A New Approach to Linear Filtering"
- Carslaw & Jaeger (1959) "Conduction of Heat in Solids"
- ASME V&V 10-2006 "Verification and Validation Standards"

### **관련 소프트웨어**
- FEATool Multiphysics: https://www.featool.com/
- GIBBON: https://www.gibboncode.org/
- MATLAB: https://www.mathworks.com/

### **학술 논문**
1. "Variable-Specific Adaptive Kalman Filtering for Multi-Physics Machining Simulation"
2. "Hierarchical Physics-Empirical-AI Fusion Framework for Titanium Machining"
3. "Comprehensive Uncertainty Quantification in Multi-Physics Manufacturing Simulation"

---

**작성자**: SFDP 개발팀  
**버전**: v17.3  
**최종 업데이트**: 2025년 5월 28일  
**라이선스**: Academic Research Use Only