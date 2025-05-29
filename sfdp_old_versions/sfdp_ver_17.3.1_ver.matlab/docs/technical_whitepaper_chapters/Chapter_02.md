# Chapter 2: Fundamental Physics in Machining

## 2.1 Heat Transfer Fundamentals in Machining

**기본 개념: 왜 가공에서 열이 중요한가?**

금속 가공에서 열 문제는 피할 수 없습니다. 도구가 재료를 깎을 때 기계적 에너지의 대부분(90% 이상)이 열로 변환됩니다. 이 열이 어디로 가는지, 얼마나 뜨거워지는지를 정확히 예측하는 것이 성공적인 가공의 핵심입니다.

### 2.1.1 Fourier's Law and 3D Heat Conduction: ∇·(k∇T) = ρcp(∂T/∂t) - Q

**푸리에 법칙의 기본 이해**

열은 항상 뜨거운 곳에서 차가운 곳으로 흘러갑니다. 이를 수학적으로 표현한 것이 푸리에 법칙입니다:

```
q = -k∇T
```

여기서:
- q: 열유속 벡터 [W/m²] - 단위 면적당 흐르는 열의 양
- k: 열전도계수 [W/m·K] - 재료가 열을 얼마나 잘 전달하는지
- ∇T: 온도 기울기 [K/m] - 온도가 얼마나 급격히 변하는지

**3D 열전도 방정식 유도**

에너지 보존 법칙을 적용하면:

```
들어오는 열 - 나가는 열 + 생성되는 열 = 축적되는 열
```

이를 수학적으로 표현하면:

```
∇·(k∇T) + Q = ρcp(∂T/∂t)
```

각 항의 물리적 의미:
- `∇·(k∇T)`: 전도로 인한 열의 확산 [W/m³]
- `Q`: 단위 체적당 열 생성률 [W/m³] (가공에서는 마찰열)
- `ρcp(∂T/∂t)`: 온도 변화로 인한 에너지 축적 [W/m³]

**SFDP에서의 실제 구현**

우리 시스템에서는 이 방정식을 다음과 같이 구현합니다:

```matlab
% SFDP_physics_suite.m:177-178에서 실제 코드
fea.phys.ht.eqn.coef{2,end} = {material_props.thermal_conductivity}; % k
fea.phys.ht.eqn.coef{3,end} = {material_props.density * material_props.specific_heat}; % ρcp
```

Ti-6Al-4V의 실제 물성값:
- 열전도계수 k = 6.7 W/m·K (온도에 따라 변함)
- 밀도 ρ = 4420 kg/m³
- 비열 cp = 526 J/kg·K (온도에 따라 변함)

### 2.1.2 Moving Heat Sources: Jaeger Theory and Extensions

**움직이는 열원의 개념**

가공에서는 도구가 움직이면서 열을 발생시킵니다. 이는 정지된 열원과 완전히 다른 현상입니다:

1. **정지 열원**: 열이 모든 방향으로 균등하게 퍼짐
2. **움직이는 열원**: 열이 뒤쪽으로 치우쳐서 퍼짐 (comet tail 현상)

**Jaeger의 움직이는 선 열원 이론**

Jaeger(1942)가 개발한 이론은 다음과 같습니다:

```
T(x,y,t) = (Q/(4πk)) × ∫[0 to t] (1/τ) × exp(-(x-vτ)²+y²)/(4ατ)) dτ
```

여기서:
- Q: 단위 길이당 열입력 [W/m]
- v: 열원의 이동속도 [m/s] (= 절삭속도)
- α: 열확산계수 [m²/s] = k/(ρcp)
- τ: 적분 변수 (시간)

**SFDP에서의 확장된 구현**

우리는 Jaeger 이론을 3D로 확장하고 가공에 특화된 수정을 가했습니다:

```matlab
% SFDP_physics_suite.m:194-198에서 실제 구현
heat_source_expr = sprintf('%.3e * exp(-((x-%.6f)^2/(%.6f)^2 + y^2/(%.6f)^2 + (z-%.6f)^2/(%.6f)^2))', ...
    heat_generation_rate, cutting_position_x, heat_source_length/2, ...
    heat_source_width/2, workpiece_height, heat_source_depth/2);
```

이 식의 의미:
- `heat_generation_rate`: 총 열발생률 [W]
- `cutting_position_x`: 시간에 따른 도구 위치
- 3D 가우시안 분포로 열원을 모델링
- 실제 도구 형상을 반영한 열원 크기

### 2.1.3 Boundary Conditions: Convection, Radiation, Contact Resistance

**경계조건의 중요성**

아무리 좋은 열전도 방정식이 있어도 경계에서 어떤 일이 일어나는지 모르면 문제를 풀 수 없습니다. 가공에서는 여러 종류의 경계조건이 동시에 존재합니다.

**1) 대류 경계조건 (Convection)**

공기나 냉각제와의 열교환:

```
q" = h(T - T_ambient)
```

여기서:
- h: 대류 열전달계수 [W/m²·K]
- T_ambient: 주변 온도 [K]

SFDP에서의 구현:
- 자연대류: h = 5-25 W/m²·K (공기 중)
- 강제대류: h = 50-500 W/m²·K (냉각제 사용시)
- 고압 쿨런트: h = 1000-5000 W/m²·K

**2) 복사 경계조건 (Radiation)**

고온에서는 복사 열손실이 중요합니다:

```
q" = εσ(T⁴ - T_ambient⁴)
```

여기서:
- ε: 방사율 [무차원] (Ti-6Al-4V의 경우 0.3-0.6)
- σ: Stefan-Boltzmann 상수 = 5.67×10⁻⁸ W/m²·K⁴

**3) 접촉 저항 (Contact Resistance)**

도구-재료 접촉면에서의 열전달:

```
q" = (T_tool - T_workpiece) / R_contact
```

여기서 R_contact는 접촉 저항으로, 다음에 의존합니다:
- 표면 거칠기
- 접촉압력
- 재료의 조합
- 온도

**SFDP에서의 통합 구현**

```matlab
% applyAdvancedThermalBoundaryConditions 함수에서
% 모든 경계조건을 동시에 적용
boundary_conditions = struct();
boundary_conditions.convection_coeff = calculate_convection_coefficient(cutting_conditions);
boundary_conditions.radiation_emissivity = material_props.emissivity;
boundary_conditions.contact_resistance = calculate_contact_resistance(contact_pressure, surface_roughness);
```

## 2.2 Mechanical Contact and Wear Physics

### 2.2.1 Contact Mechanics: Hertzian Theory and Real Surface Contact

**접촉역학이 왜 중요한가?**

가공에서 도구와 재료가 접촉하는 부분에서 모든 일이 일어납니다:
- 열이 발생하고
- 힘이 전달되고  
- 재료가 제거되고
- 도구가 마모됩니다

따라서 접촉 현상을 정확히 이해하는 것이 핵심입니다.

**Hertz 접촉 이론의 기본**

Hertz(1881)는 두 탄성체가 접촉할 때의 응력 분포를 다음과 같이 유도했습니다:

```
p(r) = p₀√(1 - r²/a²)    (r ≤ a인 영역에서)
```

여기서:
- p₀: 최대 접촉압력 [Pa]
- a: 접촉 반경 [m]
- r: 접촉 중심으로부터의 거리 [m]

접촉 반경과 최대 압력의 관계:

```
a³ = (3FR)/(4E*)
p₀ = (3F)/(2πa²)
```

여기서:
- F: 접촉력 [N]
- R: 등가 곡률반경 [m]
- E*: 등가 탄성계수 [Pa]

**실제 표면 접촉의 복잡성**

하지만 실제 표면은 Hertz 이론의 가정과 다릅니다:

1. **표면이 완전히 매끄럽지 않음**
   - 마이크로미터 수준의 거칠기 존재
   - 실제 접촉 면적 << 명목상 접촉 면적

2. **재료가 완전히 탄성적이지 않음**
   - 고온에서 점소성 변형 발생
   - 재료 특성이 온도에 따라 변함

3. **접촉면에서 화학반응 발생**
   - 확산, 산화 등의 현상
   - 새로운 화합물 형성

**SFDP에서의 접촉 모델링**

우리는 GIBBON 툴박스를 활용하여 실제적인 접촉 해석을 수행합니다:

```matlab
% calculateCoupledWearGIBBON 함수에서 (Lines 481-560)
% 1. 3D 접촉 기하학 생성
[contact_geometry] = setup_3D_contact_geometry(tool_geometry, workpiece_geometry);

% 2. 접촉 압력 분포 계산  
[pressure_distribution] = solve_contact_pressure(contact_geometry, applied_force);

% 3. 온도 의존적 재료 특성 적용
material_props_temp = update_material_properties(material_props, temperature_field);
```

### 2.2.2 Tribological Wear Mechanisms: Adhesive, Abrasive, Diffusion, Oxidation

**마모란 무엇인가?**

마모(Wear)는 표면에서 재료가 점진적으로 제거되는 현상입니다. 가공 도구의 수명을 결정하는 가장 중요한 요소입니다.

**1) 접착 마모 (Adhesive Wear)**

고온, 고압에서 도구와 재료가 서로 달라붙었다가 떨어지면서 발생:

**Archard 마모 법칙:**
```
V = k × (F × s) / H
```

여기서:
- V: 마모 체적 [m³]
- k: 마모 계수 [무차원] (일반적으로 10⁻³ ~ 10⁻⁸)
- F: 접촉력 [N]  
- s: 슬라이딩 거리 [m]
- H: 경도 [Pa]

**SFDP에서의 구현:**
```matlab
% calculateAdvancedWearPhysics 함수 Lines 561-620
adhesive_wear_volume = wear_coefficient * (normal_force * sliding_distance) / hardness;
% 온도 의존성 고려
wear_coefficient_temp = wear_coefficient * exp(activation_energy / (R * temperature));
```

**2) 연마 마모 (Abrasive Wear)**

경한 입자가 연한 표면을 긁어내면서 발생:

```
V = k_abr × (F × s × tanφ) / H
```

여기서:
- k_abr: 연마 마모 계수
- φ: 연마 입자의 공격각도

Ti-6Al-4V 가공에서는 다음이 연마 입자가 됩니다:
- 칩 파편
- 도구 코팅 박리 입자
- 냉각제 내 이물질

**3) 확산 마모 (Diffusion Wear)**

고온에서 원자들이 서로 확산되면서 발생:

```
확산 플럭스 J = -D(∂C/∂x)
```

여기서:
- D: 확산계수 [m²/s] = D₀ × exp(-Q/(RT))
- Q: 활성화 에너지 [J/mol]
- R: 기체상수
- T: 절대온도 [K]

**SFDP에서의 구현:**
```matlab
% Lines 621-680
diffusion_coefficient = D0 * exp(-activation_energy / (gas_constant * temperature));
diffusion_wear_rate = diffusion_coefficient * concentration_gradient * contact_area;
```

**4) 산화 마모 (Oxidation Wear)**

고온에서 도구 표면이 산화되면서 발생:

```
산화막 성장: x² = kt
```

여기서:
- x: 산화막 두께 [m]
- k: 산화 속도 상수 [m²/s]
- t: 시간 [s]

Ti-6Al-4V와 카바이드 도구의 경우:
- 600°C 이상에서 급격한 산화 시작
- TiO₂, Al₂O₃ 형성
- 산화막이 박리되면서 마모 가속화

### 2.2.3 Archard Wear Law and Its Limitations: V = k·F·s/H

**Archard 법칙의 성공과 한계**

Archard 마모 법칙은 가장 널리 사용되는 마모 예측 모델이지만 한계가 있습니다:

**성공적인 측면:**
- 간단하고 직관적
- 많은 경우에 합리적인 예측
- 실험적으로 검증된 케이스 많음

**한계점:**
1. **마모 계수 k가 상수가 아님**
   - 온도, 속도, 압력에 따라 변화
   - 재료 조합에 따라 크게 달라짐

2. **단일 마모 메커니즘만 고려**
   - 실제로는 여러 메커니즘이 동시 작용
   - 각 메커니즘의 기여도가 조건에 따라 변함

3. **표면 상태 변화 무시**
   - 마모가 진행되면서 표면 거칠기 변화
   - 접촉 조건이 계속 바뀜

**SFDP에서의 확장된 마모 모델**

우리는 Archard 법칙을 다음과 같이 확장했습니다:

```matlab
% calculateAdvancedWearPhysics 함수에서 종합적 마모 계산
total_wear = adhesive_wear + abrasive_wear + diffusion_wear + oxidation_wear;

% 각 메커니즘의 가중치를 온도와 조건에 따라 동적 계산
wear_weights = calculate_wear_mechanism_weights(temperature, pressure, velocity);
weighted_wear = sum(wear_mechanisms .* wear_weights);
```

온도별 마모 메커니즘 기여도 (Ti-6Al-4V 가공):
- 500°C 이하: 연마 마모 주도 (70%)
- 500-700°C: 접착 마모 증가 (50%)
- 700°C 이상: 확산 + 산화 마모 지배적 (80%)

## 2.3 Surface Physics and Multi-Scale Modeling

### 2.3.1 Surface Roughness: From Atomic to Macro Scale

**표면 거칠기가 왜 중요한가?**

표면 거칠기는 단순히 "매끄러운지 거친지"의 문제가 아닙니다:

1. **기능적 영향**
   - 마찰 특성 결정
   - 접촉 면적에 영향
   - 열전달 특성 변화

2. **품질 영향**  
   - 제품의 외관 품질
   - 부품의 수명
   - 조립 정밀도

3. **경제적 영향**
   - 후처리 공정 필요성
   - 불량률 결정
   - 생산성에 직접 영향

**다중 스케일의 개념**

표면 거칠기는 여러 스케일에서 동시에 존재합니다:

```
나노스케일 (1-100 nm): 원자 수준의 불규칙성
마이크로스케일 (0.1-100 μm): 가공 마크, 진동 흔적  
마크로스케일 (0.1-10 mm): 기계적 오차, 변형
```

각 스케일이 서로 다른 물리적 현상을 지배합니다:
- **나노스케일**: 원자간 결합, 표면 에너지
- **마이크로스케일**: 도구 형상, 재료 변형
- **마크로스케일**: 기계 강성, 진동

**표면 거칠기 매개변수들**

**1) 산술평균 거칠기 (Ra)**
```
Ra = (1/L) ∫[0 to L] |y(x)| dx
```

**2) 제곱평균제곱근 거칠기 (Rq)**  
```
Rq = √[(1/L) ∫[0 to L] y(x)² dx]
```

**3) 최대 높이 (Rt)**
```
Rt = Rmax - Rmin
```

**SFDP에서의 다중스케일 모델링**

```matlab
% calculateMultiScaleRoughnessAdvanced 함수 (Lines 821-900)
function [roughness_multi_scale] = calculateMultiScaleRoughnessAdvanced(cutting_conditions, material_props)

% 나노스케일 거칠기 (원자 수준)
nano_roughness = calculate_atomic_scale_roughness(material_props.crystal_structure);

% 마이크로스케일 거칠기 (가공 흔적)
micro_roughness = calculate_machining_mark_roughness(cutting_conditions.feed_rate, ...
    cutting_conditions.tool_nose_radius);

% 마크로스케일 거칠기 (기계 진동)
macro_roughness = calculate_vibration_induced_roughness(cutting_conditions.spindle_speed, ...
    machine_dynamics.natural_frequency);

% 종합 거칠기 계산
total_roughness = combine_multi_scale_roughness(nano_roughness, micro_roughness, macro_roughness);
```

### 2.3.2 Fractal Analysis of Machined Surfaces

**프랙탈이란?**

프랙탈(Fractal)은 자기유사성을 갖는 기하학적 구조입니다. 가공된 표면은 다음과 같은 프랙탈 특성을 보입니다:

- 확대해도 비슷한 패턴 반복
- 스케일에 무관한 통계적 특성
- 멘델브로트가 제안한 개념

**표면 프랙탈 차원 (Fractal Dimension)**

```
D = (3 + H)
```

여기서 H는 Hurst 지수로:
- H = 0.5: 완전 랜덤 표면
- H > 0.5: 부드러운 표면 (양의 상관관계)
- H < 0.5: 거친 표면 (음의 상관관계)

**SFDP에서의 프랙탈 분석 구현**

```matlab
% 실제 프랙탈 차원 계산 알고리즘
function fractal_dimension = calculate_fractal_dimension(surface_profile)
    % Box-counting 방법 사용
    scales = logspace(-6, -3, 20); % 1μm ~ 1mm 스케일
    box_counts = zeros(size(scales));
    
    for i = 1:length(scales)
        box_counts(i) = count_boxes(surface_profile, scales(i));
    end
    
    % 로그-로그 플롯의 기울기가 프랙탈 차원
    poly_fit = polyfit(log(scales), log(box_counts), 1);
    fractal_dimension = -poly_fit(1);
end
```

### 2.3.3 Surface Integrity and Microstructural Changes

**표면 인티그리티란?**

Surface Integrity는 가공 후 표면의 종합적인 품질을 의미합니다:

1. **기하학적 특성**
   - 표면 거칠기
   - 웨이브니스 (waviness)
   - 형상 정확도

2. **물리적 특성**
   - 잔류응력
   - 경도 변화
   - 결정 구조 변화

3. **화학적 특성**
   - 표면 조성 변화
   - 산화막 형성
   - 오염물질 부착

**Ti-6Al-4V의 미세구조 변화**

Ti-6Al-4V는 가공 중 다음과 같은 미세구조 변화를 겪습니다:

**1) 상변태 (Phase Transformation)**
```
α + β ↔ β (고온에서)
```
- α상: 육방 조밀 격자 (HCP)
- β상: 체심 입방 격자 (BCC)
- 변태 온도: 약 995°C

**2) 동적 재결정 (Dynamic Recrystallization)**
- 고온, 고변형률에서 발생
- 결정립 미세화
- 기계적 성질 변화

**3) 가공 경화 (Work Hardening)**
- 전위 밀도 증가
- 표면 경도 상승
- 잔류응력 발생

**SFDP에서의 미세구조 예측**

```matlab
% 온도-변형률 이력을 기반으로 미세구조 변화 예측
function microstructure = predict_microstructure_changes(temperature_history, strain_history)
    
    % 상변태 확인
    phase_transformation = check_phase_transformation(temperature_history);
    
    % 동적 재결정 여부 확인
    dynamic_recrystallization = check_dynamic_recrystallization(temperature_history, strain_history);
    
    % 가공 경화 정도 계산
    work_hardening = calculate_work_hardening(strain_history);
    
    % 종합 미세구조 상태
    microstructure = combine_microstructure_effects(phase_transformation, ...
        dynamic_recrystallization, work_hardening);
end
```

이러한 미세구조 변화는 최종 제품의 성능에 직접적인 영향을 미칩니다:
- 피로 강도 변화
- 부식 저항성 변화  
- 크리프 특성 변화