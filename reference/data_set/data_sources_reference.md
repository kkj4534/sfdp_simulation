# 📚 SFDP Framework 데이터 출처 및 레퍼런스 명세서

## 📊 데이터 분류 및 출처

### 🎯 **기존 프로젝트 데이터 (검증된 학술 논문 기반)**

#### **Ti-6Al-4V 실험 데이터** ✅ **학술 논문 검증 완료**
**출처**: 프로젝트 기존 문서 `csv_references.md`에 명시된 15개 연구 논문

| 실험 ID | 논문 출처 | 저널/학회 | 연도 | 검증 상태 |
|---------|-----------|-----------|------|-----------|
| EXP001-003 | D'Mello et al. | Int. J. Advanced Manufacturing Technology | 2018 | ✅ 검증됨 |
| EXP004-006 | Safari et al. | Wear, Vol. 432-433 | 2019 | ✅ 검증됨 |
| EXP007-009 | Agrawal et al. | CIRP Annals | 2021 | ✅ 검증됨 |
| EXP010-012 | Comparison Study | J. Manufacturing Processes | 2023 | ✅ 검증됨 |
| EXP013-015 | ADRT | Int. J. Machine Tools and Manufacture | 2022 | ✅ 검증됨 |
| EXP016-018 | MicroMilling Study | Precision Engineering | 2018 | ✅ 검증됨 |
| EXP019-020 | HSM Study | Int. J. Advanced Manufacturing Technology | 2023 | ✅ 검증됨 |
| EXP021-023 | Natarajan et al. | J. Cleaner Production | 2024 | ✅ 검증됨 |
| EXP024-025 | Drilling Study | Manufacturing Letters | 2017 | ✅ 검증됨 |

**신뢰도**: ⭐⭐⭐⭐⭐ (최고급) - 실제 학술 논문 기반

---

### 🔍 **확장 데이터 (웹 검색 기반 추정)**

#### **Al2024-T3 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 2024년 5월 웹 검색 결과 + 물리학적 모델링

**주요 참고 자료**:
- WayKen (2024): "2024 aluminum alloy properties and machining methods"
- VMT (2024): "2024 Aluminum Alloy: Understanding Its Properties"
- Xometry (2024): "All About 2024 Aluminum Alloy"
- ScienceDirect Topics: "2024 Aluminium Alloy - an overview"

**데이터 생성 방법**:
```
기준값 = Ti-6Al-4V 대비 알루미늄 특성 반영
- 온도: 60% 수준 (낮은 절삭온도)
- 마모: 40% 수준 (쉬운 가공성)
- 조도: 30% 수준 (우수한 표면 마감)
- 절삭력: 60% 수준 (낮은 강도)
```

**신뢰도**: ⭐⭐⭐☆☆ (중간급) - 웹 정보 + 물리적 추정

---

#### **SS316L 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 2024년 5월 웹 검색 결과 + 스테인리스강 가공 특성

**주요 참고 자료**:
- Scientific Reports (2024): "Inconel 718/stainless steel 316 L multi-material"
- ResearchGate: "Chemical composition of 316L steel and Inconel 718"
- Nature.com: "Process optimization and mechanical properties analysis"

**데이터 생성 방법**:
```
기준값 = Ti-6Al-4V 대비 스테인리스강 특성 반영
- 온도: 70% 수준 (양호한 열전도)
- 마모: 75% 수준 (가공경화 현상)
- 조도: 70% 수준 (일반적 표면)
- 절삭력: 115% 수준 (높은 인성)
```

**신뢰도**: ⭐⭐⭐☆☆ (중간급) - 웹 정보 + 물리적 추정

---

#### **Inconel718 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 2024년 5월 웹 검색 결과 + 초내열합금 특성

**주요 참고 자료**:
- Scientific Reports (2024): "Process optimization Inconel 718/stainless steel"
- ScienceDirect: "High temperature fracture behavior of 316L-Inconel 718"
- ResearchGate: 다수의 Inconel 718 가공 연구

**데이터 생성 방법**:
```
기준값 = Ti-6Al-4V 대비 초내열합금 특성 반영
- 온도: 150% 수준 (고온 가공)
- 마모: 160% 수준 (극한 난삭재)
- 조도: 155% 수준 (거친 표면)
- 절삭력: 260% 수준 (극고강도)
```

**신뢰도**: ⭐⭐⭐☆☆ (중간급) - 웹 정보 + 물리적 추정

---

#### **AISI1045 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 2024년 5월 웹 검색 결과 + Johnson-Cook 연구

**주요 참고 자료**:
- MDPI (2019): "Johnson Cook Material and Failure Model Parameters Estimation of AISI-1045"
- ResearchGate: "Johnson-Cook material model parameters of AISI-1045 steel"
- ScienceDirect: "Determination of Johnson-Cook material model parameters for AISI 1045"

**데이터 생성 방법**:
```
기준값 = 중탄소강 특성 + 검증된 Johnson-Cook 파라미터
- Johnson-Cook A = 553.1e6 Pa (실험값)
- Johnson-Cook B = 600.8e6 Pa (실험값)
- 기타 가공 특성은 강재 일반 특성 적용
```

**신뢰도**: ⭐⭐⭐⭐☆ (양호) - 일부 실험 데이터 포함

---

#### **AISI4140 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 합금강 일반 특성 + 문헌 추정

**데이터 생성 방법**:
```
기준값 = AISI1045 대비 합금강 특성 강화
- 온도: 115% 수준 (합금 원소 영향)
- 마모: 120% 수준 (높은 경도)
- 절삭력: 145% 수준 (강화된 강도)
```

**신뢰도**: ⭐⭐☆☆☆ (낮음) - 추정 기반

---

#### **Al6061-T6 데이터** ⚠️ **웹 정보 기반 추정**
**출처**: 범용 알루미늄 합금 특성

**데이터 생성 방법**:
```
기준값 = Al2024-T3 대비 더 쉬운 가공성
- 온도: 85% 수준 (낮은 강도)
- 마모: 75% 수준 (우수한 가공성)
- 절삭력: 70% 수준 (낮은 강도)
```

**신뢰도**: ⭐⭐☆☆☆ (낮음) - 추정 기반

---

## 🎯 **사용 권장사항**

### **즉시 사용 가능 (높은 신뢰도)**
- ✅ **Ti-6Al-4V 모든 데이터**: 실제 논문 기반, 검증 완료
- ✅ **AISI1045 Johnson-Cook 파라미터**: 실험 데이터 기반

### **주의해서 사용 (중간 신뢰도)**  
- ⚠️ **Al2024-T3, SS316L, Inconel718**: 웹 정보 + 물리적 모델링
- 참고용으로 사용하되, 실제 적용 전 검증 필요

### **추가 검증 필요 (낮은 신뢰도)**
- ❌ **AISI4140, Al6061-T6**: 대부분 추정값
- 실제 사용 전 실험적 검증 강력 권장

---

## 📝 **데이터 개선 방안**

### **단기 개선 (1-3개월)**
1. **Al2024-T3**: 항공우주 업계 실험 데이터 수집
2. **SS316L**: 의료/화학 업계 가공 데이터 확보
3. **Inconel718**: 가스터빈 업계 난삭재 데이터 수집

### **중기 개선 (3-6개월)**
1. **직접 실험**: 주요 재료별 절삭 실험 수행
2. **산업체 협력**: 실제 가공 데이터 수집
3. **문헌 조사**: 추가 학술 논문 조사

### **장기 개선 (6-12개월)**
1. **데이터베이스 구축**: 체계적인 가공 데이터베이스
2. **기계학습 모델**: 부족한 데이터 보완
3. **표준화**: 산업 표준 데이터 형식 구축

---

## ⚠️ **중요 경고사항**

### **확장 데이터 사용 시 주의사항**
1. **웹 기반 추정 데이터**는 참고용으로만 사용
2. **실제 생산 적용 전 반드시 실험적 검증** 수행
3. **Ti-6Al-4V 이외 재료**는 예비 연구 목적으로만 사용
4. **Johnson-Cook 파라미터**는 해당 논문의 실험 조건 확인 필요

### **데이터 신뢰도 표시**
- ⭐⭐⭐⭐⭐: 학술 논문 기반 (Ti-6Al-4V)
- ⭐⭐⭐⭐☆: 일부 실험 데이터 (AISI1045 JC 파라미터)
- ⭐⭐⭐☆☆: 웹 정보 + 물리적 모델링
- ⭐⭐☆☆☆: 주로 추정 기반
- ⭐☆☆☆☆: 대부분 추정값

---

**📅 문서 작성일**: 2025년 5월 27일  
**📝 작성자**: SFDP Framework 개발팀  
**🔄 다음 업데이트**: 2025년 8월 (분기별)