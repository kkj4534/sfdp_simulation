# Contributing to SFDP v17.3

## 🎯 Overview

Thank you for your interest in contributing to the Smart Fusion-based Dynamic Prediction (SFDP) framework! This document provides guidelines for contributing to the validated Ti-6Al-4V machining simulation system.

## 🚨 Critical Requirements

### 1. Physics Integrity
- **NO artificial boosters or multipliers** - All calculations must be physics-based
- **NO synthetic data generation** - Use only real experimental data
- **NO correction factors > 1.0** - Unless scientifically justified
- **Validate against experimental data** - Target ≤15% error

### 2. Code Quality
- Maintain existing coding style and patterns
- Follow the 6-layer architecture principles
- Include proper error handling and logging
- Document all changes with scientific justification

### 3. Validation Requirements
- Run full 150-iteration validation before submission
- Achieve target performance metrics (≤15% error)
- Document performance impact of changes
- Maintain or improve system convergence

## 📋 Contribution Process

### 1. Before Starting
1. Read the [User Guide](./USER_GUIDE.md) and [White Paper](./docs/WHITE_PAPER.md)
2. Understand the current system performance (10.634% ± 1.820%)
3. Identify the specific improvement or fix needed
4. Check existing issues and documentation

### 2. Development Guidelines

#### Code Structure
```python
# Follow existing patterns:
def your_function(param1, param2, config=None):
    """
    Brief description of function.
    
    Args:
        param1: Description
        param2: Description 
        config: Optional configuration dict
        
    Returns:
        Expected return format
        
    Raises:
        SpecificException: When this occurs
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Error in your_function: {e}")
        raise
```

#### Physics Calculations
- Use established formulas (Kienzle, Carslaw & Jaeger, etc.)
- Include proper units and dimensional analysis
- Validate against known experimental results
- Document all assumptions clearly

#### Data Handling
- Use only validated experimental datasets
- Implement proper data validation checks
- Handle missing data gracefully
- Maintain data traceability

### 3. Testing Requirements

#### Validation Testing
```bash
# Run full validation suite
cd code/
python sfdp_fixed_validation_140.py

# Run continuous tuning
python sfdp_continuous_tuning_150.py

# Verify performance metrics
python sfdp_v17_3_main.py
```

#### Expected Results
- Validation error: ≤15% (target)
- Success rate: >90%
- System convergence: Stable performance
- No physics violations: Conservation laws maintained

### 4. Submission Process

#### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Physics Validation
- [ ] No artificial boosters added
- [ ] Uses real experimental data only
- [ ] Maintains physics laws compliance
- [ ] Validation error ≤15%

## Testing
- [ ] 150-iteration validation completed
- [ ] Performance metrics documented
- [ ] No regression in system performance
- [ ] All existing tests pass

## Documentation
- [ ] Code comments updated
- [ ] User guide updated (if needed)
- [ ] API reference updated (if needed)
- [ ] Performance impact documented
```

## 🔍 Code Review Criteria

### Automatic Checks
1. **Physics Integrity**: No artificial performance boosters
2. **Data Authenticity**: Only real experimental data used
3. **Performance**: Maintains ≤15% validation error
4. **Convergence**: System remains stable

### Manual Review
1. **Code Quality**: Follows existing patterns and style
2. **Documentation**: Clear and comprehensive
3. **Testing**: Adequate test coverage
4. **Impact**: Positive contribution to system performance

## 🐛 Bug Reports

### Issue Template
```markdown
**Bug Description**
Clear description of the bug

**Environment**
- Python version:
- OS:
- Dependencies:

**Reproduction Steps**
1. Step 1
2. Step 2
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Validation Results**
Current system performance metrics

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Enhancement Guidelines
1. **Scientific Justification**: Based on research or experimental evidence
2. **Performance Impact**: Must not degrade existing performance
3. **Implementation Plan**: Clear development approach
4. **Validation Strategy**: How to verify the enhancement works

### Priority Areas
1. **Layer 1 Enhancements**: Advanced FEM implementations
2. **Material Support**: Additional alloy types
3. **Tool Modeling**: Enhanced tool wear predictions
4. **Optimization**: Performance improvements
5. **Validation**: Enhanced experimental data integration

## 📚 Resources

### Technical Documentation
- [White Paper](./docs/WHITE_PAPER.md): Implementation overview
- [API Reference](./docs/API_REFERENCE.md): Detailed API documentation
- [MATLAB Technical Whitepaper](https://github.com/your-username/sfdp_simulation/tree/main/sfdp_old_versions/sfdp_ver_17.3.1_ver.matlab/docs/technical_whitepaper_chapters/): Theoretical foundations

### Validation Data
- [Experimental Datasets](./data/): Real Ti-6Al-4V machining data
- [Extended References](./reference/): Additional validation datasets
- [Performance Logs](./tuning_logs/): Historical performance tracking

## 🤝 Community Guidelines

### Be Respectful
- Professional and constructive communication
- Focus on technical merit and scientific accuracy
- Acknowledge contributions from others

### Be Collaborative
- Share knowledge and expertise
- Help others understand complex concepts
- Provide constructive feedback

### Be Scientific
- Base arguments on data and evidence
- Cite relevant literature and experiments
- Maintain objectivity in discussions

## 📞 Contact

For questions about contributing:
- **Technical Issues**: Create a GitHub issue
- **General Questions**: memento1087@gmail.com
- **Physics/Theory**: Reference the [MATLAB technical documentation](https://github.com/your-username/sfdp_simulation/tree/main/sfdp_old_versions/sfdp_ver_17.3.1_ver.matlab/docs/technical_whitepaper_chapters/)

---

**Remember**: The goal is to maintain and improve a legitimate, physics-based simulation system that advances the field of machining prediction for Ti-6Al-4V materials.