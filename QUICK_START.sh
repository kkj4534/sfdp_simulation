#!/bin/bash
# SFDP v17.3 Quick Start Script
# Usage: chmod +x QUICK_START.sh && ./QUICK_START.sh

echo "🏗️  SFDP v17.3 Quick Start Setup"
echo "=================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Python 3 not found. Please install Python 3.12+ first."
    exit 1
}

# Navigate to code directory
cd code/ || {
    echo "❌ Code directory not found. Run this script from validated_17/ directory."
    exit 1
}

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt || {
    echo "⚠️  Dependency installation failed. Trying with --user flag..."
    pip3 install --user -r requirements.txt
}

echo ""
echo "🚀 Running system check..."
python3 -c "
try:
    from modules.sfdp_initialize_system import sfdp_initialize_system
    print('✅ Module imports: OK')
    
    state = sfdp_initialize_system()
    print('✅ System initialization: OK')
    
    print('✅ SFDP v17.3 is ready to use!')
    print('')
    print('📖 Next steps:')
    print('   1. Read USER_GUIDE.md for detailed instructions')
    print('   2. Run: python3 sfdp_v17_3_main.py')
    print('   3. For validation: python3 sfdp_fixed_validation_140.py')
    print('')
    
except Exception as e:
    print(f'❌ Setup error: {e}')
    print('📖 Check USER_GUIDE.md for troubleshooting')
    exit(1)
" 2>/dev/null || {
    echo "❌ System check failed. See USER_GUIDE.md for troubleshooting."
    exit 1
}

echo "🎯 Setup complete! SFDP v17.3 is ready."
echo "📖 See USER_GUIDE.md for detailed usage instructions."