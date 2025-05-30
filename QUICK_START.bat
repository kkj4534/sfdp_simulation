@echo off
REM SFDP v17.3 Quick Start Script for Windows
REM Usage: Double-click this file or run from command prompt

echo 🏗️  SFDP v17.3 Quick Start Setup
echo ==================================

REM Check Python version
echo 🐍 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.12+ first.
    pause
    exit /b 1
)

REM Navigate to code directory
cd code
if errorlevel 1 (
    echo ❌ Code directory not found. Run this script from validated_17\ directory.
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️  Dependency installation failed. Trying with --user flag...
    pip install --user -r requirements.txt
)

echo.
echo 🚀 Running system check...
python -c "try: from modules.sfdp_initialize_system import sfdp_initialize_system; print('✅ Module imports: OK'); state = sfdp_initialize_system(); print('✅ System initialization: OK'); print('✅ SFDP v17.3 is ready to use!'); print(''); print('📖 Next steps:'); print('   1. Read USER_GUIDE.md for detailed instructions'); print('   2. Run: python sfdp_v17_3_main.py'); print('   3. For validation: python sfdp_fixed_validation_140.py'); print(''); except Exception as e: print(f'❌ Setup error: {e}'); print('📖 Check USER_GUIDE.md for troubleshooting'); exit(1)" 2>nul
if errorlevel 1 (
    echo ❌ System check failed. See USER_GUIDE.md for troubleshooting.
    pause
    exit /b 1
)

echo 🎯 Setup complete! SFDP v17.3 is ready.
echo 📖 See USER_GUIDE.md for detailed usage instructions.
echo.
echo Press any key to continue...
pause >nul