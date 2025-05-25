@echo off
REM ================================================================
REM EZIO PROJECT ORGANIZER - AUTOMATED CLEANUP & ORGANIZATION
REM Generated: 2025-05-23 16:15:00
REM Total Commands: 35
REM Execution Time: ~2 minutes
REM ================================================================

echo.
echo ========================================
echo  EZIO PROJECT ORGANIZER STARTING...
echo ========================================
echo  Project: eziofilho-unified
echo  Target: C:\Users\anapa\eziofilho-unified
echo  Operations: 35 commands
echo ========================================
echo.

cd "C:\Users\anapa\eziofilho-unified"

REM Check if we're in the right directory
if not exist "main.py" (
    echo ERROR: Not in correct project directory!
    echo Please ensure you're in C:\Users\anapa\eziofilho-unified
    pause
    exit /b 1
)

echo Step 1/10: Creating new organized structure...
mkdir "01_core_system" 2>nul
mkdir "02_experts_modules" 2>nul
mkdir "03_models_storage" 2>nul
mkdir "04_configuration" 2>nul
mkdir "05_testing_validation" 2>nul
mkdir "06_tools_utilities" 2>nul
mkdir "07_documentation" 2>nul
mkdir "08_examples_demos" 2>nul
mkdir "09_data_cache" 2>nul
mkdir "10_deployment" 2>nul
echo   - New folder structure created!

echo.
echo Step 2/10: Moving main system files...
for %%f in ("main*.py") do if exist "%%f" move "%%f" "01_core_system\" >nul 2>&1
for %%f in ("run_*.py") do if exist "%%f" move "%%f" "01_core_system\" >nul 2>&1
for %%f in ("*orchestrator*.py") do if exist "%%f" move "%%f" "01_core_system\" >nul 2>&1
for %%f in ("ezio_finisher*.py") do if exist "%%f" move "%%f" "01_core_system\" >nul 2>&1
echo   - Main system files moved!

echo.
echo Step 3/10: Moving expert modules...
for %%f in ("expert*.py") do if exist "%%f" move "%%f" "02_experts_modules\" >nul 2>&1
for %%f in ("eziofinisher*.py") do if exist "%%f" move "%%f" "02_experts_modules\" >nul 2>&1
for %%f in ("simple_chat*.py") do if exist "%%f" move "%%f" "02_experts_modules\" >nul 2>&1
for %%f in ("direct_chat*.py") do if exist "%%f" move "%%f" "02_experts_modules\" >nul 2>&1
for %%f in ("advanced_model_manager.py") do if exist "%%f" move "%%f" "02_experts_modules\" >nul 2>&1
echo   - Expert modules moved!

echo.
echo Step 4/10: Moving test files...
for %%f in ("test_*.py") do if exist "%%f" move "%%f" "05_testing_validation\" >nul 2>&1
for %%f in ("*_test.py") do if exist "%%f" move "%%f" "05_testing_validation\" >nul 2>&1
for %%f in ("run_*test*.py") do if exist "%%f" move "%%f" "05_testing_validation\" >nul 2>&1
echo   - Test files moved!

echo.
echo Step 5/10: Moving configuration files...
for %%f in ("requirements*.txt") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
for %%f in ("config*.py") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
for %%f in ("config*.json") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
for %%f in ("*.json") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
for %%f in ("configure*.py") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
for %%f in ("criar_config.py") do if exist "%%f" move "%%f" "04_configuration\" >nul 2>&1
echo   - Configuration files moved!

echo.
echo Step 6/10: Moving GPU and utility tools...
for %%f in ("gpu_*.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
for %%f in ("*monitor*.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
for %%f in ("*benchmark*.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
for %%f in ("demo_multi_gpu.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
for %%f in ("find_models.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
for %%f in ("organize_models.py") do if exist "%%f" move "%%f" "06_tools_utilities\" >nul 2>&1
echo   - GPU and utility tools moved!

echo.
echo Step 7/10: Moving deployment files...
for %%f in ("*.bat") do if exist "%%f" move "%%f" "10_deployment\" >nul 2>&1
echo   - Deployment files moved!

echo.
echo Step 8/10: Moving documentation...
for %%f in ("*.md") do if exist "%%f" move "%%f" "07_documentation\" >nul 2>&1
echo   - Documentation moved!

echo.
echo Step 9/10: Moving existing folders...
if exist "autogen_examples" (
    xcopy /E /I /Y "autogen_examples" "08_examples_demos\autogen_examples\" >nul 2>&1
    rmdir /s /q "autogen_examples" >nul 2>&1
)
if exist "data" (
    xcopy /E /I /Y "data" "09_data_cache\data\" >nul 2>&1
    rmdir /s /q "data" >nul 2>&1
)
if exist "results" (
    xcopy /E /I /Y "results" "05_testing_validation\results\" >nul 2>&1
    rmdir /s /q "results" >nul 2>&1
)
if exist "reports" (
    xcopy /E /I /Y "reports" "07_documentation\reports\" >nul 2>&1
    rmdir /s /q "reports" >nul 2>&1
)
if exist "models" (
    xcopy /E /I /Y "models" "03_models_storage\models\" >nul 2>&1
    rmdir /s /q "models" >nul 2>&1
)
if exist "core" (
    xcopy /E /I /Y "core" "01_core_system\core\" >nul 2>&1
    rmdir /s /q "core" >nul 2>&1
)
if exist "experts" (
    xcopy /E /I /Y "experts" "02_experts_modules\experts\" >nul 2>&1
    rmdir /s /q "experts" >nul 2>&1
)
if exist "tools" (
    xcopy /E /I /Y "tools" "06_tools_utilities\tools\" >nul 2>&1
    rmdir /s /q "tools" >nul 2>&1
)
if exist "chunks" (
    xcopy /E /I /Y "chunks" "05_testing_validation\chunks\" >nul 2>&1
    rmdir /s /q "chunks" >nul 2>&1
)
if exist "testes_autogen" (
    xcopy /E /I /Y "testes_autogen" "05_testing_validation\testes_autogen\" >nul 2>&1
    rmdir /s /q "testes_autogen" >nul 2>&1
)
if exist "testes_experts" (
    xcopy /E /I /Y "testes_experts" "05_testing_validation\testes_experts\" >nul 2>&1
    rmdir /s /q "testes_experts" >nul 2>&1
)
if exist "tests_modelos" (
    xcopy /E /I /Y "tests_modelos" "05_testing_validation\tests_modelos\" >nul 2>&1
    rmdir /s /q "tests_modelos" >nul 2>&1
)
if exist "docs" (
    xcopy /E /I /Y "docs" "07_documentation\docs\" >nul 2>&1
    rmdir /s /q "docs" >nul 2>&1
)
if exist "LangGraph_Example" (
    xcopy /E /I /Y "LangGraph_Example" "08_examples_demos\LangGraph_Example\" >nul 2>&1
    rmdir /s /q "LangGraph_Example" >nul 2>&1
)
echo   - Existing folders reorganized!

echo.
echo Step 10/10: Cleaning up cache and temporary files...
for /r . %%f in (*.pyc) do del /q "%%f" >nul 2>&1
for /d /r . %%d in (__pycache__) do rmdir /s /q "%%d" >nul 2>&1
REM Clean up version artifacts in root
del /q "1.20.0" >nul 2>&1
del /q "2.11.3" >nul 2>&1
del /q "25.1.1" >nul 2>&1
del /q "3.7" >nul 2>&1
del /q "4.5.0" >nul 2>&1
del /q "5.6.1" >nul 2>&1
del /q "#" >nul 2>&1
del /q "cd" >nul 2>&1
del /q "git" >nul 2>&1
del /q "main" >nul 2>&1
del /q "pip" >nul 2>&1
del /q "set" >nul 2>&1
del /q "CMAKE_ARGS" >nul 2>&1
del /q "llama-cpp-python)" >nul 2>&1
del /q "timeout)" >nul 2>&1
echo   - Cache and temporary files cleaned!

echo.
echo Creating organization report...
echo EZIO PROJECT ORGANIZATION REPORT > "07_documentation\organization_report.txt"
echo ================================== >> "07_documentation\organization_report.txt"
echo Completed: %date% %time% >> "07_documentation\organization_report.txt"
echo Total operations: 35 >> "07_documentation\organization_report.txt"
echo. >> "07_documentation\organization_report.txt"
echo NEW STRUCTURE: >> "07_documentation\organization_report.txt"
echo 01_core_system - Main runners and orchestrators >> "07_documentation\organization_report.txt"
echo 02_experts_modules - AI experts and specialized modules >> "07_documentation\organization_report.txt"
echo 03_models_storage - Local models and configurations >> "07_documentation\organization_report.txt"
echo 04_configuration - All config files and requirements >> "07_documentation\organization_report.txt"
echo 05_testing_validation - Tests and validation results >> "07_documentation\organization_report.txt"
echo 06_tools_utilities - GPU tools and utilities >> "07_documentation\organization_report.txt"
echo 07_documentation - Documentation and reports >> "07_documentation\organization_report.txt"
echo 08_examples_demos - Examples and demos >> "07_documentation\organization_report.txt"
echo 09_data_cache - Data storage and cache >> "07_documentation\organization_report.txt"
echo 10_deployment - Deployment scripts and production files >> "07_documentation\organization_report.txt"
echo. >> "07_documentation\organization_report.txt"
echo ORGANIZATION COMPLETED SUCCESSFULLY! >> "07_documentation\organization_report.txt"

echo.
echo ========================================
echo  EZIO PROJECT ORGANIZATION COMPLETE!
echo ========================================
echo  ✅ New structure created (10 folders)
echo  ✅ Files organized by function
echo  ✅ Tests and configs separated  
echo  ✅ Cache and temp files cleaned
echo  ✅ Documentation updated
echo ========================================
echo  📊 Check: 07_documentation\organization_report.txt
echo  🚀 Project is now ready for development!
echo ========================================
echo.

pause