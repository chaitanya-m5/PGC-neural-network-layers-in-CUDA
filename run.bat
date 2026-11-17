# This file executes the compiled CUDA program.

@echo off
echo ==========================================
echo   Experiment 5 - CUDA Execution
echo ==========================================

if not exist neural_network.exe (
    echo ERROR: neural_network.exe not found.
    echo Please run build.bat first.
    pause
    exit /b 1
)

echo.
echo Running CUDA Neural Network Experiment...
echo.

neural_network.exe

echo.
echo ==========================================
echo   Execution Finished
echo ==========================================

pause
