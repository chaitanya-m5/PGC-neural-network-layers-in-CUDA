# This file compiles your neural_network.cu source code and generates neural_network.exe.

@echo off
echo ==========================================
echo   Experiment 5 - CUDA Build Script
echo ==========================================

echo.
echo Checking CUDA compiler...

where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: NVCC not found.
    echo Please install CUDA Toolkit and add NVCC to PATH.
    pause
    exit /b 1
)

echo CUDA compiler found.

echo.
echo Compiling neural_network.cu...

nvcc -O2 neural_network.cu -o neural_network.exe

if errorlevel 1 (
    echo.
    echo BUILD FAILED!
    pause
    exit /b 1
)

echo.
echo BUILD SUCCESSFUL!
echo Executable created: neural_network.exe

pause
