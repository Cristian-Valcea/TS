@echo off
echo Checking for pip updates...

:: Check pip version
for /f "tokens=*" %%i in ('pip --version') do set PIP_VERSION_INFO=%%i
echo Current %PIP_VERSION_INFO%

pip list > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    pip list | findstr "new release" > nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Updating pip...
        python -m pip install --upgrade pip
        echo Pip has been updated.
    ) else (
        echo Pip is up to date.
    )
) else (
    echo Failed to check pip version.
)

echo.
echo Installing PyTorch packages (with CUDA 12.1)...
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing/updating packages from requirements.txt...
pip install -r requirements.txt

echo.
echo All done!
pause
