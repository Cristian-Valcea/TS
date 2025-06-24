@echo off
SET VENV_NAME=venv
SET PY310_PATH=C:\Python310\python.exe

echo -----------------------------------------
echo Checking if virtual environment exists...
echo -----------------------------------------

IF EXIST "%VENV_NAME%\Scripts\activate.bat" (
    echo Virtual environment "%VENV_NAME%" already exists.
) ELSE (
    echo Creating virtual environment: %VENV_NAME%
    "%PY310_PATH%" -m venv %VENV_NAME%
)

echo.
echo -----------------------------------------
echo Activating virtual environment
echo -----------------------------------------
call %VENV_NAME%\Scripts\activate

echo.
echo -----------------------------------------
echo Upgrading pip
echo -----------------------------------------
"%PY310_PATH%" -m pip install --upgrade pip

echo.
echo -----------------------------------------
echo Installing all packages (with PyTorch CUDA 12.1)
echo -----------------------------------------
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

echo.
echo -----------------------------------------
echo Testing PyTorch GPU availability
echo -----------------------------------------
"%PY310_PATH%" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo -----------------------------------------
echo Environment setup complete!
echo -----------------------------------------
pause
