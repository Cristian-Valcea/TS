@echo off
echo Activating virtual environment...

call C:\Envs\autogen-env\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate Python 3.10 environment.
    pause
    exit /b 1
)

echo Environment activated: %VIRTUAL_ENV%

@echo off
echo ================================
echo Starting MCP Agent Framework...
echo ================================

REM Start MCP Server
start cmd /k "cd /d ".\MCPServerMain\" && python -m uvicorn MCPServerWS:app --reload"

REM Start Strategy Manager Agent
REM start cmd /k "cd /d ".\MCPServer\" && "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe" StrategyManagerWS.py"

REM Start Trainer Agent
REM start cmd /k "cd /d ".\MCPServer\" && "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe" TrainerAgentWS.py"

REM Start Code Agent
start cmd /k ""python.exe" .\CodeAgent\CodeAgentWS.py"

REM Start Backtester Agent
start cmd /k ""python.exe" "BackTester\BackTesterAgent.py""


REM Start Trader Agent
start cmd /k ""python.exe" "TraderAgent\TraderAgentWS.py""


REM Open Dashboard in browser
start "" ".\dashboard\frontend\unified_uiCopilot.html"

echo All components launched. Close this window to stop.
#pause
