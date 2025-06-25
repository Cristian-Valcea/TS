# Trading System Toolkit (TS)

MyTS is a Python-based framework for designing, testing, and deploying trading strategies. It integrates Large Language Models (LLMs) for strategy generation and utilizes robust backtesting and data handling capabilities.

## Project Overview

This project provides a suite of tools for quantitative trading, including:

-   **Strategy Design & Generation**:
    -   A Streamlit web application (`app.py`) for manually designing strategies or using LLMs to generate strategy ideas and code.
    -   Agent-based system (`main_autogen.py` and `agents/`) that uses an `OrchestratorAgent` to manage a team of specialized agents (e.g., `CodeAgent`, `DataProvisioningAgent`, `ReviewerAgent`, `BacktesterAgent`) for an automated strategy development pipeline.
-   **Data Handling**:
    -   Fetching historical market data using `yfinance` (via `agents/data_provisioning_agent.py` and `common_logic/intraday/intelligent_data_fetcher.py`).
    -   Support for fetching data from Interactive Brokers (via `tools/GetIBKRData.py`).
    -   Data cleaning and validation processes.
-   **Backtesting**:
    -   A core backtesting engine built on the Backtrader library (`common_logic/backtesting_services/backtrader_runner.py`).
    -   Calculation of various performance metrics (Sharpe Ratio, PnL, Drawdown, etc.).
-   **Optimization**:
    -   Bayesian optimization of strategy parameters (facilitated by `agents/bay_agent.py`).
-   **Configuration**:
    -   Centralized configuration for API keys, file paths, and LLM model mappings (`config.py`).

## Key Components

-   **`app.py`**: Streamlit web interface for interactive strategy design, data fetching, and backtesting.
-   **`main_autogen.py`**: Main entry point for the automated, agent-based strategy development workflow.
-   **`config.py`**: Central configuration file for the system.
-   **`agents/`**: Contains the definitions for various autonomous agents:
    -   `OrchestratorAgent`: Manages the overall strategy development workflow.
    -   `DataProvisioningAgent`: Fetches and saves market data.
    -   `CodeAgent`: Generates strategy code based on prompts and schemas.
    -   `ReviewerAgent`: Reviews generated code for quality and security.
    -   `BacktesterAgent`: Executes backtests using the Backtrader engine.
    -   `BayAgent`: Performs Bayesian optimization of strategy parameters.
    -   Other specialized agents for UI interaction, strategy design, etc.
-   **`common_logic/`**: Houses shared business logic:
    -   `backtesting_services/`: Core Backtrader integration and runner.
    -   `data_provisioning/`: Modules related to data fetching and preparation (though some yfinance logic is also in `agents` or `intraday`).
    -   `intraday/`: Logic specific to intraday data handling and strategy generation, used by `app.py`.
    -   `strategies/`: (Typically) Stores strategy code or templates.
    -   `templates/`: Strategy code templates.
-   **`tools/`**: Utility scripts, including the Interactive Brokers data fetching tool.
-   **`shared_work_dir/`**: Default directory for storing data, generated strategies, and other artifacts.
-   **`tests/`**: Unit tests for critical components.

## Getting Started

(TODO: Add detailed setup instructions, dependencies, and how to run the application/agent system.)

1.  **Prerequisites**:
    -   Python 3.8+
    -   IBKR TWS or Gateway (if using `tools/GetIBKRData.py`)
    -   API keys for LLM providers (e.g., Gemini) set as environment variables (see `config.py`).
2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    # (May need separate installs for specific features like ib_insync or autogen variants)
    ```
3.  **Running the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
4.  **Running the Automated Agent System**:
    ```bash
    python main_autogen.py
    ```

## Refactoring Notes

This repository has undergone significant refactoring to:
-   Improve code readability and organization.
-   Enhance robustness through better error handling and data validation.
-   Standardize file paths and configurations.
-   Remove duplicated and obsolete files.
-   Add unit tests for critical components.

Further improvements can include:
-   More comprehensive unit and integration testing.
-   Detailed setup and usage instructions.
-   Parsing strategy parameters (`strategy_name`, `ticker`, etc.) from the `initial_task` in `OrchestratorAgent` instead of hardcoding.
-   Consolidating yfinance data fetching logic into a shared utility.
