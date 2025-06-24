# agents/reviewer_agent.py

import logging
import sys
from pathlib import Path
import subprocess  # Use subprocess for running external tools

# --- Autogen and Project-Specific Imports ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from config import get_llm_client

logger = logging.getLogger(__name__)

# =================================================================================
# 1. DEFINE THE TOOL FUNCTIONS
# These are the real-world tools the reviewer will use.
# =================================================================================

def analyze_code_with_pylint(file_path: str) -> str:
    """
    Analyzes a Python file using the pylint static analysis tool to check for
    code quality, style violations, and programming errors.

    Args:
        file_path (str): The absolute path to the Python script to be analyzed.

    Returns:
        str: The pylint report. If the score is high, it returns a success message.
             If issues are found, it returns the detailed report for fixing.
    """
    logger.info(f"Reviewer Tool: Running pylint on {file_path}")
    try:
        # We run pylint as a subprocess. We're interested in its output.
        result = subprocess.run(
            ["pylint", file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check if pylint itself ran successfully
        if result.returncode != 0 and "No such file or directory" in result.stderr:
             return f"Error: File not found at path: {file_path}"

        pylint_output = result.stdout
        
        # A simple check: if the output is very short, it likely passed with a good score.
        if len(pylint_output.splitlines()) < 10:
            return "Pylint analysis passed with a good score. No major issues found."
        else:
            # Return the detailed report for the CodeAgent to fix.
            return f"Pylint found issues that need fixing:\n\n{pylint_output}"

    except FileNotFoundError:
        return "Error: 'pylint' command not found. Please ensure pylint is installed."
    except Exception as e:
        return f"An unexpected error occurred during pylint analysis: {e}"

def analyze_code_with_bandit(file_path: str) -> str:
    """
    Analyzes a Python file using the bandit security linter to find common
    security vulnerabilities.

    Args:
        file_path (str): The absolute path to the Python script to be analyzed.

    Returns:
        str: The bandit security report. Returns a success message if no issues
             are found, otherwise returns the list of potential vulnerabilities.
    """
    logger.info(f"Reviewer Tool: Running bandit on {file_path}")
    try:
        result = subprocess.run(
            ["bandit", "-r", file_path, "-f", "txt"], # '-f txt' for plain text output
            capture_output=True,
            text=True,
            timeout=60
        )

        if "No such file or directory" in result.stderr:
            return f"Error: File not found at path: {file_path}"

        bandit_output = result.stdout
        
        # Bandit reports "No issues found" on success.
        if "No issues found." in bandit_output:
            return "Bandit security analysis passed. No vulnerabilities found."
        else:
            return f"Bandit found potential security issues:\n\n{bandit_output}"
            
    except FileNotFoundError:
        return "Error: 'bandit' command not found. Please ensure bandit is installed."
    except Exception as e:
        return f"An unexpected error occurred during bandit analysis: {e}"

# =================================================================================
# 2. WRAP THE FUNCTIONS IN FunctionTool
# =================================================================================

pylint_tool = FunctionTool(analyze_code_with_pylint,description="lint corrections", name="analyze_code_with_pylint")
bandit_tool = FunctionTool(analyze_code_with_bandit,description="bandit corrections", name="analyze_code_with_bandit")

tools_for_reviewer = [pylint_tool, bandit_tool]

# =================================================================================
# 3. DEFINE THE AGENT'S SYSTEM MESSAGE
# =================================================================================

reviewer_system_message = """You are a meticulous Code Reviewer Agent. Your sole purpose is to ensure code quality and security before it is backtested.

**CRITICAL WORKFLOW:**
1. You will be given a file path to a Python script by the CodeAgent.
2. You MUST use your `analyze_code_with_pylint` tool on the provided file path.
3. You MUST use your `analyze_code_with_bandit` tool on the provided file path.
4. **Decision Time:**
   - **If BOTH tools pass** (return success messages like "No issues found"), your final response MUST be: "Code review passed. **BacktesterAgent**, you are cleared to proceed with testing the script at [file_path]."
   - **If EITHER tool finds issues**, your final response MUST be: "**CodeAgent**, the script at [file_path] failed review. Please fix the following issues:\n\n[Combine the reports from pylint and bandit here]."
5. Do not write or fix code yourself. Your only job is to run the analysis tools and report the results to the correct agent.
"""

# =================================================================================
# 4. INITIALIZE AND EXPORT THE AGENT
# =================================================================================

logger.info("Attempting to configure and initialize ReviewerAgent...")

init_kwargs = {
    "name": "ReviewerAgent",
    "description": "Reviews Python code for quality and security using pylint and bandit tools.",
    "system_message": reviewer_system_message,
    "tools": tools_for_reviewer
}

# The ReviewerAgent needs to call tools, so it should use a capable model like Gemini.
llm_client = get_llm_client("ReviewerAgent") # Assumes a "ReviewerAgent" entry in config.py

if llm_client is None:
    raise ValueError("ReviewerAgent LLM client configuration failed and is None.")
else:
    init_kwargs["model_client"] = llm_client

try:
    reviewer_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ ReviewerAgent '{reviewer_agent.name}' initialized successfully.")
except Exception as e:
    logger.error(f"❌ FAILED to initialize ReviewerAgent: {e}", exc_info=True)
    raise


