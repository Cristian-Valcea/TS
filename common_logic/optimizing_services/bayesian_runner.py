"""

Example CLI Call:
python common_logic/optimizing_services/bayesian_runner.py \
    --strategy-file shared_work_dir/strategies/sma_cross_strategy.py \
    --data-file shared_work_dir/data/AAPL_2020-01-01_2023-12-31.csv \
    --strategy-class-name SmaCrossover \
    --param-space '{"fast_ma": [10, 50], "slow_ma": [100, 200]}' \
    --n-iter 20 \
    --init-points 5 \
    --objective "sharpe_ratio"
"""


"""
Bayesian Optimizer Runner Script.

This script performs hyperparameter optimization for a Backtrader strategy
using the 'bayesian-optimization' library. It calls the existing
backtrader_runner.py script to evaluate each parameter set.
"""
import argparse
import json
import subprocess
import sys
import logging
from bayes_opt import BayesianOptimization
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BayesianRunner")

# --- Robust Pathing to find the backtrader runner ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    BACKTRADER_RUNNER_PATH = PROJECT_ROOT / "common_logic" / "backtesting_services" / "backtrader_runner.py"
    if not BACKTRADER_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Could not find backtrader_runner.py at: {BACKTRADER_RUNNER_PATH}")
except Exception as e:
    logger.error(f"Critical pathing error: {e}", exc_info=True)
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Bayesian Optimization strategy optimizer.")
    parser.add_argument("--strategy-file", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--strategy-class-name", required=True)
    parser.add_argument("--param-space", required=True, help="JSON string defining parameter bounds. e.g., '{\"param\": [min, max]}'")
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--init-points", type=int, default=5)
    parser.add_argument("--objective", type=str, default="sharpe_ratio")
    return parser.parse_args()

def run_backtest_evaluation(args, params_json):
    """Calls the backtrader_runner.py script and returns the parsed JSON results."""
    command = [
        sys.executable,
        str(BACKTRADER_RUNNER_PATH),
        "--strategy-file", args.strategy_file,
        "--data-file", args.data_file,
        "--strategy-class-name", args.strategy_class_name,
        "--strategy-params", params_json,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=str(PROJECT_ROOT))
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.error(f"Backtest evaluation failed for params {params_json}. Error: {e}")
        if hasattr(e, 'stderr'):
            logger.error(f"Stderr: {e.stderr.strip()}")
        return None

def create_black_box_function(args):
    """Creates the objective function that the BayesianOptimizer will call."""
    def black_box_function(**params):
        int_params = {key: round(value) for key, value in params.items()}
        params_json = json.dumps(int_params)
        
        logger.info(f"Evaluating with params: {params_json}")
        results = run_backtest_evaluation(args, params_json)
        
        if results and results.get("status") == "completed" and args.objective in results:
            objective_value = results[args.objective]
            return float(objective_value) if objective_value is not None else -999.0
        else:
            return -999.0 # Return a very poor value for failed backtests

    return black_box_function

def main():
    """Main execution function."""
    args = parse_arguments()
    param_space = json.loads(args.param_space)
    
    logger.info(f"Starting Bayesian Optimization with {args.n_iter} iterations...")
    logger.info(f"Objective: maximize '{args.objective}'")

    optimizer = BayesianOptimization(
        f=create_black_box_function(args),
        pbounds=param_space,
        random_state=1,
        verbose=2
    )
    
    optimizer.maximize(init_points=args.init_points, n_iter=args.n_iter)
    
    logger.info("Bayesian Optimization complete.")

    final_result = {
        "status": "success",
        "best_value": optimizer.max['target'],
        "best_params": {key: round(value) for key, value in optimizer.max['params'].items()},
    }
    
    print(json.dumps(final_result, indent=2))

if __name__ == "__main__":
    main()