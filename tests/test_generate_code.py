import json
import os
from agents.InstantCodeAgent import generate_code_from_json

# Load a sample strategy JSON file
filename = "strategies/EMA_RSI_Scalper.json"
if not os.path.exists(filename):
    raise FileNotFoundError(f"Strategy file not found: {filename}")

with open(filename, "r") as f:
    strategy_json = json.load(f)

# Generate Backtrader code
code = generate_code_from_json(strategy_json)

# Save to /generated
out_file = "generated/" + strategy_json["strategy_name"].replace(" ", "_") + ".py"
os.makedirs("generated", exist_ok=True)
with open(out_file, "w") as f:
    f.write(code)

print(f"✅ Strategy code saved to: {out_file}")
print("📄 Preview:\n")
print(code[:500] + "\n...\n")
