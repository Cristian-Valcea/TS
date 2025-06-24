
import streamlit as st
import os
import re
from agents.StrategyDesignerAgent import design_strategy
from common_logic.intraday.orchestrator import (
    generate_strategy_from_prompt,
    parse_strategy_output,
    generate_code_from_strategy_json,
    run_backtest_from_code,
    run_backtest_from_file,
)
from common_logic.intraday.intelligent_data_fetcher import fetch_intraday_period
import datetime

st.set_page_config(page_title="Intraday Strategy Chat", layout="wide")
st.title("🧠 Intraday Strategy Designer")

# Initialize session state
if "strategy_json" not in st.session_state:
    st.session_state.strategy_json = None
if "strategy_code" not in st.session_state:
    st.session_state.strategy_code = None
if "code_file" not in st.session_state:
    st.session_state.code_file = None
if "data_valid" not in st.session_state:
    st.session_state.data_valid = False

mode = st.selectbox("Choose mode", ["Manual", "Strategy Generator"])

if mode == "Manual":
    user_input = st.text_area("Paste your strategy prompt")
    if st.button("➕ Design Strategy"):
        agent_reply = design_strategy(user_input)
        print(agent_reply)
        markdown, parsed_json = parse_strategy_output(agent_reply)
        st.markdown("### Strategy Description")
        st.markdown(markdown)
        st.session_state.strategy_json = parsed_json

elif mode == "Strategy Generator":
    user_input = st.text_area("Describe your strategy")
    if st.button("⚙️ Generate"):
        reply = generate_strategy_from_prompt(user_input)
        markdown, parsed_json = parse_strategy_output(reply)
        st.markdown("### Strategy Description")
        st.markdown(markdown)
        st.session_state.strategy_json = parsed_json

# Save JSON
if st.session_state.strategy_json:
    file_name = st.text_input("Save as", value="strategy.json")
    if st.button("💾 Save Strategy JSON"):
        try:
            os.makedirs("strategies", exist_ok=True)
            with open(os.path.join("strategies", file_name), "w") as f:
                import json
                json.dump(st.session_state.strategy_json, f, indent=2)
            st.success(f"✅ Saved as {file_name}")
        except Exception as e:
            st.error(f"❌ Failed to save: {e}")

# Generate Code
if st.session_state.strategy_json:
    if st.button("🧠 Generate Python Code"):
        raw_code = generate_code_from_strategy_json(st.session_state.strategy_json)
        # strip code fences
        cleaned = re.sub(r"```(?:python)?\n", "", raw_code)
        cleaned = re.sub(r"\n```$", "", cleaned)
        st.session_state.strategy_code = cleaned

        # extract class name for filename
        m = re.search(r"class\s+(\w+)\s*\(", cleaned)
        if m:
            class_name = m.group(1)
        else:
            class_name = os.path.splitext(file_name)[0]
        code_file = os.path.join("generated", f"{class_name}.py")
        os.makedirs("generated", exist_ok=True)
        with open(code_file, "w") as f:
            f.write(cleaned)
        st.session_state.code_file = code_file
        st.success(f"📜 Code saved to {code_file}")
        st.code(cleaned)

# --- Data Fetching ---
with st.expander("📥 Fetch Intraday Data", expanded=False):
    ticker = st.text_input("Ticker (must be in liquid list):", value="AAPL")
    period = st.selectbox("Select Period", options=["5d", "15d", "30d"], index=0)
    force_save = st.checkbox("⚠️ Force save even if data is incomplete")
    include_extended = st.checkbox("🕒 Include extended-hours data")

    if st.button("📡 Fetch Data"):
        result = fetch_intraday_period(
            ticker.upper(),
            period=period,
            max_missing_days=5 if not force_save else 1000
        ) if not include_extended else fetch_intraday_period(
            ticker.upper(),
            period=period,
            max_missing_days=5 if not force_save else 1000,
            include_extended_hours=True
        )
        st.markdown("### 🧾 Data Quality Report")
        st.code(result)
        st.session_state.data_valid = result.startswith("🕵️") and "✅" in result

# Run Backtest
if st.session_state.get("data_valid") and st.session_state.get("strategy_code"):
    if st.button("📊 Run Backtest"):
        strategy_path = os.path.join("generated", file_name.replace(".json", ".py"))
        print(strategy_path)
        print(st.session_state.code_file)
        result = run_backtest_from_file(st.session_state.code_file)#strategy_path)
        st.text("Backtest Results:")
        st.markdown(result)

        if os.path.exists("sandbox/strategy_plot.png"):
            st.image("sandbox/strategy_plot.png", caption="📈 Strategy Entry/Exit Overlay")
else:
    st.warning("⚠️ Please generate a strategy, code, and fetch valid data before backtesting.")
