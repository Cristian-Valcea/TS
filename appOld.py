import streamlit as st
import os
import json
import re

from agents.StrategyDesignerAgent import design_strategy
from common_logic.intraday import orchestrator
#from common_logic.intraday import intelligent_data_fetcher

from common_logic.intraday.orchestrator import (
    generate_strategy_from_prompt,
    parse_strategy_output,
    generate_code_from_strategy_json,
    run_backtest_from_code
)

st.set_page_config(page_title="Intraday Strategy Chat", layout="wide")

STRATEGY_DIR = "strategies"
os.makedirs(STRATEGY_DIR, exist_ok=True)
os.makedirs("generated", exist_ok=True)

# --- Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "strategy_json" not in st.session_state:
    st.session_state.strategy_json = None

st.title("🤖 Intraday Strategy Designer")

# --- Mode selection
mode = st.radio("Interaction Mode", ["Manual", "AutoGen Orchestrator", "Strategy Generator"])

# --- Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input
user_input = st.chat_input("Type your strategy request or feedback...")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Handle modes
    if mode == "Manual":
        agent_reply = design_strategy(user_input)

    elif mode == "Strategy Generator":
        agent_reply = generate_strategy_from_prompt(user_input)

    else:  # AutoGen Orchestrator
        agent_reply = orchestrator.send_to_ui_user(user_input)

    # Display agent response
    st.session_state.chat.append({"role": "assistant", "content": agent_reply})
    with st.chat_message("assistant"):
        st.markdown(agent_reply)

    # Parse structured output
    markdown, parsed_json = parse_strategy_output(agent_reply)
    if parsed_json:
        st.session_state.strategy_json = parsed_json


# --- Data Fetching ---
with st.expander("📥 Fetch Intraday Data", expanded=False):
    ticker = st.text_input("Ticker (must be in liquid list):", value="AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if st.button("📡 Fetch Data"):
        from common_logic.intraday.intelligent_data_fetcher import fetch_intraday_data
        result = fetch_intraday_data(ticker.upper(), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        st.write(result)
        if "✅" in result:
            st.success("Data saved as data.csv and ready for backtest.")
        else:
            st.error(result)



strategy_json = st.session_state.strategy_json
if strategy_json:
    file_name = strategy_json["strategy_name"].replace(" ", "_") + ".json"
    full_path = os.path.join(STRATEGY_DIR, file_name)

    with st.expander("📋 Strategy JSON", expanded=False):
        st.json(strategy_json)

    if st.button("💾 Save Strategy"):
        try:
            with open(full_path, "w") as f:
                json.dump(strategy_json, f, indent=4)
            st.success(f"✅ Strategy saved to {full_path}")
        except Exception as e:
            st.error(f"❌ Failed to save: {e}")

    if st.button("🧠 Generate Python Code"):
        code = generate_code_from_strategy_json(strategy_json)
        code_file = os.path.join("generated", file_name.replace(".json", ".py"))
        with open(code_file, "w") as f:
            f.write(code)
        st.code(code, language="python")
        st.success(f"📜 Code saved to: {code_file}")

    if st.button("📊 Run Backtest"):
        code_path = os.path.join("generated", file_name.replace(".json", ".py"))
        if os.path.exists(code_path):
            with open(code_path) as f:
                code = f.read()
            result = run_backtest_from_code(code)
            st.text("Backtest Results:")
            st.markdown(result)
            if os.path.exists("sandbox/strategy_plot.png"):
                st.image("sandbox/strategy_plot.png", caption="📈 Entry/Exit Chart")
        else:
            st.warning("⚠️ No strategy code file found. Generate code first.")

    # After backtest result is shown, preview chart if exists
    if os.path.exists("sandbox/strategy_plot.png"):
        st.image("sandbox/strategy_plot.png", caption="📈 Strategy Entry/Exit Overlay")


