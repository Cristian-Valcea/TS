import streamlit as st
import os, re, json
from datetime import datetime
from agents.StrategyDesignerAgent import design_strategy
from common_logic.intraday.orchestrator import (
    generate_strategy_from_prompt,
    parse_strategy_output,
    generate_code_from_strategy_json,
    run_backtest_from_code,
    run_backtest_from_file,
)
from common_logic.intraday.intelligent_data_fetcher import fetch_intraday_period
from agents.StrategyRaterAgent import auto_rate_strategy

st.set_page_config(page_title="Intraday Strategy Chat", layout="wide")
st.title("🧠 Intraday Strategy Designer")

if "strategy_json" not in st.session_state:
    st.session_state.strategy_json = None
if "strategy_code" not in st.session_state:
    st.session_state.strategy_code = None
if "code_file" not in st.session_state:
    st.session_state.code_file = None
if "data_valid" not in st.session_state:
    st.session_state.data_valid = False

# --- Mode selector
mode = st.selectbox("Choose mode", ["Manual", "Strategy Generator"])

# --- Strategy design
if mode == "Manual":
    user_input = st.text_area("Paste your strategy prompt")
    if st.button("➕ Design Strategy"):
        reply = design_strategy(user_input)
        md, parsed = parse_strategy_output(reply)
        st.markdown("### Strategy Description")
        st.markdown(md)
        st.session_state.strategy_json = parsed

else:
    user_input = st.text_area("Describe your strategy")
    if st.button("⚙️ Generate"):
        reply = generate_strategy_from_prompt(user_input)
        md, parsed = parse_strategy_output(reply)
        st.markdown("### Strategy Description")
        st.markdown(md)
        st.session_state.strategy_json = parsed

# --- Save JSON
if st.session_state.strategy_json:
    file_name = st.text_input("Save as", value="strategy.json")
    if st.button("💾 Save Strategy JSON"):
        os.makedirs("strategies", exist_ok=True)
        with open(os.path.join("strategies", file_name), "w") as f:
            json.dump(st.session_state.strategy_json, f, indent=2)
        st.success(f"✅ Saved as {file_name}")


# ---------------------------------------------------
#  ▶️ Manual & Auto-Rating UI
# ---------------------------------------------------
st.subheader("📝 Rate This Strategy")

# manual sliders
rr = st.slider("Risk/Reward", 1, 5, 3)
cl = st.slider("Clarity",    1, 5, 4)
nv = st.slider("Novelty",    1, 5, 3)
ov = st.slider("Overall",    1, 5, round((rr+cl+nv)/3))

# Save manual rating
if st.button("💾 Save Manual Rating"):
    st.session_state.strategy_json["rating"] = {
        "risk_reward": rr,
        "clarity":    cl,
        "novelty":    nv,
        "overall":    ov,
    }
    st.success("Manual rating saved!")

# Auto-rate with LLM
if st.button("🤖 Auto-Rate with LLM"):
    with st.spinner("Auto-rating…"):
        rating = auto_rate_strategy(st.session_state.strategy_json)
        st.session_state.strategy_json["rating"] = rating
    st.success("Auto-rating complete!")

# Show current rating
if isinstance(st.session_state.strategy_json, dict) and "rating" in st.session_state.strategy_json:
    st.markdown("**Current Rating**")
    st.json(st.session_state.strategy_json["rating"])

# --- Generate Code
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
    period = st.selectbox("Select Period", ["5d", "15d", "30d"], index=0)
    force = st.checkbox("⚠️ Force save even if incomplete")
    ext   = st.checkbox("🕒 Include extended-hours")

    if st.button("📡 Fetch Data"):
        result = fetch_intraday_period(
            ticker.upper(),
            period=period,
            interval="5m",
            max_missing_days=(999 if force else 2),
            include_extended_hours=ext
        )
        st.markdown("### 🧾 Data Quality Report")
        st.code(result)
        st.session_state.data_valid = result.startswith("🕵️") and "✅ Saved" in result

# --- Backtest
if st.session_state.data_valid and st.session_state.strategy_code:
    if st.button("📊 Run Backtest"):
        strategy_path = os.path.join("generated", file_name.replace(".json", ".py"))
        print(strategy_path)
        print(st.session_state.code_file)
        result = run_backtest_from_file(st.session_state.code_file)#strategy_path)
        st.text("Backtest Results:")
        st.markdown(result)
        if os.path.exists("sandbox/strategy_plot.png"):
            st.image("sandbox/strategy_plot.png", caption="📈 Entry/Exit Overlay")
else:
    st.warning("⚠️ Generate strategy, code, and fetch valid data before backtesting.")

