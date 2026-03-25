import streamlit as st
import pandas as pd
import subprocess
import sys
import csv
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="DeepTrack - Subsystem C", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SUBSYSTEM_A = BASE_DIR / "SubsystemA.py"
SUBSYSTEM_B = BASE_DIR / "SubsystemB.py"
CURRENT_STOCKS = BASE_DIR / "CurrentStocks.csv"
FORECAST_REPORT = BASE_DIR / "ForecastReport.csv"
FINAL_REPORT = BASE_DIR / "FinalReport.csv"

def run_python_script(script_path, args=None):
    """
    Runs a Python script and returns (success, stdout, stderr).
    """
    if args is None:
        args = []

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False
        )

        success = result.returncode == 0
        return success, result.stdout, result.stderr

    except Exception as e:
        return False, "", str(e)


def read_csv_safe(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {path.name}: {e}")
            return None
    return None


def download_button_for_file(path, label):
    if path.exists():
        with open(path, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=path.name,
                mime="text/csv"
            )

def update_final_report():
    if not FORECAST_REPORT.exists():
        return None

    forecast_df = read_csv_safe(FORECAST_REPORT)

    safety_buffers = {
        0: 10,  # Frontdoor keys
        1: 25,  # Phone
        2: 18,  # Toothpaste
        3: 20   # Wrist watch
    }

    reorder_results = []

    for _, row in forecast_df.iterrows():
        item_id = int(row['item_id'])
        item_name = row['item_name']
        current_qty = int(row['current_stock'])
        three_day_demand = int(row['sum'])

        buffer = safety_buffers.get(item_id, 10)

        remaining_after_3_days = current_qty - three_day_demand

        reorder_count = buffer - remaining_after_3_days

        status = None

        if reorder_count <= 0:
            status = "Good"
        elif reorder_count <= buffer:
            status = "Warning"
        elif reorder_count > buffer:
            status = "Critical"

        reorder_results.append({
            'item_id': item_id,
            'item_name': item_name,
            'remaining_after_3_days': remaining_after_3_days,
            'reorder_needed': max(0, reorder_count), # Don't show negative reorders
            'status': status
        })

    last_date = pd.to_datetime(forecast_df['forecast_date']).max()
    st.caption(f"📅 Forecast generated based on data up to: **{last_date.strftime('%Y-%m-%d')}**")

    final_df = pd.DataFrame(reorder_results)
    final_df.to_csv("FinalReport.csv", index=False)
    return final_df
    
    


st.title("DeepTrack - Subsystem C")
st.caption("Streamlit controller for Subsystem A (inventory detection) and Subsystem B (3-day demand forecasting)")

# Sidebar
st.sidebar.header("Controls")
run_a = st.sidebar.button("Run Subsystem A")
run_b = st.sidebar.button("Run Subsystem B")
run_full = st.sidebar.button("Run Full Pipeline (A → B)")
retrain_b = st.sidebar.checkbox("Force retrain forecasting model in Subsystem B", value=False)

st.sidebar.markdown("---")
st.sidebar.write(f"**Current time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main layout
tab1, tab2, tab3, tab4 = st.tabs(["Pipeline Control", "Current Stocks", "Forecast Report", "Final Reorder Report"])

with tab1:
    st.subheader("Pipeline Execution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Subsystem A")
        st.write("Runs YOLO-based stock detection and updates `CurrentStocks.csv`.")

    with col2:
        st.markdown("### Subsystem B")
        st.write("Reads `CurrentStocks.csv`, forecasts next 3 days, and updates `ForecastReport.csv`.")

    with col3:
        st.markdown("### Subsystem C")
        st.write("This Streamlit app coordinates both systems and displays the outputs.")

    if run_a:
        st.info("Running Subsystem A...")
        ok, stdout, stderr = run_python_script(SUBSYSTEM_A)

        if ok:
            st.success("Subsystem A completed successfully.")
        else:
            st.error("Subsystem A failed.")

        with st.expander("Subsystem A Output Log", expanded=True):
            st.text(stdout if stdout else "No stdout returned.")
            if stderr:
                st.text("STDERR:\n" + stderr)

    if run_b:
        st.info("Running Subsystem B...")
        args = ["--retrain"] if retrain_b else []
        ok, stdout, stderr = run_python_script(SUBSYSTEM_B, args=args)

        if ok:
            st.success("Subsystem B completed successfully.")
            update_final_report()
        else:
            st.error("Subsystem B failed.")

        with st.expander("Subsystem B Output Log", expanded=True):
            st.text(stdout if stdout else "No stdout returned.")
            if stderr:
                st.text("STDERR:\n" + stderr)

    if run_full:
        st.info("Running full pipeline...")

        st.write("### Step 1: Subsystem A")
        ok_a, stdout_a, stderr_a = run_python_script(SUBSYSTEM_A)

        if ok_a:
            st.success("Subsystem A completed successfully.")
        else:
            st.error("Subsystem A failed.")

        with st.expander("Subsystem A Output Log", expanded=False):
            st.text(stdout_a if stdout_a else "No stdout returned.")
            if stderr_a:
                st.text("STDERR:\n" + stderr_a)

        if ok_a:
            st.write("### Step 2: Subsystem B")
            args = ["--retrain"] if retrain_b else []
            ok_b, stdout_b, stderr_b = run_python_script(SUBSYSTEM_B, args=args)

            if ok_b:
                st.success("Subsystem B completed successfully.")
                update_final_report()
            else:
                st.error("Subsystem B failed.")

            with st.expander("Subsystem B Output Log", expanded=False):
                st.text(stdout_b if stdout_b else "No stdout returned.")
                if stderr_b:
                    st.text("STDERR:\n" + stderr_b)
        else:
            st.warning("Subsystem B was not run because Subsystem A failed.")

    st.markdown("---")
    st.subheader("File Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.write("**CurrentStocks.csv**")
        if CURRENT_STOCKS.exists():
            st.success("Found")
            st.write(f"Path: `{CURRENT_STOCKS}`")
        else:
            st.warning("Not found")

    with status_col2:
        st.write("**ForecastReport.csv**")
        if FORECAST_REPORT.exists():
            st.success("Found")
            st.write(f"Path: `{FORECAST_REPORT}`")
        else:
            st.warning("Not found")
    
    with status_col3:
        st.write("**FinalReport.csv**")
        if FINAL_REPORT.exists():
            st.success("Found")
            st.write(f"Path: `{FINAL_REPORT}`")
        else:
            st.warning("Not found")

with tab2:
    st.subheader("Current Stocks Output")
    current_df = read_csv_safe(CURRENT_STOCKS)

    if current_df is not None:
        st.dataframe(current_df, use_container_width=True)
        download_button_for_file(CURRENT_STOCKS, "Download CurrentStocks.csv")
    else:
        st.info("`CurrentStocks.csv` has not been generated yet. Run Subsystem A first.")

with tab3:
    st.subheader("Forecast Report Output")
    forecast_df = read_csv_safe(FORECAST_REPORT)

    if forecast_df is not None:
        st.dataframe(forecast_df, use_container_width=True)

        # Optional summary section
        st.markdown("### Forecast Summary")
        if "sum" in forecast_df.columns:
            total_predicted = forecast_df["sum"].sum()
            st.metric("Total Predicted Demand (Next 3 Days)", int(total_predicted))

        if "confidence" in forecast_df.columns:
            avg_conf = forecast_df["confidence"].mean()
            st.metric("Average Forecast Confidence", f"{avg_conf:.1f}%")

        if "item_name" in forecast_df.columns and "sum" in forecast_df.columns:
            st.markdown("### Demand by Item")
            chart_df = forecast_df[["item_name", "sum"]].copy()
            chart_df = chart_df.rename(columns={"item_name": "Item", "sum": "Predicted 3-Day Demand"})
            st.bar_chart(chart_df.set_index("Item"))

        download_button_for_file(FORECAST_REPORT, "Download ForecastReport.csv")
    else:
        st.info("`ForecastReport.csv` has not been generated yet. Run Subsystem B after Subsystem A.")

with tab4:
    st.subheader("Reorder Decision Support")
    final_df = update_final_report()

    if final_df is not None:
        def color_status(val):
            color = 'red' if val == 'Critical' else 'orange' if val == 'Warning' else 'green'
            return f'background-color: {color}; color: white; font-weight: bold'
        
        st.table(final_df.style.applymap(color_status, subset=['status']))
        st.info("💡 **Good**: Stock is sufficient. **Warning**: Stock low after 3 days. **Critical**: Immediate reorder required.")
        download_button_for_file(FINAL_REPORT, "Download FinalReport.csv")

    else:
        st.info("`ForecastReport.csv` has not been generated yet. Run Subsystem B after Subsystem A.")


st.markdown("---")
st.caption("Recommended run order: Subsystem A first, then Subsystem B.")