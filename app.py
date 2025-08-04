import streamlit as st
import pandas as pd
import requests
import time
import json

st.set_page_config(page_title="Verifiable Unlearning System", page_icon="🛡️", layout="wide")
API_URL = "http://127.0.0.1:8000"

def init_session_state():
    defaults = {
        "df": pd.DataFrame(), "edited_df": pd.DataFrame(), "model_trained": False,
        "initial_benchmark_run": False, "unlearning_complete": False,
        "initial_accuracy": 0.0, "final_accuracy": 0.0, "avg_unlearning_time": 0.0,
        "certificates": [], "file_uploader_key": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def on_train_click():
    st.session_state.model_trained = True
    payload = {"data": st.session_state.df.to_dict('records')}
    with st.spinner("🚀 Initiating model training... Check backend console for progress."):
        try:
            requests.post(f"{API_URL}/train", json=payload, timeout=600)
            st.success("✅ Training command sent! Please wait for it to complete.")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.session_state.model_trained = False

def on_benchmark_click(initial=True):
    key = "initial_accuracy" if initial else "final_accuracy"
    with st.spinner("📊 Running benchmark..."):
        try:
            res = requests.get(f"{API_URL}/benchmark", timeout=60).json()
            st.session_state[key] = res.get("accuracy", 0)
            if initial: st.session_state.initial_benchmark_run = True
            st.success(f"Benchmark complete! Accuracy: {st.session_state[key]:.2%}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

def on_unlearn_click():
    subset_to_unlearn = st.session_state.edited_df[st.session_state.edited_df["unlearn"] == True]
    if subset_to_unlearn.empty:
        st.warning("No records were selected for unlearning.")
        return

    unlearn_payload = [
        {"id": row['account_number'], "text": row['feedback_text']}
        for _, row in subset_to_unlearn.iterrows()
    ]
    
    with st.spinner(f"Unlearning {len(unlearn_payload)} records... This may take a while. Check backend console."):
        try:
            response = requests.post(f"{API_URL}/unlearn_subset", json=unlearn_payload, timeout=1800) # 30 min timeout
            if response.status_code == 200:
                results = response.json()
                st.session_state.certificates = results.get("certificates", [])
                st.session_state.avg_unlearning_time = results.get("average_unlearning_time", 0)
                st.session_state.unlearning_complete = True
                st.success("Unlearning process complete!")
            else:
                st.error(f"Unlearning failed: {response.text}")
        except Exception as e:
            st.error(f"Connection Error during unlearning: {e}")

def reset_workflow():
    st.session_state.file_uploader_key += 1
    init_session_state()

st.title("🛡️ Verifiable Unlearning with Ground Truth & Subsets")
st.markdown("""
**Workflow:** 1. Upload CSV with `ground_truth` -> 2. Train -> 3. Run Initial Benchmark -> 4. Select rows & Unlearn -> 5. Run Final Benchmark
""")

st.sidebar.header("Workflow Control")
uploaded_file = st.sidebar.file_uploader("Upload financial data", type=["csv"], key=f"file_uploader_{st.session_state.file_uploader_key}")

if uploaded_file and st.session_state.df.empty:
    df = pd.read_csv(uploaded_file)
    if 'ground_truth' not in df.columns:
        st.error("Uploaded CSV must contain a 'ground_truth' column.")
        st.stop()
    df["unlearn"] = False
    st.session_state.df = df

st.sidebar.button("Train", on_click=on_train_click, disabled=st.session_state.df.empty or st.session_state.model_trained)
st.sidebar.button("Initial Benchmark", on_click=lambda: on_benchmark_click(initial=True), disabled=not st.session_state.model_trained or st.session_state.initial_benchmark_run)
st.sidebar.button("Unlearn Selected", on_click=on_unlearn_click, disabled=not st.session_state.initial_benchmark_run or st.session_state.unlearning_complete)
st.sidebar.button("Final Benchmark", on_click=lambda: on_benchmark_click(initial=False), disabled=not st.session_state.unlearning_complete)
st.sidebar.button("Reset Workflow", on_click=reset_workflow)

st.header("Results Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy Before Unlearning", f"{st.session_state.initial_accuracy:.2%}")
col2.metric("Accuracy After Unlearning", f"{st.session_state.final_accuracy:.2%}")
col3.metric("Avg. Unlearning Time", f"{st.session_state.avg_unlearning_time:.2f} s")

if st.session_state.certificates:
    st.download_button(
        label="Download Unlearning Certificates",
        data=json.dumps(st.session_state.certificates, indent=2),
        file_name="unlearning_certificates.json",
        mime="application/json"
    )

if not st.session_state.df.empty:
    st.info("Select rows in the 'unlearn' column below and click 'Unlearn Selected'.")
    edited_df = st.data_editor(
        st.session_state.df,
        column_config={"unlearn": st.column_config.CheckboxColumn(required=True)},
        disabled=st.session_state.unlearning_complete,
        use_container_width=True
    )
    st.session_state.edited_df = edited_df
else:
    st.info("Please upload a CSV file with a 'ground_truth' column to begin.")