import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(
    page_title="Financial Unlearning System",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

def check_backend_status():
    """Pings the backend to check if services are online."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

def main():
    """Main function to render the Streamlit app."""
    st.title("🛡️ Financial Data Unlearning Certification System")
    st.markdown("""
        Upload a financial dataset (CSV) to perform sentiment analysis. Records with **Negative** or **Neutral** sentiment will be "unlearned" from our model using the SISA method, and a certificate of this action 
        will be logged to IPFS.
    """)

    with st.sidebar:

        uploaded_file = st.file_uploader(
            "Upload your financial data (CSV)",
            type=["csv"],
            help="CSV must contain 'account_number' and 'feedback_text' columns."
        )

        start_button = st.button("Start Unlearning & Certification", disabled=(uploaded_file is None))

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if uploaded_file is not None and ('df' not in st.session_state or st.session_state.df.empty):
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['account_number', 'feedback_text']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must have the following columns: {', '.join(required_cols)}.")
            else:
                # Initialize DataFrame for display
                df['status'] = 'Pending'
                df['sentiment'] = 'N/A'
                df['score'] = 0.0
                df['cid'] = 'N/A'
                # Ensure account_number is a string
                df['account_number'] = df['account_number'].astype(str)
                st.session_state.df = df
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if start_button and not st.session_state.df.empty:
        process_data()

    if not st.session_state.df.empty:
        st.dataframe(
            st.session_state.df[['account_name', 'feedback_text', 'status', 'sentiment', 'score', 'cid']],
            use_container_width=True
        )

def process_data():
    """Handles the logic of sending data to the backend and updating the UI."""
    df = st.session_state.df
    progress_bar = st.progress(0)
    status_text = st.empty()
    placeholder = st.empty()

    for index, row in df.iterrows():
        # Update status text
        status_text.text(f"Processing Account: {row['account_number']} ({index + 1}/{len(df)})...")
        
        try:
            # Prepare request payload
            payload = {
                "id": row['account_number'],
                "text": row['feedback_text'],
                "full_record": row.to_dict() # Send the entire row as a dictionary
            }
            
            # API call to backend
            response = requests.post(f"{API_URL}/certify-unlearning", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                df.loc[index, 'status'] = result['status']
                df.loc[index, 'sentiment'] = result['sentiment']['label']
                df.loc[index, 'score'] = result['sentiment']['score']
                df.loc[index, 'cid'] = result.get('cid', 'N/A')
            else:
                df.loc[index, 'status'] = 'Failed'
                df.loc[index, 'sentiment'] = f"Error: {response.status_code}"
                df.loc[index, 'cid'] = response.text

        except requests.exceptions.RequestException as e:
            df.loc[index, 'status'] = 'Failed'
            df.loc[index, 'sentiment'] = "Connection Failed"
            st.error(f"Could not connect to the backend at {API_URL}. Is it running?")
            break # Stop processing if backend is down

        # Update UI
        st.session_state.df = df
        progress_bar.progress((index + 1) / len(df))
        
        # Rerender the dataframe in place
        with placeholder.container():
             st.dataframe(
                st.session_state.df[['account_name', 'feedback_text', 'status', 'sentiment', 'score', 'cid']],
                use_container_width=True
            )
        time.sleep(0.1) # Small delay for smoother UI update

    status_text.success("Processing complete!")


if __name__ == "__main__":
    main()
