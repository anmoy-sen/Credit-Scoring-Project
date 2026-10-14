"""Interactive Underwriting & Credit Risk Dashboard.

Built with Streamlit for loan officers, risk analysts, and underwriters:
- Interactive single applicant risk evaluation with presets
- Credit score (300-850) gauge and risk tier classification
- Configurable decision thresholds aligned with cost-matrix optimization
- Batch portfolio scoring via CSV upload with distribution charts
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path so imports resolve
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.predict import load_pipeline, score_applicant, score_batch, get_risk_tier
from src.features import RAW_FEATURE_COLUMNS

st.set_page_config(
    page_title="Credit Scoring & Underwriting Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pipeline loading to keep UI responsive
@st.cache_resource
def get_cached_pipeline():
    try:
        return load_pipeline()
    except Exception as e:
        st.error(f"Failed to load champion pipeline: {e}")
        return None

pipeline = get_cached_pipeline()

# Preset profiles for rapid testing
PRESETS = {
    "Custom Input": None,
    "Prime Borrower (Low Risk)": {
        "LIMIT_BAL": 200000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 38,
        "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 25000.0, "BILL_AMT2": 24000.0, "BILL_AMT3": 22000.0,
        "BILL_AMT4": 20000.0, "BILL_AMT5": 19000.0, "BILL_AMT6": 18000.0,
        "PAY_AMT1": 25000.0, "PAY_AMT2": 24000.0, "PAY_AMT3": 22000.0,
        "PAY_AMT4": 20000.0, "PAY_AMT5": 19000.0, "PAY_AMT6": 18000.0
    },
    "Borderline Borrower (Medium Risk)": {
        "LIMIT_BAL": 80000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 29,
        "PAY_1": 1, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 65000.0, "BILL_AMT2": 62000.0, "BILL_AMT3": 58000.0,
        "BILL_AMT4": 55000.0, "BILL_AMT5": 50000.0, "BILL_AMT6": 48000.0,
        "PAY_AMT1": 2500.0, "PAY_AMT2": 2500.0, "PAY_AMT3": 2000.0,
        "PAY_AMT4": 2000.0, "PAY_AMT5": 2000.0, "PAY_AMT6": 2000.0
    },
    "Delinquent Borrower (High Risk)": {
        "LIMIT_BAL": 30000.0, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 25,
        "PAY_1": 2, "PAY_2": 2, "PAY_3": 3, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
        "BILL_AMT1": 32000.0, "BILL_AMT2": 31500.0, "BILL_AMT3": 31000.0,
        "BILL_AMT4": 30500.0, "BILL_AMT5": 30000.0, "BILL_AMT6": 29500.0,
        "PAY_AMT1": 0.0, "PAY_AMT2": 500.0, "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0, "PAY_AMT5": 500.0, "PAY_AMT6": 0.0
    }
}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("💳 Risk Parameters")
    st.markdown("Configure operational underwriting thresholds.")

    threshold = st.slider(
        "Decision Cutoff Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.01,
        help="Applicants with default probability equal to or exceeding this cutoff are flagged for DECLINE. (Optimal financial cutoff: 0.40)."
    )

    app_mode = st.radio(
        "Underwriting Mode",
        ["Single Applicant Scoring", "Batch Portfolio Scoring (CSV)"]
    )

    st.divider()
    st.markdown("### System Metadata")
    st.info(
        "**Pipeline**: `champion_pipeline.pkl`\n\n"
        "**Tracking**: DVC v3.67 & Git\n\n"
        "**Score Range**: 300 - 850 (FICO scale)\n\n"
        "**Optimal Threshold**: 0.40"
    )

# --- MAIN DASHBOARD ---
st.title("Credit Risk Scoring & Decisioning Portal")
st.markdown("AI-driven underwriting system evaluating probability of default and issuing standardized credit scores.")

if pipeline is None:
    st.error("Model pipeline is not found! Please run `dvc repro` or `python -m src.train` first.")
    st.stop()

# --- MODE 1: SINGLE APPLICANT SCORING ---
if app_mode == "Single Applicant Scoring":
    st.subheader("1. Applicant Assessment Form")

    preset_name = st.selectbox(
        "Load Preset Profile for Rapid Testing:",
        options=list(PRESETS.keys())
    )

    preset_data = PRESETS[preset_name]

    with st.form("applicant_form"):
        col_demo, col_credit = st.columns(2)

        with col_demo:
            st.markdown("#### Demographics & Line")
            limit_bal = st.number_input(
                "Credit Limit (LIMIT_BAL in NT$)",
                min_value=5000.0,
                max_value=1000000.0,
                value=float(preset_data["LIMIT_BAL"]) if preset_data else 50000.0,
                step=5000.0
            )
            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=int(preset_data["AGE"]) if preset_data else 32
            )
            sex = st.selectbox(
                "Gender",
                options=[1, 2],
                format_func=lambda x: "Male (1)" if x == 1 else "Female (2)",
                index=0 if (preset_data and preset_data["SEX"] == 1) else 1
            )
            education = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Graduate School (1)", 2: "University (2)", 3: "High School (3)", 4: "Others (4)"}.get(x),
                index=(preset_data["EDUCATION"] - 1) if preset_data and preset_data["EDUCATION"] in [1, 2, 3, 4] else 1
            )
            marriage = st.selectbox(
                "Marital Status",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Married (1)", 2: "Single (2)", 3: "Others / Divorced (3)"}.get(x),
                index=(preset_data["MARRIAGE"] - 1) if preset_data and preset_data["MARRIAGE"] in [1, 2, 3] else 0
            )

        with col_credit:
            st.markdown("#### Repayment History (Past 6 Months)")
            st.caption("Key: -1 = Paid full, 0 = Revolving balance, 1 = 1 month delay, 2 = 2 months delay, etc.")
            p_cols = st.columns(3)
            pay_vals = {}
            for idx, m_name in enumerate(["Sept (PAY_1)", "Aug (PAY_2)", "Jul (PAY_3)", "Jun (PAY_4)", "May (PAY_5)", "Apr (PAY_6)"]):
                with p_cols[idx % 3]:
                    key = f"PAY_{idx+1}"
                    val = preset_data[key] if preset_data else 0
                    pay_vals[key] = st.selectbox(
                        m_name,
                        options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                        index=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].index(val)
                    )

        st.markdown("#### Monthly Statement & Payment Details")
        b_cols = st.columns(6)
        bill_vals = {}
        for idx in range(1, 7):
            with b_cols[idx - 1]:
                b_key = f"BILL_AMT{idx}"
                def_b = preset_data[b_key] if preset_data else 20000.0
                bill_vals[b_key] = st.number_input(f"Bill M-{idx}", value=float(def_b), step=1000.0)

        p_cols = st.columns(6)
        pay_amt_vals = {}
        for idx in range(1, 7):
            with p_cols[idx - 1]:
                p_key = f"PAY_AMT{idx}"
                def_p = preset_data[p_key] if preset_data else 2000.0
                pay_amt_vals[p_key] = st.number_input(f"Paid M-{idx}", min_value=0.0, value=float(def_p), step=500.0)

        submit_btn = st.form_submit_button("🚀 Run Risk Evaluation", use_container_width=True)

    if submit_btn:
        # Build dictionary
        raw_payload = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **pay_vals,
            **bill_vals,
            **pay_amt_vals
        }

        with st.spinner("Processing applicant scoring..."):
            result = score_applicant(raw_payload, pipeline=pipeline, threshold=threshold)

        st.divider()
        st.subheader("2. Underwriting Decision & Scoring Report")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            score = result["credit_score"]
            score_color = "🟢" if score >= 700 else ("🟡" if score >= 600 else "🔴")
            st.metric(label="Credit Score (300-850)", value=f"{score_color} {score}")

        with res_col2:
            prob = result["probability_of_default"]
            st.metric(label="Default Probability", value=f"{prob * 100:.1f}%")

        with res_col3:
            st.metric(label="Assigned Risk Tier", value=result["risk_tier"])

        with res_col4:
            dec = result["decision"]
            st.metric(label="Recommendation", value=f"✅ {dec}" if dec == "APPROVE" else f"❌ {dec}")

        # Big banner
        if result["decision"] == "APPROVE":
            st.success(f"**DECISION: APPROVE APPLICATION** (Default Risk: {prob * 100:.1f}% is below operational cutoff {threshold * 100:.0f}%)")
        else:
            st.error(f"**DECISION: DECLINE APPLICATION** (Default Risk: {prob * 100:.1f}% exceeds operational cutoff {threshold * 100:.0f}%)")

        # Computed indices
        st.markdown("##### Key Behavioral Financial Indicators")
        kcol1, kcol2, kcol3 = st.columns(3)
        avg_bill = np.mean(list(bill_vals.values()))
        utilization = (abs(avg_bill) / (limit_bal + 1)) * 100
        late_count = sum(1 for v in pay_vals.values() if v >= 1)
        avg_payment = np.mean(list(pay_amt_vals.values()))

        kcol1.metric("Credit Utilization", f"{utilization:.1f}%")
        kcol2.metric("Delinquent Months", f"{late_count} of 6")
        kcol3.metric("Avg Monthly Payment", f"NT$ {avg_payment:,.0f}")

# --- MODE 2: BATCH PORTFOLIO SCORING ---
else:
    st.subheader("Batch Portfolio Evaluation")
    st.markdown("Upload a CSV file containing applicant records. The pipeline will engineer features, predict default probabilities, compute credit scores, and apply the operational decision cutoff.")

    uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df_batch):,}** records with **{df_batch.shape[1]}** columns.")

        if st.button("⚡ Score Full Portfolio", use_container_width=True):
            with st.spinner("Scoring portfolio across ML pipeline..."):
                df_features = df_batch.drop(columns=["default"]) if "default" in df_batch.columns else df_batch
                scored_df = score_batch(df_features, pipeline=pipeline, threshold=threshold)

            st.success(f"Successfully processed {len(scored_df):,} applicants!")

            # Summary Metrics
            approved = int((scored_df["decision"] == "APPROVE").sum())
            declined = int((scored_df["decision"] == "DECLINE").sum())
            appr_rate = (approved / len(scored_df)) * 100
            avg_score = scored_df["credit_score"].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Applicants", f"{len(scored_df):,}")
            m2.metric("Approved", f"{approved:,} ({appr_rate:.1f}%)")
            m3.metric("Declined", f"{declined:,} ({100 - appr_rate:.1f}%)")
            m4.metric("Avg Portfolio Credit Score", f"{avg_score:.0f}")

            # Visualizations
            st.markdown("#### Portfolio Risk Distribution")
            vcol1, vcol2 = st.columns(2)

            with vcol1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(scored_df["credit_score"], bins=30, kde=True, color="#1f77b4", ax=ax)
                ax.set_title("Credit Score Distribution (300 - 850)")
                ax.set_xlabel("Credit Score")
                ax.set_ylabel("Applicant Count")
                st.pyplot(fig)

            with vcol2:
                fig, ax = plt.subplots(figsize=(6, 4))
                tier_counts = scored_df["risk_tier"].value_counts()
                ax.pie(tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%", startangle=140, colors=["#2ca02c", "#ff7f0e", "#d62728"])
                ax.set_title("Risk Tier Breakdown")
                st.pyplot(fig)

            # Table & Download
            st.markdown("#### Scored Records")
            st.dataframe(scored_df[["LIMIT_BAL", "AGE", "probability_of_default", "credit_score", "risk_tier", "decision"]].head(100))

            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Scored CSV",
                data=csv_data,
                file_name="portfolio_scored_results.csv",
                mime="text/csv",
                use_container_width=True
            )
