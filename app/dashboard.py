"""Interactive Underwriting & Credit Risk Dashboard.

Built with Streamlit for loan officers, risk analysts, and underwriters:
- Interactive single applicant risk evaluation with real-world presets
- Structured 6-month financial and repayment ledger
- FICO-style credit score (300-850) visual gauge and risk tier classification
- Configurable decision thresholds aligned with cost-matrix optimization
- Explainable AI with TreeSHAP attributions & adverse action notices (FCRA / ECOA)
- Batch portfolio scoring via CSV upload or instant built-in demo data
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path so src imports resolve cleanly in any environment
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

# Custom light styling enhancements
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .badge-approved {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-declined {
        background-color: #f8d7da;
        color: #842029;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Cache pipeline loading to keep UI responsive
@st.cache_resource
def get_cached_pipeline():
    try:
        return load_pipeline()
    except Exception as e:
        st.error(f"Failed to load champion pipeline: {e}")
        return None


pipeline = get_cached_pipeline()


@st.cache_resource
def get_cached_explainer():
    p = get_cached_pipeline()
    if p is not None:
        try:
            from src.explainability import CreditExplainer
            return CreditExplainer(p)
        except Exception as e:
            st.error(f"Failed to load explainer: {e}")
            return None
    return None


explainer = get_cached_explainer()

# Presets for rapid testing
PRESETS = {
    "Custom Applicant (Manual Entry)": None,
    "Prime Borrower (Low Risk — Auto Approve)": {
        "LIMIT_BAL": 200000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 38,
        "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 25000.0, "BILL_AMT2": 24000.0, "BILL_AMT3": 22000.0,
        "BILL_AMT4": 20000.0, "BILL_AMT5": 19000.0, "BILL_AMT6": 18000.0,
        "PAY_AMT1": 25000.0, "PAY_AMT2": 24000.0, "PAY_AMT3": 22000.0,
        "PAY_AMT4": 20000.0, "PAY_AMT5": 19000.0, "PAY_AMT6": 18000.0
    },
    "Borderline Borrower (Moderate Risk — Manual Review)": {
        "LIMIT_BAL": 80000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 29,
        "PAY_1": 1, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 65000.0, "BILL_AMT2": 62000.0, "BILL_AMT3": 58000.0,
        "BILL_AMT4": 55000.0, "BILL_AMT5": 50000.0, "BILL_AMT6": 48000.0,
        "PAY_AMT1": 2500.0, "PAY_AMT2": 2500.0, "PAY_AMT3": 2000.0,
        "PAY_AMT4": 2000.0, "PAY_AMT5": 2000.0, "PAY_AMT6": 2000.0
    },
    "Delinquent Borrower (High Risk — Immediate Decline)": {
        "LIMIT_BAL": 30000.0, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 25,
        "PAY_1": 2, "PAY_2": 2, "PAY_3": 3, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
        "BILL_AMT1": 32000.0, "BILL_AMT2": 31500.0, "BILL_AMT3": 31000.0,
        "BILL_AMT4": 30500.0, "BILL_AMT5": 30000.0, "BILL_AMT6": 29500.0,
        "PAY_AMT1": 0.0, "PAY_AMT2": 500.0, "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0, "PAY_AMT5": 500.0, "PAY_AMT6": 0.0
    }
}

# Human-readable payment status descriptions
STATUS_OPTIONS = [-1, 0, -2, 1, 2, 3, 4, 5, 6, 7, 8]
STATUS_LABELS = {
    -1: "Paid in full (On time)",
    0: "Revolving balance (Paid minimum)",
    -2: "No balance / Inactive",
    1: "1 month delay",
    2: "2 months delay",
    3: "3 months delay",
    4: "4 months delay",
    5: "5 months delay",
    6: "6 months delay",
    7: "7 months delay",
    8: "8+ months delay"
}

# 6 Months Metadata (September to April)
MONTHS_METADATA = [
    {"num": 1, "name": "Month 1 (Most Recent — Sep)", "col_pay": "PAY_1", "col_bill": "BILL_AMT1", "col_amt": "PAY_AMT1"},
    {"num": 2, "name": "Month 2 (2 Mos Ago — Aug)", "col_pay": "PAY_2", "col_bill": "BILL_AMT2", "col_amt": "PAY_AMT2"},
    {"num": 3, "name": "Month 3 (3 Mos Ago — Jul)", "col_pay": "PAY_3", "col_bill": "BILL_AMT3", "col_amt": "PAY_AMT3"},
    {"num": 4, "name": "Month 4 (4 Mos Ago — Jun)", "col_pay": "PAY_4", "col_bill": "BILL_AMT4", "col_amt": "PAY_AMT4"},
    {"num": 5, "name": "Month 5 (5 Mos Ago — May)", "col_pay": "PAY_5", "col_bill": "BILL_AMT5", "col_amt": "PAY_AMT5"},
    {"num": 6, "name": "Month 6 (6 Mos Ago — Apr)", "col_pay": "PAY_6", "col_bill": "BILL_AMT6", "col_amt": "PAY_AMT6"},
]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("💳 Underwriting Controls")
    st.markdown("Configure operational decision thresholds and underwriting workflows.")

    threshold = st.slider(
        "Decision Cutoff Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.01,
        help="Applicants with default probability at or above this cutoff are flagged for DECLINE. Optimal cost cutoff is 0.40."
    )

    app_mode = st.radio(
        "Workflow Mode",
        ["Single Applicant Scoring", "Batch Portfolio Scoring (CSV)"]
    )

    st.divider()
    st.markdown("### System Metadata")
    st.info(
        "**Production Model**: Champion Pipeline (Random Forest)\n\n"
        "**Score Range**: 300 - 850 (FICO standard)\n\n"
        "**Optimal Financial Cutoff**: 0.40\n\n"
        "**Governance**: TreeSHAP Adverse Action (FCRA/ECOA)"
    )


# --- HEADER ---
st.title("Credit Risk Scoring & Underwriting Portal")
st.markdown(
    "Enterprise AI platform evaluating credit default risk, assigning standardized 300–850 credit scores, "
    "and explaining underwriting decisions through TreeSHAP attributions."
)

if pipeline is None:
    st.error("Champion model pipeline artifact not found at `models/champion_pipeline.pkl`. Please train or track the model first.")
    st.stop()


# --- MODE 1: SINGLE APPLICANT SCORING ---
if app_mode == "Single Applicant Scoring":
    st.subheader("1. Applicant Information & Financial Profile")

    preset_name = st.selectbox(
        "Load Preset Profile for Rapid Testing:",
        options=list(PRESETS.keys()),
        help="Quickly populate the form with realistic risk profiles."
    )

    preset_data = PRESETS[preset_name]

    with st.form("applicant_evaluation_form"):
        # Profile & Demographics Section
        st.markdown("#### 👤 Demographics & Assigned Credit Line")
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)

        with d_col1:
            limit_bal = st.number_input(
                "Total Credit Line (NT$)",
                min_value=5000.0,
                max_value=1000000.0,
                value=float(preset_data["LIMIT_BAL"]) if preset_data else 100000.0,
                step=5000.0,
                help="Total approved credit line granted to the applicant."
            )

        with d_col2:
            age = st.number_input(
                "Applicant Age",
                min_value=18,
                max_value=100,
                value=int(preset_data["AGE"]) if preset_data else 32
            )

        with d_col3:
            sex = st.selectbox(
                "Gender",
                options=[1, 2],
                format_func=lambda x: "Male" if x == 1 else "Female",
                index=0 if (preset_data and preset_data["SEX"] == 1) else 1
            )

        with d_col4:
            education = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Graduate School",
                    2: "University",
                    3: "High School",
                    4: "Other / Vocational"
                }.get(x, "Other"),
                index=(preset_data["EDUCATION"] - 1) if preset_data and preset_data["EDUCATION"] in [1, 2, 3, 4] else 1
            )

        with d_col5:
            marriage = st.selectbox(
                "Marital Status",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Married",
                    2: "Single",
                    3: "Divorced / Other"
                }.get(x, "Other"),
                index=(preset_data["MARRIAGE"] - 1) if preset_data and preset_data["MARRIAGE"] in [1, 2, 3] else 0
            )

        st.divider()

        # 6-Month Ledger Section (Clean presentation without raw column names)
        st.markdown("#### 🗓️ Last 6 Months Credit & Financial Records")
        st.caption(
            "Review statement balances, payments made, and repayment timeliness for each of the last six billing cycles (September down to April)."
        )

        col_recent_q, col_prior_q = st.columns(2)
        pay_vals = {}
        bill_vals = {}
        pay_amt_vals = {}

        # Recent Quarter (Months 1 to 3)
        with col_recent_q:
            st.markdown("##### 🕒 Recent Quarter (Months 1 – 3)")
            for m in MONTHS_METADATA[:3]:
                idx = m["num"]
                def_pay = preset_data[m["col_pay"]] if preset_data else 0
                def_bill = preset_data[m["col_bill"]] if preset_data else 20000.0
                def_amt = preset_data[m["col_amt"]] if preset_data else 2000.0

                with st.container(border=True):
                    st.markdown(f"**{m['name']}**")
                    sc1, sc2, sc3 = st.columns([1.4, 1.1, 1.1])
                    with sc1:
                        pay_idx = STATUS_OPTIONS.index(def_pay) if def_pay in STATUS_OPTIONS else 1
                        pay_vals[m["col_pay"]] = st.selectbox(
                            "Repayment Status",
                            options=STATUS_OPTIONS,
                            format_func=lambda x: STATUS_LABELS.get(x, f"{x} mos delay"),
                            index=pay_idx,
                            key=f"status_m_{idx}"
                        )
                    with sc2:
                        bill_vals[m["col_bill"]] = st.number_input(
                            "Statement Balance",
                            value=float(def_bill),
                            step=1000.0,
                            key=f"bill_m_{idx}"
                        )
                    with sc3:
                        pay_amt_vals[m["col_amt"]] = st.number_input(
                            "Payment Made",
                            min_value=0.0,
                            value=float(def_amt),
                            step=500.0,
                            key=f"pay_amt_m_{idx}"
                        )

        # Prior Quarter (Months 4 to 6)
        with col_prior_q:
            st.markdown("##### 🕒 Prior Quarter (Months 4 – 6)")
            for m in MONTHS_METADATA[3:]:
                idx = m["num"]
                def_pay = preset_data[m["col_pay"]] if preset_data else 0
                def_bill = preset_data[m["col_bill"]] if preset_data else 20000.0
                def_amt = preset_data[m["col_amt"]] if preset_data else 2000.0

                with st.container(border=True):
                    st.markdown(f"**{m['name']}**")
                    sc1, sc2, sc3 = st.columns([1.4, 1.1, 1.1])
                    with sc1:
                        pay_idx = STATUS_OPTIONS.index(def_pay) if def_pay in STATUS_OPTIONS else 1
                        pay_vals[m["col_pay"]] = st.selectbox(
                            "Repayment Status",
                            options=STATUS_OPTIONS,
                            format_func=lambda x: STATUS_LABELS.get(x, f"{x} mos delay"),
                            index=pay_idx,
                            key=f"status_m_{idx}"
                        )
                    with sc2:
                        bill_vals[m["col_bill"]] = st.number_input(
                            "Statement Balance",
                            value=float(def_bill),
                            step=1000.0,
                            key=f"bill_m_{idx}"
                        )
                    with sc3:
                        pay_amt_vals[m["col_amt"]] = st.number_input(
                            "Payment Made",
                            min_value=0.0,
                            value=float(def_amt),
                            step=500.0,
                            key=f"pay_amt_m_{idx}"
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Run Comprehensive Risk Assessment", use_container_width=True)

    if submit_btn:
        # Build ingestion payload
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

        with st.spinner("Executing feature engineering & model scoring..."):
            result = score_applicant(raw_payload, pipeline=pipeline, threshold=threshold)

        st.divider()
        st.subheader("2. Underwriting Decision & Risk Scoring Report")

        # Top summary banner
        prob = result["probability_of_default"]
        score = result["credit_score"]
        decision = result["decision"]
        risk_tier = result["risk_tier"]

        if decision == "APPROVE":
            st.success(
                f"### ✅ APPLICATION APPROVED\n"
                f"**Assessed Default Risk**: `{prob * 100:.1f}%` is within operational tolerance `{threshold * 100:.0f}%`."
            )
        else:
            st.error(
                f"### ❌ APPLICATION DECLINED\n"
                f"**Assessed Default Risk**: `{prob * 100:.1f}%` exceeds operational tolerance `{threshold * 100:.0f}%`."
            )

        # Primary Metrics Grid
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown("##### Standard Credit Score")
            st.metric(
                label="FICO Scale (300 – 850)",
                value=f"{score}",
                help="Calibrated logarithmic translation from default probability to standard credit score."
            )

        with m_col2:
            st.markdown("##### Default Probability")
            st.metric(
                label="Likelihood of 90+ DPD",
                value=f"{prob * 100:.1f}%",
                delta=f"{(prob - threshold) * 100:+.1f}% vs Cutoff",
                delta_color="inverse"
            )

        with m_col3:
            st.markdown("##### Assigned Risk Tier")
            st.metric(
                label="Portfolio Segmentation",
                value=risk_tier
            )

        with m_col4:
            st.markdown("##### Recommendation")
            st.metric(
                label="Final System Action",
                value=f"{decision}"
            )

        # Visual Credit Score Gauge Progress Bar
        st.markdown("##### Credit Score Quality Indicator")
        score_pct = max(0.0, min(1.0, (score - 300) / (850 - 300)))
        st.progress(score_pct)

        if score >= 750:
            rating_text = "🟢 **Excellent (750 - 850)** — Prime Tier, minimal historical default risk."
        elif score >= 700:
            rating_text = "🟢 **Good (700 - 749)** — Low risk, favorable repayment behavior."
        elif score >= 650:
            rating_text = "🟡 **Fair (650 - 699)** — Moderate risk, eligible with standard conditions."
        elif score >= 600:
            rating_text = "🟠 **Poor (600 - 649)** — Elevated risk, manual underwriter review recommended."
        else:
            rating_text = "🔴 **Very Poor (300 - 599)** — Deep subprime risk, high probability of delinquency."
        st.caption(rating_text)

        # Financial Health Indicators
        st.markdown("---")
        st.markdown("##### 📊 Derived Behavioral Financial Health Indicators")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)

        avg_bill = np.mean(list(bill_vals.values()))
        avg_payment = np.mean(list(pay_amt_vals.values()))
        utilization = (abs(avg_bill) / (limit_bal + 1)) * 100
        delinquent_count = sum(1 for v in pay_vals.values() if v >= 1)
        coverage_ratio = (avg_payment / (abs(avg_bill) + 1)) * 100

        with kcol1:
            st.metric("Credit Line Utilization", f"{utilization:.1f}%", help="Average monthly balance relative to approved credit line.")
        with kcol2:
            st.metric("Delinquent Months", f"{delinquent_count} / 6", help="Number of months with 1+ month payment delay.")
        with kcol3:
            st.metric("Avg Monthly Statement", f"NT$ {avg_bill:,.0f}")
        with kcol4:
            st.metric("Repayment Coverage", f"{coverage_ratio:.1f}%", help="Average payment relative to monthly statement balance.")

        # Explainability & Regulatory Adverse Action (TreeSHAP)
        st.markdown("---")
        st.subheader("3. Explainability & Regulatory Compliance (TreeSHAP)")
        st.markdown(
            "Under FCRA and ECOA guidelines, lenders must provide clear, quantifiable explanations for adverse actions. "
            "TreeSHAP isolates the exact contribution of each factor to the final risk decision."
        )

        if explainer is not None:
            with st.spinner("Computing local TreeSHAP feature attributions..."):
                explanation = explainer.explain_applicant(raw_payload, top_k=5)

            adverse = explanation.get("adverse_factors", [])
            protective = explanation.get("protective_factors", [])

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("#### ⚠️ Primary Adverse Factors (Risk Drivers)")
                st.caption("Key reasons contributing negatively to creditworthiness (Adverse Action Notice):")
                if adverse:
                    for f in adverse:
                        st.error(f"**{f['label']}**  \nRisk impact: `+{f['shap_impact']:.4f}`")
                else:
                    st.info("No significant negative risk factors identified.")

            with exp_col2:
                st.markdown("#### 🟢 Positive Factors (Credit Builders)")
                st.caption("Key mitigating factors that improved the credit evaluation:")
                if protective:
                    for f in protective:
                        st.success(f"**{f['label']}**  \nRisk reduction: `{f['shap_impact']:.4f}`")
                else:
                    st.info("No significant protective factors identified.")

            # Decision Attribution Force / Bar Chart
            st.markdown("#### Decision Attribution Summary (Top 10 Factors)")
            top_attrs = explanation.get("all_attributions", [])[:10]
            if top_attrs:
                fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
                plot_labels = [a["label"] for a in reversed(top_attrs)]
                plot_impacts = [a["shap_impact"] for a in reversed(top_attrs)]
                plot_colors = ["#dc3545" if x > 0 else "#198754" for x in plot_impacts]

                ax.barh(plot_labels, plot_impacts, color=plot_colors, alpha=0.9, height=0.6)
                ax.axvline(0, color="#212529", linestyle="--", linewidth=0.9)
                ax.set_xlabel("SHAP Impact on Default Probability (Red = Increases Risk, Green = Reduces Risk)")
                ax.set_title("Top 10 Feature Attributions for this Applicant", fontsize=12, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("TreeSHAP Explainer module is not initialized. Ensure `shap` is installed in your environment.")


# --- MODE 2: BATCH PORTFOLIO SCORING ---
else:
    st.subheader("Batch Portfolio Underwriting")
    st.markdown(
        "Upload a batch CSV file of applicant records. The automated scoring pipeline will engineer behavioral features, "
        "predict default probabilities, compute standard credit scores, and apply the operational decision cutoff."
    )

    sample_csv_path = project_root / "data" / "sample_applicants.csv"
    demo_df = None
    if sample_csv_path.exists():
        demo_df = pd.read_csv(sample_csv_path)

    col_btn, col_info = st.columns([1, 2])
    use_demo = False
    with col_btn:
        if demo_df is not None:
            if st.button("🧪 Quick Test: Load Built-in Demo Portfolio (15 Applicants)", use_container_width=True):
                use_demo = True

    uploaded_file = st.file_uploader("Or Upload Your Applicant CSV File:", type=["csv"])

    active_batch_df = None
    if uploaded_file is not None:
        active_batch_df = pd.read_csv(uploaded_file)
    elif use_demo and demo_df is not None:
        active_batch_df = demo_df.copy()
        st.info("Loaded built-in portfolio of 15 sample applicants for rapid testing.")

    if active_batch_df is not None:
        st.write(f"Loaded **{len(active_batch_df):,}** applicant records with **{active_batch_df.shape[1]}** features.")

        if st.button("⚡ Score Full Portfolio", use_container_width=True):
            with st.spinner("Scoring portfolio across end-to-end ML pipeline..."):
                df_features = active_batch_df.drop(columns=["default"]) if "default" in active_batch_df.columns else active_batch_df
                scored_df = score_batch(df_features, pipeline=pipeline, threshold=threshold)

            st.success(f"Successfully processed and scored {len(scored_df):,} applicants!")

            # Summary Metrics
            approved = int((scored_df["decision"] == "APPROVE").sum())
            declined = int((scored_df["decision"] == "DECLINE").sum())
            appr_rate = (approved / len(scored_df)) * 100
            avg_score = scored_df["credit_score"].mean()
            avg_prob = scored_df["probability_of_default"].mean()

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total Applicants", f"{len(scored_df):,}")
            sm2.metric("Approved", f"{approved:,} ({appr_rate:.1f}%)")
            sm3.metric("Declined", f"{declined:,} ({100 - appr_rate:.1f}%)")
            sm4.metric("Avg Portfolio Credit Score", f"{avg_score:.0f}")

            # Visualizations
            st.markdown("#### Portfolio Risk & Score Distributions")
            vcol1, vcol2 = st.columns(2)

            with vcol1:
                fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
                sns.histplot(scored_df["credit_score"], bins=20, kde=True, color="#0d6efd", ax=ax)
                ax.set_title("Credit Score Distribution (300 – 850)", fontweight="bold")
                ax.set_xlabel("Credit Score")
                ax.set_ylabel("Applicant Count")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

            with vcol2:
                fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
                tier_counts = scored_df["risk_tier"].value_counts()
                colors = ["#198754", "#fd7e14", "#dc3545"]
                ax.pie(
                    tier_counts.values,
                    labels=tier_counts.index,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=colors[:len(tier_counts)]
                )
                ax.set_title("Portfolio Risk Tier Breakdown", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)

            # Table & Download
            st.markdown("#### Scored Records Preview")
            st.dataframe(
                scored_df[[
                    "LIMIT_BAL", "AGE", "probability_of_default", "credit_score", "risk_tier", "decision"
                ]].head(100),
                use_container_width=True
            )

            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Scored Results (CSV)",
                data=csv_data,
                file_name="portfolio_scored_results.csv",
                mime="text/csv",
                use_container_width=True
            )
