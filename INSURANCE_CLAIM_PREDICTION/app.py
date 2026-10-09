from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance_claims.xlsx"
MODEL_PATH = BASE_DIR / "models" / "insurance_fraud_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Insurance Claim Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main .block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#0b2545,#164e63); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin-bottom:.35rem;}
.hero p {color:#e0f2fe; font-size:1.03rem; margin-bottom:0;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25); padding:14px 16px; border-radius:14px;}
.small-note {font-size:.88rem; opacity:.8;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    data = pd.read_excel(DATA_PATH, engine="openpyxl")
    data = data.replace("?", pd.NA)
    return data

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

df = load_data()
model = load_model()
metrics = load_metrics()

st.markdown("""
<div class="hero">
  <h1>🛡️ Insurance Claim Intelligence</h1>
  <p>Explore claim patterns, review model performance, and test a machine-learning fraud-risk prediction workflow.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Choose a workspace",
        ["Overview", "Explore Claims", "Predict Claim", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Machine Learning • Analytics • Streamlit")
    st.caption("Educational decision-support prototype")

if df is None:
    st.error("Dataset not found. Ensure data/insurance_claims.xlsx is included in the repository.")
    st.stop()

target_col = "fraud_reported"
if target_col not in df.columns:
    st.error("Expected target column 'fraud_reported' was not found in the dataset.")
    st.stop()

df_view = df.copy()
df_view[target_col] = df_view[target_col].astype("string").replace({"Y": "Fraud reported", "N": "No fraud reported"})

if page == "Overview":
    st.subheader("Project overview")
    st.write("A machine-learning case study using insurance policy and incident information to estimate whether a claim resembles patterns associated with previously reported fraud.")
    total = len(df)
    fraud_count = int((df[target_col].astype(str).str.upper() == "Y").sum())
    no_fraud_count = int((df[target_col].astype(str).str.upper() == "N").sum())
    fraud_rate = fraud_count / total if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{total:,}")
    c2.metric("Fraud reported", f"{fraud_count:,}")
    c3.metric("No fraud reported", f"{no_fraud_count:,}")
    c4.metric("Reported-fraud share", f"{fraud_rate:.1%}")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Target distribution")
        counts = df_view[target_col].value_counts().rename_axis("Claim label").reset_index(name="Records")
        st.bar_chart(counts.set_index("Claim label"))
    with right:
        st.markdown("#### Dataset snapshot")
        st.write(f"**Rows:** {df.shape[0]:,}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")
        st.write(f"**Model status:** {'Trained model available' if model else 'Model file not found'}")
    st.info("This is an educational prototype. A model output must not be used as the sole basis for approving, denying, or investigating an actual insurance claim.")

elif page == "Explore Claims":
    st.subheader("📊 Explore claims")
    st.caption("Filter the dataset and explore claim and incident patterns.")
    filtered = df_view.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        if "incident_type" in filtered:
            options = sorted(filtered["incident_type"].dropna().astype(str).unique())
            selected = st.multiselect("Incident type", options, default=options)
            if selected:
                filtered = filtered[filtered["incident_type"].astype(str).isin(selected)]
    with col2:
        if "incident_severity" in filtered:
            options = sorted(filtered["incident_severity"].dropna().astype(str).unique())
            selected = st.multiselect("Incident severity", options, default=options)
            if selected:
                filtered = filtered[filtered["incident_severity"].astype(str).isin(selected)]
    with col3:
        options = sorted(filtered[target_col].dropna().astype(str).unique())
        selected = st.multiselect("Claim label", options, default=options)
        if selected:
            filtered = filtered[filtered[target_col].astype(str).isin(selected)]
    a, b, c = st.columns(3)
    a.metric("Filtered claims", f"{len(filtered):,}")
    if "total_claim_amount" in filtered:
        b.metric("Average total claim", f"{filtered['total_claim_amount'].mean():,.0f}")
    if "total_claim_amount" in filtered:
        c.metric("Median total claim", f"{filtered['total_claim_amount'].median():,.0f}")
    left, right = st.columns(2)
    with left:
        if "incident_type" in filtered:
            st.markdown("#### Incident type")
            st.bar_chart(filtered["incident_type"].value_counts())
    with right:
        if "incident_severity" in filtered:
            st.markdown("#### Incident severity by reported fraud")
            cross = pd.crosstab(filtered["incident_severity"], filtered[target_col])
            st.bar_chart(cross)
    if {"total_claim_amount", target_col}.issubset(filtered.columns):
        st.markdown("#### Claim amount distribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(data=filtered, x=target_col, y="total_claim_amount", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Total claim amount")
        ax.tick_params(axis="x", rotation=10)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("#### Filtered records")
    st.dataframe(filtered.head(500), use_container_width=True)
    st.download_button("Download filtered data (CSV)", filtered.to_csv(index=False).encode("utf-8"), "filtered_insurance_claims.csv", "text/csv")

elif page == "Predict Claim":
    st.subheader("🧾 Claim risk screening demo")
    st.write("Enter claim details below to see the trained model's estimated class. This is not a final fraud determination.")
    if model is None:
        st.warning("The trained model is not available yet. Run `python train_model.py` to create it, then redeploy.")
    else:
        # Input schema is aligned with training features.
        with st.form("claim_form"):
            st.markdown("#### Policy details")
            a, b, c = st.columns(3)
            months_as_customer = a.number_input("Months as customer", 0, 1000, 120)
            age = b.number_input("Insured age", 18, 100, 38)
            policy_state = c.selectbox("Policy state", sorted(df["policy_state"].dropna().astype(str).unique()))
            policy_csl = a.selectbox("Policy CSL", sorted(df["policy_csl"].dropna().astype(str).unique()))
            deductible = b.selectbox("Policy deductible", sorted(pd.to_numeric(df["policy_deductable"], errors="coerce").dropna().unique().tolist()))
            premium = c.number_input("Annual premium", min_value=0.0, value=float(df["policy_annual_premium"].median()), step=25.0)
            umbrella = a.selectbox("Umbrella limit", sorted(pd.to_numeric(df["umbrella_limit"], errors="coerce").dropna().unique().tolist()))
            sex = b.selectbox("Insured sex", sorted(df["insured_sex"].dropna().astype(str).unique()))
            education = c.selectbox("Education level", sorted(df["insured_education_level"].dropna().astype(str).unique()))
            occupation = a.selectbox("Occupation", sorted(df["insured_occupation"].dropna().astype(str).unique()))
            relationship = b.selectbox("Relationship", sorted(df["insured_relationship"].dropna().astype(str).unique()))
            hobbies = c.selectbox("Hobby", sorted(df["insured_hobbies"].dropna().astype(str).unique()))
            st.markdown("#### Incident details")
            a, b, c = st.columns(3)
            incident_type = a.selectbox("Incident type", sorted(df["incident_type"].dropna().astype(str).unique()))
            severity = b.selectbox("Incident severity", sorted(df["incident_severity"].dropna().astype(str).unique()))
            collision = c.selectbox("Collision type", sorted(df["collision_type"].dropna().astype(str).replace({"<NA>": "Unknown"}).unique()))
            authorities = a.selectbox("Authorities contacted", sorted(df["authorities_contacted"].dropna().astype(str).unique()))
            incident_state = b.selectbox("Incident state", sorted(df["incident_state"].dropna().astype(str).unique()))
            city = c.selectbox("Incident city", sorted(df["incident_city"].dropna().astype(str).unique()))
            hour = a.slider("Incident hour", 0, 23, 12)
            vehicles = b.slider("Vehicles involved", 1, 10, 2)
            bodily = c.slider("Bodily injuries", 0, 10, 1)
            witnesses = a.slider("Witnesses", 0, 10, 1)
            property_damage = b.selectbox("Property damage", ["NO", "YES"])
            police_report = c.selectbox("Police report available", ["NO", "YES"])
            total_claim = a.number_input("Total claim amount", min_value=0.0, value=float(df["total_claim_amount"].median()), step=500.0)
            injury_claim = b.number_input("Injury claim", min_value=0.0, value=float(df["injury_claim"].median()), step=100.0)
            property_claim = c.number_input("Property claim", min_value=0.0, value=float(df["property_claim"].median()), step=100.0)
            vehicle_claim = a.number_input("Vehicle claim", min_value=0.0, value=float(df["vehicle_claim"].median()), step=500.0)
            capital_gains = b.number_input("Capital gains", value=0, step=1000)
            capital_loss = c.number_input("Capital loss", value=0, step=1000)
            number_year = a.selectbox("Vehicle year", sorted(pd.to_numeric(df["auto_year"], errors="coerce").dropna().astype(int).unique().tolist()))
            auto_make = b.selectbox("Vehicle make", sorted(df["auto_make"].dropna().astype(str).unique()))
            auto_model = c.selectbox("Vehicle model", sorted(df["auto_model"].dropna().astype(str).unique()))
            submit = st.form_submit_button("Estimate claim risk", type="primary", use_container_width=True)

        if submit:
            # Create a record using the exact feature columns saved during training.
            record = {
                "months_as_customer": months_as_customer, "age": age,
                "policy_state": policy_state, "policy_csl": policy_csl,
                "policy_deductable": deductible, "policy_annual_premium": premium,
                "umbrella_limit": umbrella, "insured_sex": sex,
                "insured_education_level": education, "insured_occupation": occupation,
                "insured_hobbies": hobbies, "insured_relationship": relationship,
                "capital-gains": capital_gains, "capital-loss": capital_loss,
                "incident_type": incident_type, "collision_type": collision,
                "incident_severity": severity, "authorities_contacted": authorities,
                "incident_state": incident_state, "incident_city": city,
                "incident_hour_of_the_day": hour, "number_of_vehicles_involved": vehicles,
                "property_damage": property_damage, "bodily_injuries": bodily,
                "witnesses": witnesses, "police_report_available": police_report,
                "total_claim_amount": total_claim, "injury_claim": injury_claim,
                "property_claim": property_claim, "vehicle_claim": vehicle_claim,
                "auto_make": auto_make, "auto_model": auto_model, "auto_year": number_year,
            }
            try:
                input_df = pd.DataFrame([record])
                expected = list(getattr(model, "feature_names_in_", input_df.columns))
                for col in expected:
                    if col not in input_df.columns:
                        input_df[col] = pd.NA
                input_df = input_df[expected]
                prediction = int(model.predict(input_df)[0])
                probability = None
                if hasattr(model, "predict_proba"):
                    classes = list(model.classes_)
                    if 1 in classes:
                        probability = float(model.predict_proba(input_df)[0][classes.index(1)])
                st.markdown("### Model result")
                r1, r2 = st.columns(2)
                with r1:
                    if prediction == 1:
                        st.warning("Higher model-estimated fraud-risk pattern")
                    else:
                        st.success("Lower model-estimated fraud-risk pattern")
                with r2:
                    if probability is not None:
                        st.metric("Estimated fraud probability", f"{probability:.1%}")
                st.info("This estimate is for demonstration only. It is not proof of fraud and must not be used alone to deny a claim. Real cases require human review, verified evidence, and applicable legal safeguards.")
            except Exception as exc:
                st.error(f"Prediction could not be completed: {type(exc).__name__}: {exc}")

elif page == "Model Performance":
    st.subheader("📈 Model performance")
    if not metrics:
        st.warning("Metrics are not available. Run `python train_model.py` and include models/metrics.json in the repository.")
    else:
        comparison = metrics.get("model_comparison", [])
        if comparison:
            comp = pd.DataFrame(comparison)
            st.dataframe(comp, use_container_width=True, hide_index=True)
            metric_choice = st.selectbox("Compare metric", ["accuracy", "precision", "recall", "f1"])
            if metric_choice in comp.columns and "model" in comp.columns:
                st.bar_chart(comp.set_index("model")[[metric_choice]])
        best = metrics.get("best_model")
        if best:
            st.success(f"Selected model by test F1 score: {best}")
        report = metrics.get("classification_report")
        if report:
            st.markdown("#### Classification report")
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        cm = metrics.get("confusion_matrix")
        if cm:
            st.markdown("#### Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No fraud", "Fraud"], yticklabels=["No fraud", "Fraud"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Metrics are calculated on a held-out test split. They describe this dataset and do not guarantee performance on future claims.")

st.divider()
st.caption("Insurance Claim Intelligence • Educational portfolio project • Human review required for real-world use")
