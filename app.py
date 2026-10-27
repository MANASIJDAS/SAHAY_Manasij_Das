import os
import zipfile
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.svm import SVC


st.set_page_config(
    page_title="SAHAY | Stress & Welfare Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


STRESS_LEVELS = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Extremely Severe",
]
EXPECTED_MODEL_NAMES = ["Random Forest", "Logistic Regression", "KNN", "SVM"]
MODEL_DISPLAY_ORDER = ["Random Forest", "SVM", "Logistic Regression", "KNN"]


st.markdown(
    """
    <style>

    .stApp {
        background: #fffbea;
        color: #302a20;
    }

    section[data-testid="stSidebar"] {
        background: #fff7d6;
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #fff4b8,
            #fffdf1
        );
        border: 1px solid #ead88a;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
        color: #3b321d !important;
    }

    .hero p {
        font-size: 17px;
        color: #62583d !important;
    }

    .card {
        background: #fffdf4;
        border: 1px solid #eadfba;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0px 3px 12px rgba(80, 70, 30, 0.07);
    }

    .metric-card {
        background: #fffdf4;
        border: 1px solid #eadfba;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        color: #756b50 !important;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #332b1c !important;
        font-size: 28px;
        font-weight: 700;
    }

    .risk-normal {
        color: #347a46 !important;
        font-size: 30px;
        font-weight: 700;
    }

    .risk-mild {
        color: #a17819 !important;
        font-size: 30px;
        font-weight: 700;
    }

    .risk-moderate {
        color: #c36c13 !important;
        font-size: 30px;
        font-weight: 700;
    }

    .risk-severe {
        color: #bd4b32 !important;
        font-size: 30px;
        font-weight: 700;
    }

    .risk-extreme {
        color: #9d3030 !important;
        font-size: 30px;
        font-weight: 700;
    }

    .recommendation {
        background: #fff7d6;
        border-left: 5px solid #d8b62d;
        padding: 18px;
        border-radius: 12px;
        font-size: 16px;
        line-height: 1.6;
    }

    .training-message {
        background: #000000 !important;
        color: #ffffff !important;
        padding: 18px 24px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0;
    }

    .training-message * {
        color: #ffffff !important;
    }

    .stApp *,
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    .training-message,
    .training-message * {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def apply_black_plotly_text(fig):
    fig.update_layout(
        font=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )

    if hasattr(fig.layout, "legend") and fig.layout.legend is not None:
        fig.update_layout(
            legend=dict(font=dict(color="#000000"))
        )

    if hasattr(fig.layout, "xaxis") and fig.layout.xaxis is not None:
        fig.update_layout(
            xaxis=dict(
                title_font=dict(color="#000000"),
                tickfont=dict(color="#000000"),
            )
        )

    if hasattr(fig.layout, "yaxis") and fig.layout.yaxis is not None:
        fig.update_layout(
            yaxis=dict(
                title_font=dict(color="#000000"),
                tickfont=dict(color="#000000"),
            )
        )

    for trace in fig.data:
        if hasattr(trace, "textfont"):
            try:
                trace.textfont = dict(color="#000000")
            except Exception:
                pass

    return fig


def get_model_probabilities(model, input_df):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        return dict(zip(model.classes_, probs))

    if hasattr(model, "decision_function"):
        scores = model.decision_function(input_df)[0]
        scores = np.asarray(scores, dtype=float)

        if scores.ndim == 0:
            scores = np.array([scores])

        scores = scores - np.max(scores)
        exps = np.exp(scores)
        probs = exps / exps.sum()
        return dict(zip(model.classes_, probs))

    class_count = len(model.classes_)
    uniform = 1.0 / class_count
    return {cls: uniform for cls in model.classes_}


def calculate_model_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


FEATURE_GROUPS = {
    "Lack of Positive Feelings": ["Q3A", "Q5A", "Q16A"],
    "Lack of Motivation & Drive": ["Q10A", "Q17A", "Q31A"],
    "Hopelessness & Meaninglessness": ["Q21A", "Q37A", "Q38A"],
    "Low Mood & Self-Worth": ["Q13A", "Q26A", "Q34A"],
    "Lack of Interest & Enjoyment": ["Q24A", "Q42A", "Q29A"],
    "Physical Anxiety Symptoms": ["Q2A", "Q7A", "Q23A"],
    "Breathing & Heart Anxiety Symptoms": ["Q4A", "Q25A", "Q41A"],
    "Fear & Panic Symptoms": ["Q9A", "Q20A", "Q28A"],
    "Situational Anxiety Symptoms": ["Q15A", "Q30A", "Q36A"],
    "Nervousness & Tension": ["Q6A", "Q18A", "Q39A"],
    "Irritability & Agitation": ["Q8A", "Q11A", "Q27A"],
    "Difficulty Relaxing": ["Q12A", "Q22A", "Q32A"],
    "Stress Intolerance": ["Q14A", "Q33A", "Q35A"],
    "Emotional Reactivity": ["Q1A", "Q19A", "Q40A"],
}

STRESS_ITEMS = [
    "Q1A",
    "Q6A",
    "Q8A",
    "Q11A",
    "Q12A",
    "Q14A",
    "Q18A",
    "Q22A",
    "Q27A",
    "Q29A",
    "Q32A",
    "Q33A",
    "Q35A",
    "Q39A",
]

DEPRESSION_ITEMS = [
    "Q3A",
    "Q5A",
    "Q10A",
    "Q13A",
    "Q16A",
    "Q17A",
    "Q21A",
    "Q24A",
    "Q26A",
    "Q31A",
    "Q34A",
    "Q37A",
    "Q38A",
    "Q42A",
]

ANXIETY_ITEMS = [
    "Q2A",
    "Q4A",
    "Q7A",
    "Q9A",
    "Q15A",
    "Q19A",
    "Q20A",
    "Q23A",
    "Q25A",
    "Q28A",
    "Q30A",
    "Q36A",
    "Q40A",
    "Q41A",
]

ZIP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DASS_data_21.02.19 (1).zip",
)


@st.cache_data
def load_dataset():
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError("DASS ZIP file was not found at: " + ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        csv_files = [
            name for name in z.namelist() if name.lower().endswith("data.csv")
        ]

        if not csv_files:
            raise FileNotFoundError("data.csv was not found inside the ZIP file.")

        with z.open(csv_files[0]) as f:
            data = pd.read_csv(f, sep="\t")

    return data


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sahay_models.joblib",
)
METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sahay_model_metrics.json",
)


def load_saved_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "The metrics JSON file was not found: " + METRICS_PATH
        )

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The saved metrics file is not valid JSON: " + METRICS_PATH
        ) from exc

    required_top_level = {
        "project",
        "training_records",
        "testing_records",
        "feature_names",
        "feature_groups",
        "stress_items",
        "stress_levels",
        "models",
    }
    missing_keys = sorted(required_top_level.difference(metrics.keys()))
    if missing_keys:
        raise ValueError(
            "The metrics file is missing required keys: " + ", ".join(missing_keys)
        )

    if metrics["feature_names"] != list(FEATURE_GROUPS.keys()):
        raise ValueError(
            "The feature names in sahay_model_metrics.json do not match the app configuration."
        )
    if metrics["feature_groups"] != FEATURE_GROUPS:
        raise ValueError(
            "The feature-group mappings in sahay_model_metrics.json do not match the app configuration."
        )
    if metrics["stress_items"] != STRESS_ITEMS:
        raise ValueError(
            "The stress-item list in sahay_model_metrics.json does not match the app configuration."
        )

    missing_models = [name for name in EXPECTED_MODEL_NAMES if name not in metrics["models"]]
    if missing_models:
        raise ValueError(
            "The metrics file is missing model entries for: "
            + ", ".join(missing_models)
        )

    return metrics


@st.cache_resource
def train_model():
    """Load the already-trained models and evaluate them on the held-out split."""
    data = load_dataset()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The trained model file was not found: "
            + MODEL_PATH
            + ". Add sahay_models.joblib beside app.py."
        )

    bundle = joblib.load(MODEL_PATH)
    saved_metrics = load_saved_metrics()

    required_bundle_keys = {
        "models",
        "feature_names",
        "feature_groups",
        "stress_items",
        "stress_levels",
    }
    missing_keys = required_bundle_keys.difference(bundle.keys())
    if missing_keys:
        raise ValueError(
            "The model bundle is missing required metadata: "
            + ", ".join(sorted(missing_keys))
        )

    if bundle["feature_names"] != list(FEATURE_GROUPS.keys()):
        raise ValueError(
            "Feature names/order in sahay_models.joblib do not match app.py."
        )
    if bundle["feature_groups"] != FEATURE_GROUPS:
        raise ValueError(
            "DASS question mappings in sahay_models.joblib do not match app.py."
        )
    if bundle["stress_items"] != STRESS_ITEMS:
        raise ValueError(
            "Stress-item definitions in sahay_models.joblib do not match app.py."
        )

    required_items = list(dict.fromkeys(
        item
        for items in FEATURE_GROUPS.values()
        for item in items
    ))
    required_items = list(dict.fromkeys(required_items + STRESS_ITEMS))
    missing_columns = [item for item in required_items if item not in data.columns]
    if missing_columns:
        raise ValueError("Missing DASS columns: " + ", ".join(missing_columns))

    dass_scores = data[required_items].apply(pd.to_numeric, errors="coerce") - 1

    X = pd.DataFrame(index=data.index)
    for feature_name, items in FEATURE_GROUPS.items():
        X[feature_name] = dass_scores[items].mean(axis=1)
    X = X[list(FEATURE_GROUPS.keys())]

    stress_sum = dass_scores[STRESS_ITEMS].sum(
        axis=1, min_count=len(STRESS_ITEMS)
    )
    stress_score = stress_sum * 2

    def stress_level(score):
        if pd.isna(score):
            return np.nan
        if score <= 14:
            return "Normal"
        if score <= 18:
            return "Mild"
        if score <= 25:
            return "Moderate"
        if score <= 33:
            return "Severe"
        return "Extremely Severe"

    y = stress_score.apply(stress_level)
    valid = X.notna().any(axis=1) & y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = bundle["models"]
    missing_models = [name for name in EXPECTED_MODEL_NAMES if name not in models]
    if missing_models:
        raise ValueError(
            "The model bundle is missing: " + ", ".join(missing_models)
        )

    results = {}
    for model_name in MODEL_DISPLAY_ORDER:
        estimator = models[model_name]
        predictions = estimator.predict(X_test)
        classifier = estimator.named_steps.get("classifier", estimator)

        if hasattr(classifier, "feature_importances_"):
            importance_values = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            coef_values = np.asarray(classifier.coef_)
            importance_values = np.abs(coef_values).mean(axis=0)
        else:
            importance_values = np.zeros(len(X.columns))

        importance_df_model = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance_values,
        }).sort_values("Importance", ascending=True)

        metrics = calculate_model_metrics(y_test, predictions)
        results[model_name] = {
            "estimator": estimator,
            "metrics": metrics,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "weighted_f1": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=STRESS_LEVELS,
            ),
            "importance_df": importance_df_model,
            "y_pred": predictions,
        }

    best_model_name = max(
        results,
        key=lambda model_name: results[model_name]["f1"],
    )
    best_result = results[best_model_name]
    model = best_result["estimator"]

    saved_best_model = saved_metrics.get("best_model_name_by_macro_f1")
    if saved_best_model and saved_best_model in results:
        best_model_name = saved_best_model
        model = results[best_model_name]["estimator"]

    distribution = y.value_counts().reindex(STRESS_LEVELS, fill_value=0)
    distribution = distribution.reset_index()
    distribution.columns = ["Stress Level", "Count"]

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "results": results,
        "distribution": distribution,
        "best_model_name": best_model_name,
        "saved_metrics": saved_metrics,
        "metrics_best_model": saved_metrics.get("best_model_name_by_macro_f1"),
    }


def compute_cv_macro_f1_scores(results, X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model_rows = []

    for model_name in MODEL_DISPLAY_ORDER:
        estimator = results[model_name]["estimator"]
        fold_scores = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
        )
        model_rows.append(
            {
                "Model": model_name,
                "Fold 1 Macro F1": fold_scores[0],
                "Fold 2 Macro F1": fold_scores[1],
                "Fold 3 Macro F1": fold_scores[2],
                "Mean Macro F1": fold_scores.mean(),
            }
        )

    return pd.DataFrame(model_rows).round(4)


loading_message = st.empty()
loading_message.markdown(
    """
    <div class="training-message">
        Hold a bit of patience! The model is getting trained.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model_bundle = train_model()
except Exception as e:
    loading_message.empty()
    st.error(str(e))
    st.stop()

loading_message.empty()

model = model_bundle["model"]
X = model_bundle["X"]
y = model_bundle["y"]
X_test = model_bundle["X_test"]
y_test = model_bundle["y_test"]
results = model_bundle["results"]
distribution = model_bundle["distribution"]
best_model_name = model_bundle["best_model_name"]
saved_metrics = model_bundle["saved_metrics"]
metrics_best_model = model_bundle["metrics_best_model"]

st.markdown(
    """
    <div class="hero">

    <h1>🛡️ SAHAY</h1>

    <p>
    SAHAY is an AI-Based Predictive Stress and Welfare Monitoring System
    </p>

    <p style="font-style: italic;">
    "There is no health without mental health"
    </p>

    <p>
    — David Satcher
    </p>

    <p>
    DASS-42 based stress-category classification prototype
    powered by the best-performing trained model.
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## SAHAY")
    st.markdown("### Wellbeing Assessment")
    st.info("Enter each response from 0 to 3.")
    st.markdown(
        """
        **Response Scale**

        0 — Did not apply to me at all

        1 — Applied to me to some degree

        2 — Applied to me to a considerable degree

        3 — Applied to me very much / most of the time
        """
    )
    st.markdown("### Prediction Model")
    default_model_index = (
        MODEL_DISPLAY_ORDER.index(best_model_name)
        if best_model_name in MODEL_DISPLAY_ORDER
        else 0
    )
    selected_model_name = st.radio(
        "Choose the model for individual prediction",
        options=MODEL_DISPLAY_ORDER,
        index=default_model_index,
        key="prediction_model_selector",
        horizontal=True,
    )
    selected_model = results[selected_model_name]["estimator"]

st.markdown("## Wellbeing Assessment")
st.markdown("Enter the individual's responses to the 14 grouped assessment dimensions.")

user_values = {}
feature_names = list(FEATURE_GROUPS.keys())

for row_start in range(0, len(feature_names), 3):
    cols = st.columns(3)
    for col_index, feature_name in enumerate(feature_names[row_start : row_start + 3]):
        with cols[col_index]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {feature_name}")
            st.caption("DASS items: " + ", ".join(FEATURE_GROUPS[feature_name]))
            value = st.slider(
                "Response",
                min_value=0,
                max_value=3,
                value=0,
                step=1,
                key="input_" + feature_name,
            )
            user_values[feature_name] = value
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")


if st.button("🔍 Assess Stress Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([user_values], columns=feature_names)

    prediction_model = selected_model
    prediction = prediction_model.predict(input_df)[0]
    probability_dict = get_model_probabilities(prediction_model, input_df)
    classes = list(probability_dict.keys())
    probabilities = [probability_dict[x] for x in classes]

    normal_probability = probability_dict.get("Normal", 0)
    risk_probability = (1 - normal_probability) * 100
    confidence = max(probabilities) * 100

    score_map = {
        "Normal": 20,
        "Mild": 35,
        "Moderate": 55,
        "Severe": 75,
        "Extremely Severe": 90,
    }
    risk_score = score_map[prediction]

    if prediction == "Normal":
        warning = "No Immediate Warning"
        recommendation = (
            "Maintain healthy sleep, recovery, social support, and regular wellbeing practices."
        )
        risk_class = "risk-normal"
    elif prediction == "Mild":
        warning = "Monitor"
        recommendation = (
            "Consider monitoring stress levels and maintaining adequate recovery, sleep, and peer support."
        )
        risk_class = "risk-mild"
    elif prediction == "Moderate":
        warning = "Attention Required"
        recommendation = (
            "Consider a welfare check-in and additional recovery support. Persistent or worsening symptoms should be referred through appropriate professional channels."
        )
        risk_class = "risk-moderate"
    elif prediction == "Severe":
        warning = "Attention Required"
        recommendation = (
            "Prioritize a welfare check-in and appropriate support. Professional assessment should be considered where symptoms are persistent or worsening."
        )
        risk_class = "risk-severe"
    else:
        warning = "Attention Required"
        recommendation = (
            "Prompt welfare attention is recommended. Appropriate professional support should be considered."
        )
        risk_class = "risk-extreme"

    st.markdown("## Assessment Result")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">Stress Risk Level</div>
            <div class="{risk_class}">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">Risk Probability</div>
            <div class="metric-value">{risk_probability:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">Overall Risk Score</div>
            <div class="metric-value">{risk_score}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    c4, c5 = st.columns(2)

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">Early-Warning Flag</div>
            <div class="metric-value">{warning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">Model Confidence</div>
            <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Welfare Recommendation")
    st.markdown(
        f"""
        <div class="recommendation">{recommendation}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Distribution")
        risk_df = pd.DataFrame(
            {"Stress Level": list(probability_dict.keys()), "Probability": [probability_dict[x] * 100 for x in probability_dict]}
        )
        fig = px.bar(risk_df, x="Stress Level", y="Probability", text="Probability")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            plot_bgcolor="#fffdf4",
            paper_bgcolor="#fffdf4",
        )
        apply_black_plotly_text(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Overall Risk Score")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Risk Score", "font": {"color": "#000000"}},
                number={"font": {"color": "#000000"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#000000"}},
                    "threshold": {"line": {"width": 4}, "value": risk_score},
                },
            )
        )
        gauge.update_layout(paper_bgcolor="#fffdf4")
        apply_black_plotly_text(gauge)
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("### Input Profile")
    input_chart_df = pd.DataFrame(
        {"Feature": list(user_values.keys()), "Response": list(user_values.values())}
    )
    fig_input = px.bar(
        input_chart_df,
        x="Response",
        y="Feature",
        orientation="h",
        range_x=[0, 3],
    )
    fig_input.update_layout(
        plot_bgcolor="#fffdf4",
        paper_bgcolor="#fffdf4",
        xaxis_title="DASS Response",
        yaxis_title="",
    )
    apply_black_plotly_text(fig_input)
    st.plotly_chart(fig_input, use_container_width=True)

st.markdown("---")
st.markdown("## Model Performance")
st.caption("Performance metrics are calculated on the held-out test set.")

if metrics_best_model:
    st.caption(
        f"Primary application model: {best_model_name}. "
        f"Best macro-F1 model recorded in the saved metrics JSON: {metrics_best_model}."
    )
else:
    st.caption(f"Primary application model: {best_model_name}")

st.markdown("### All Four Model Metrics")
for model_name in MODEL_DISPLAY_ORDER:
    model_result = results[model_name]
    model_metrics = model_result["metrics"]
    st.markdown(f"#### {model_name}")
    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Score": model_metrics["accuracy"],
            },
            {
                "Metric": "Macro Precision",
                "Score": model_metrics["precision"],
            },
            {
                "Metric": "Macro Recall",
                "Score": model_metrics["recall"],
            },
            {
                "Metric": "Macro F1-score",
                "Score": model_metrics["f1"],
            },
        ]
    )
    st.dataframe(
        metrics_df.assign(Score=lambda df: (df["Score"] * 100).round(2)).style.format({"Score": "{:.2f}%"}),
        use_container_width=True,
    )

    cm_df = pd.DataFrame(
        model_result["confusion_matrix"],
        index=STRESS_LEVELS,
        columns=STRESS_LEVELS,
    )
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd")
    fig_cm.update_layout(
        title=f"{model_name} Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        plot_bgcolor="#fffdf4",
        paper_bgcolor="#fffdf4",
    )
    apply_black_plotly_text(fig_cm)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("---")

st.markdown("## Comparative Analysis of Machine Learning Models")
comparison_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Accuracy": results[model_name]["metrics"]["accuracy"],
            "Macro Precision": results[model_name]["metrics"]["precision"],
            "Macro Recall": results[model_name]["metrics"]["recall"],
            "Macro F1-score": results[model_name]["metrics"]["f1"],
        }
        for model_name in MODEL_DISPLAY_ORDER
    ]
)

st.dataframe(
    comparison_df.assign(
        Accuracy=lambda df: (df["Accuracy"] * 100).round(2),
        **{
            "Macro Precision": lambda df: (df["Macro Precision"] * 100).round(2),
            "Macro Recall": lambda df: (df["Macro Recall"] * 100).round(2),
            "Macro F1-score": lambda df: (df["Macro F1-score"] * 100).round(2),
        },
    ).style.format({
        "Accuracy": "{:.2f}%",
        "Macro Precision": "{:.2f}%",
        "Macro Recall": "{:.2f}%",
        "Macro F1-score": "{:.2f}%",
    }),
    use_container_width=True,
)

metric_df = comparison_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)
fig_comparison = px.bar(
    metric_df,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text="Score",
)
fig_comparison.update_traces(texttemplate="%{text:.2%}", textposition="outside")
fig_comparison.update_layout(
    yaxis_title="Score",
    xaxis_title="Metric",
    yaxis_range=[0, 1.0],
    plot_bgcolor="#fffdf4",
    paper_bgcolor="#fffdf4",
    legend_title_text="Model",
)
apply_black_plotly_text(fig_comparison)
st.plotly_chart(fig_comparison, use_container_width=True)

best_by_metric = {
    metric: comparison_df.loc[comparison_df[metric].idxmax(), "Model"]
    for metric in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"]
}

st.markdown("### Best Model by Metric")
for metric, model_name in best_by_metric.items():
    value = comparison_df.loc[comparison_df["Model"] == model_name, metric].iloc[0]
    st.write(f"- {metric}: {model_name} ({value * 100:.2f}%)")

best_model_summary = max(
    comparison_df[["Model", "Macro F1-score"]].itertuples(index=False),
    key=lambda row: row[1],
)
model_summary = (
    f"The strongest overall test-set performance is {best_model_summary[0]} "
    f"with macro F1 = {best_model_summary[1] * 100:.2f}%. "
)
if best_model_summary[0] == "SVM":
    model_summary += "SVM is slightly ahead of Logistic Regression on the held-out dataset, while Random Forest and KNN remain competitive but lower in the macro-F1 comparison."
elif best_model_summary[0] == "Logistic Regression":
    model_summary += "Logistic Regression leads the held-out macro-F1 comparison, with SVM close behind and Random Forest and KNN trailing on this metric."
elif best_model_summary[0] == "Random Forest":
    model_summary += "Random Forest leads the held-out macro-F1 comparison, but it remains behind the better-balanced SVM and Logistic Regression performance on the full metric set."
else:
    model_summary += "KNN remains competitive on selected categories but trails the SVM and Logistic Regression models on macro recall and macro F1."

st.info(model_summary)

st.markdown("### Dataset Stress-Level Distribution")
fig_dist = px.bar(distribution, x="Stress Level", y="Count", text="Count")
fig_dist.update_layout(
    xaxis_title="Stress Level",
    yaxis_title="Number of Records",
    plot_bgcolor="#fffdf4",
    paper_bgcolor="#fffdf4",
)
apply_black_plotly_text(fig_dist)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("### Cross-Validation")
cv_scores_df = compute_cv_macro_f1_scores(results, X, y)
st.dataframe(cv_scores_df, use_container_width=True)

st.markdown("### Dataset Information")
d1, d2, d3 = st.columns(3)
with d1:
    st.metric("Dataset Records", f"{len(y):,}")
with d2:
    st.metric("Input Features", len(FEATURE_GROUPS))
with d3:
    st.metric("DASS Items Represented", 42)

st.markdown(
    """
    <div class="card">
    <b>Important note:</b>
    SAHAY is a research/academic prototype based on DASS-42
    responses. The displayed risk probability, risk score,
    confidence, and welfare recommendations are application-level
    indicators and should not be interpreted as clinical diagnosis
    or definitive individual risk probabilities.
    </div>
    """,
    unsafe_allow_html=True,
)
