import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


st.set_page_config(
    page_title="SAHAY | Stress & Welfare Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    unsafe_allow_html=True
)


def apply_black_plotly_text(fig):

    fig.update_layout(
        font=dict(
            color="#000000"
        ),
        title_font=dict(
            color="#000000"
        )
    )

    if hasattr(fig.layout, "legend") and fig.layout.legend is not None:
        fig.update_layout(
            legend=dict(
                font=dict(
                    color="#000000"
                )
            )
        )

    if hasattr(fig.layout, "xaxis") and fig.layout.xaxis is not None:
        fig.update_layout(
            xaxis=dict(
                title_font=dict(
                    color="#000000"
                ),
                tickfont=dict(
                    color="#000000"
                )
            )
        )

    if hasattr(fig.layout, "yaxis") and fig.layout.yaxis is not None:
        fig.update_layout(
            yaxis=dict(
                title_font=dict(
                    color="#000000"
                ),
                tickfont=dict(
                    color="#000000"
                )
            )
        )

    for trace in fig.data:

        if hasattr(trace, "textfont"):
            try:
                trace.textfont = dict(
                    color="#000000"
                )
            except Exception:
                pass

    return fig


FEATURE_GROUPS = {
    "Lack of Positive Feelings": [
        "Q3A",
        "Q5A",
        "Q16A"
    ],
    "Lack of Motivation & Drive": [
        "Q10A",
        "Q17A",
        "Q31A"
    ],
    "Hopelessness & Meaninglessness": [
        "Q21A",
        "Q37A",
        "Q38A"
    ],
    "Low Mood & Self-Worth": [
        "Q13A",
        "Q26A",
        "Q34A"
    ],
    "Lack of Interest & Enjoyment": [
        "Q24A",
        "Q42A",
        "Q29A"
    ],
    "Physical Anxiety Symptoms": [
        "Q2A",
        "Q7A",
        "Q23A"
    ],
    "Breathing & Heart Anxiety Symptoms": [
        "Q4A",
        "Q25A",
        "Q41A"
    ],
    "Fear & Panic Symptoms": [
        "Q9A",
        "Q20A",
        "Q28A"
    ],
    "Situational Anxiety Symptoms": [
        "Q15A",
        "Q30A",
        "Q36A"
    ],
    "Nervousness & Tension": [
        "Q6A",
        "Q18A",
        "Q39A"
    ],
    "Irritability & Agitation": [
        "Q8A",
        "Q11A",
        "Q27A"
    ],
    "Difficulty Relaxing": [
        "Q12A",
        "Q22A",
        "Q32A"
    ],
    "Stress Intolerance": [
        "Q14A",
        "Q33A",
        "Q35A"
    ],
    "Emotional Reactivity": [
        "Q1A",
        "Q19A",
        "Q40A"
    ]
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
    "Q39A"
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
    "Q42A"
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
    "Q41A"
]


ZIP_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "DASS_data_21.02.19 (1).zip"
)


@st.cache_data
def load_dataset():

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            "DASS ZIP file was not found at: " + ZIP_PATH
        )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith("data.csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "data.csv was not found inside the ZIP file."
            )

        with z.open(
            csv_files[0]
        ) as f:

            data = pd.read_csv(
                f,
                sep="\t"
            )

    return data


@st.cache_resource
def train_model():

    data = load_dataset()

    required_items = []

    for items in FEATURE_GROUPS.values():
        required_items.extend(items)

    required_items = list(
        dict.fromkeys(
            required_items
        )
    )

    missing = [
        item
        for item in required_items
        if item not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing DASS columns: " +
            ", ".join(missing)
        )

    dass_scores = data[
        required_items
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ) - 1

    X = pd.DataFrame(
        index=data.index
    )

    for feature_name, items in FEATURE_GROUPS.items():

        X[feature_name] = (
            dass_scores[
                items
            ].mean(
                axis=1
            )
        )

    stress_sum = dass_scores[
        STRESS_ITEMS
    ].sum(
        axis=1,
        min_count=len(STRESS_ITEMS)
    )

    stress_score = stress_sum * 2

    def stress_level(score):

        if pd.isna(score):
            return np.nan

        if score <= 14:
            return "Normal"

        elif score <= 18:
            return "Mild"

        elif score <= 25:
            return "Moderate"

        elif score <= 33:
            return "Severe"

        else:
            return "Extremely Severe"

    y = stress_score.apply(
        stress_level
    )

    valid = (
        X.notna().any(
            axis=1
        )
        & y.notna()
    )

    X = X.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=1
                )
            )
        ]
    )

    param_grid = {
        "classifier__n_estimators": [
            100,
            150
        ],
        "classifier__max_depth": [
            None,
            15
        ],
        "classifier__min_samples_split": [
            2
        ],
        "classifier__min_samples_leaf": [
            1
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    grid.fit(
        X_train,
        y_train
    )

    model = grid.best_estimator_

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    best_index = grid.best_index_

    cv_scores = np.array(
        [
            grid.cv_results_[
                f"split{i}_test_score"
            ][best_index]
            for i in range(
                cv.n_splits
            )
        ]
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "Normal",
            "Mild",
            "Moderate",
            "Severe",
            "Extremely Severe"
        ]
    )

    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importances
        }
    ).sort_values(
        "Importance",
        ascending=True
    )

    distribution = (
        y.value_counts()
        .reindex(
            [
                "Normal",
                "Mild",
                "Moderate",
                "Severe",
                "Extremely Severe"
            ],
            fill_value=0
        )
        .reset_index()
    )

    distribution.columns = [
        "Stress Level",
        "Count"
    ]

    return (
        model,
        X,
        y,
        X_test,
        y_test,
        accuracy,
        precision,
        recall,
        f1,
        cv_scores,
        cm,
        importance_df,
        distribution,
        grid.best_params_
    )


loading_message = st.empty()

loading_message.markdown(
    """
    <div class="training-message">
        Hold a bit of patience! The model is getting trained.
    </div>
    """,
    unsafe_allow_html=True
)


try:

    (
        model,
        X,
        y,
        X_test,
        y_test,
        accuracy,
        precision,
        recall,
        f1,
        cv_scores,
        cm,
        importance_df,
        distribution,
        best_params
    ) = train_model()

except Exception as e:

    loading_message.empty()

    st.error(
        str(e)
    )

    st.stop()


loading_message.empty()


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
    DASS-42 based stress screening and welfare-support prototype
    powered by Random Forest.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:

    st.markdown(
        "## SAHAY"
    )

    st.markdown(
        "### Wellbeing Assessment"
    )

    st.info(
        "Enter each response from 0 to 3."
    )

    st.markdown(
        """
        **Response Scale**

        0 — Did not apply to me at all

        1 — Applied to me to some degree

        2 — Applied to me to a considerable degree

        3 — Applied to me very much / most of the time
        """
    )


st.markdown(
    "## Wellbeing Assessment"
)

st.markdown(
    "Enter the individual's responses to the 14 grouped assessment dimensions."
)


user_values = {}

feature_names = list(
    FEATURE_GROUPS.keys()
)


for row_start in range(
    0,
    len(feature_names),
    3
):

    cols = st.columns(3)

    for col_index, feature_name in enumerate(
        feature_names[
            row_start:row_start + 3
        ]
    ):

        with cols[col_index]:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"### {feature_name}"
            )

            st.caption(
                "DASS items: " +
                ", ".join(
                    FEATURE_GROUPS[
                        feature_name
                    ]
                )
            )

            value = st.slider(
                "Response",
                min_value=0,
                max_value=3,
                value=0,
                step=1,
                key="input_" + feature_name
            )

            user_values[
                feature_name
            ] = value

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


st.markdown("---")


if st.button(
    "🔍 Assess Stress Risk",
    use_container_width=True,
    type="primary"
):

    input_df = pd.DataFrame(
        [user_values],
        columns=feature_names
    )

    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    classes = model.classes_

    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )

    normal_probability = probability_dict.get(
        "Normal",
        0
    )

    risk_probability = (
        1 - normal_probability
    ) * 100

    confidence = (
        max(probabilities) * 100
    )

    score_map = {
        "Normal": 20,
        "Mild": 35,
        "Moderate": 55,
        "Severe": 75,
        "Extremely Severe": 90
    }

    risk_score = score_map[
        prediction
    ]

    if prediction == "Normal":

        warning = "No Immediate Warning"

        recommendation = (
            "Maintain healthy sleep, recovery, social support, "
            "and regular wellbeing practices."
        )

        risk_class = "risk-normal"

    elif prediction == "Mild":

        warning = "Monitor"

        recommendation = (
            "Consider monitoring stress levels and maintaining "
            "adequate recovery, sleep, and peer support."
        )

        risk_class = "risk-mild"

    elif prediction == "Moderate":

        warning = "Attention Required"

        recommendation = (
            "Consider a welfare check-in and additional recovery "
            "support. Persistent or worsening symptoms should be "
            "referred through appropriate professional channels."
        )

        risk_class = "risk-moderate"

    elif prediction == "Severe":

        warning = "Attention Required"

        recommendation = (
            "Prioritize a welfare check-in and appropriate support. "
            "Professional assessment should be considered where "
            "symptoms are persistent or worsening."
        )

        risk_class = "risk-severe"

    else:

        warning = "Attention Required"

        recommendation = (
            "Prompt welfare attention is recommended. Appropriate "
            "professional support should be considered."
        )

        risk_class = "risk-extreme"


    st.markdown(
        "## Assessment Result"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Stress Risk Level
            </div>

            <div class="{risk_class}">
            {prediction}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Risk Probability
            </div>

            <div class="metric-value">
            {risk_probability:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Overall Risk Score
            </div>

            <div class="metric-value">
            {risk_score}/100
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    c4, c5 = st.columns(2)


    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Early-Warning Flag
            </div>

            <div class="metric-value">
            {warning}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c5:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Model Confidence
            </div>

            <div class="metric-value">
            {confidence:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    st.markdown(
        "### Welfare Recommendation"
    )

    st.markdown(
        f"""
        <div class="recommendation">
        {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("")


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### Risk Distribution"
        )

        risk_df = pd.DataFrame(
            {
                "Stress Level": list(
                    probability_dict.keys()
                ),
                "Probability": [
                    probability_dict[x] * 100
                    for x in probability_dict
                ]
            }
        )

        fig = px.bar(
            risk_df,
            x="Stress Level",
            y="Probability",
            text="Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            plot_bgcolor="#fffdf4",
            paper_bgcolor="#fffdf4"
        )

        apply_black_plotly_text(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### Overall Risk Score"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={
                    "text": "Risk Score",
                    "font": {
                        "color": "#000000"
                    }
                },
                number={
                    "font": {
                        "color": "#000000"
                    }
                },
                gauge={
                    "axis": {
                        "range": [
                            0,
                            100
                        ],
                        "tickfont": {
                            "color": "#000000"
                        }
                    },
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "value": risk_score
                    }
                }
            )
        )

        gauge.update_layout(
            paper_bgcolor="#fffdf4"
        )

        apply_black_plotly_text(
            gauge
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


    st.markdown(
        "### Input Profile"
    )


    input_chart_df = pd.DataFrame(
        {
            "Feature": list(
                user_values.keys()
            ),
            "Response": list(
                user_values.values()
            )
        }
    )


    fig_input = px.bar(
        input_chart_df,
        x="Response",
        y="Feature",
        orientation="h",
        range_x=[
            0,
            3
        ]
    )


    fig_input.update_layout(
        plot_bgcolor="#fffdf4",
        paper_bgcolor="#fffdf4",
        xaxis_title="DASS Response",
        yaxis_title=""
    )


    apply_black_plotly_text(
        fig_input
    )


    st.plotly_chart(
        fig_input,
        use_container_width=True
    )


st.markdown("---")


st.markdown(
    "## Model Performance"
)


st.caption(
    "Performance metrics are calculated on the held-out test set. "
    "Five-fold cross-validation is used during model evaluation."
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Accuracy
        </div>

        <div class="metric-value">
        {accuracy * 100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Macro Precision
        </div>

        <div class="metric-value">
        {precision * 100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Macro Recall
        </div>

        <div class="metric-value">
        {recall * 100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Macro F1
        </div>

        <div class="metric-value">
        {f1 * 100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("")


perf_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Score": [
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100
        ]
    }
)


fig_perf = px.bar(
    perf_df,
    x="Metric",
    y="Score",
    text="Score"
)


fig_perf.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig_perf.update_layout(
    yaxis_range=[
        0,
        100
    ],
    yaxis_title="Score (%)",
    plot_bgcolor="#fffdf4",
    paper_bgcolor="#fffdf4"
)


apply_black_plotly_text(
    fig_perf
)


st.plotly_chart(
    fig_perf,
    use_container_width=True
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### Confusion Matrix"
    )

    labels = [
        "Normal",
        "Mild",
        "Moderate",
        "Severe",
        "Extremely Severe"
    ]

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )


    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto"
    )


    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        plot_bgcolor="#fffdf4",
        paper_bgcolor="#fffdf4"
    )


    apply_black_plotly_text(
        fig_cm
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


with col2:

    st.markdown(
        "### Feature Importance"
    )


    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )


    fig_imp.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        plot_bgcolor="#fffdf4",
        paper_bgcolor="#fffdf4"
    )


    apply_black_plotly_text(
        fig_imp
    )


    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )


st.markdown(
    "### Dataset Stress-Level Distribution"
)


fig_dist = px.bar(
    distribution,
    x="Stress Level",
    y="Count",
    text="Count"
)


fig_dist.update_layout(
    xaxis_title="Stress Level",
    yaxis_title="Number of Records",
    plot_bgcolor="#fffdf4",
    paper_bgcolor="#fffdf4"
)


apply_black_plotly_text(
    fig_dist
)


st.plotly_chart(
    fig_dist,
    use_container_width=True
)


st.markdown(
    "### 5-Fold Cross-Validation"
)


cv_col1, cv_col2 = st.columns(2)


with cv_col1:

    st.metric(
        "Mean Macro F1",
        f"{cv_scores.mean() * 100:.2f}%"
    )


with cv_col2:

    st.metric(
        "Standard Deviation",
        f"{cv_scores.std() * 100:.2f}%"
    )


cv_df = pd.DataFrame(
    {
        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],
        "Macro F1": cv_scores * 100
    }
)


fig_cv = px.bar(
    cv_df,
    x="Fold",
    y="Macro F1",
    text="Macro F1"
)


fig_cv.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig_cv.update_layout(
    yaxis_range=[
        0,
        100
    ],
    yaxis_title="Macro F1 (%)",
    plot_bgcolor="#fffdf4",
    paper_bgcolor="#fffdf4"
)


apply_black_plotly_text(
    fig_cv
)


st.plotly_chart(
    fig_cv,
    use_container_width=True
)


st.markdown(
    "### Dataset Information"
)


d1, d2, d3 = st.columns(3)


with d1:

    st.metric(
        "Dataset Records",
        f"{len(y):,}"
    )


with d2:

    st.metric(
        "Input Features",
        len(FEATURE_GROUPS)
    )


with d3:

    st.metric(
        "DASS Items Represented",
        42
    )


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
    unsafe_allow_html=True
)