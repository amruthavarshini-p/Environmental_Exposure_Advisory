import os
import sys

import numpy as np
import streamlit as st


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# ==========================================================
# IMPORT PIPELINE
# ==========================================================

from advisory_pipeline import generate_advisory


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Environmental Exposure Advisory",
    page_icon="🌍",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    """
    <h1 style="margin-bottom: 0;">
        🌍 Environmental Exposure Advisory
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="font-size:18px; color:#888;">
    AI-Based Personalized Recommendations Based on Environmental Exposure
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================================
# USER PROFILE
# ==========================================================

st.markdown("## 👤 User Profile")

profile_col1, profile_col2, profile_col3 = st.columns(3)

with profile_col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=20
    )

with profile_col2:

    sensitivity = st.selectbox(
        "Environmental Sensitivity",
        [
            "General",
            "Sensitive",
            "Highly Sensitive"
        ]
    )

with profile_col3:

    activity = st.selectbox(
        "Current Activity",
        [
            "sitting",
            "working",
            "cooking",
            "walking",
            "exercising",
            "sleeping"
        ]
    )


# ==========================================================
# EXPOSURE CONTEXT
# ==========================================================

st.markdown("## 🌤️ Exposure Context")

context_col1, context_col2, context_col3 = st.columns(3)

with context_col1:

    duration_minutes = st.slider(
        "Exposure Duration",
        min_value=15,
        max_value=360,
        value=60,
        step=15
    )

with context_col2:

    location = st.selectbox(
        "Environment",
        [
            "indoor",
            "outdoor"
        ]
    )

with context_col3:

    ventilation = st.selectbox(
        "Ventilation",
        [
            "good",
            "moderate",
            "poor"
        ]
    )


# ==========================================================
# USER QUERY
# ==========================================================

st.markdown("## 💬 Ask About Your Exposure")

user_query = st.text_area(
    "What would you like to know?",
    placeholder=(
        "Example: Can I exercise outdoors for one hour?"
    ),
    height=100
)


# ==========================================================
# INFORMATION
# ==========================================================

with st.expander("ℹ️ How does the system work?"):

    st.write(
        """
        The system uses a trained LSTM model to predict
        the next PM2.5 concentration from a 24-hour
        environmental sequence.

        The predicted environmental exposure is then
        combined with your profile and exposure context
        to calculate a personalized risk level.

        The recommendation engine generates actions based
        on the predicted exposure and your activity,
        duration, environment and ventilation.
        """
    )


# ==========================================================
# LOAD TEST SEQUENCE
# ==========================================================

X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_test.npy"
)

if not os.path.exists(X_TEST_PATH):

    st.error(
        "X_test.npy was not found in data/processed."
    )

    st.stop()

X_test = np.load(X_TEST_PATH)

test_sequence = X_test[0]


# ==========================================================
# GENERATE ADVISORY
# ==========================================================

st.markdown("---")

if st.button(
    "🔍 Generate Personalized Advisory",
    type="primary",
    use_container_width=True
):

    # ------------------------------------------------------
    # VALIDATE QUERY
    # ------------------------------------------------------

    if not user_query.strip():

        st.warning(
            "Please enter a question or describe "
            "what you want advice about."
        )

        st.stop()


    # ------------------------------------------------------
    # SHOW PROCESSING
    # ------------------------------------------------------

    with st.spinner(
        "Analyzing your environmental exposure..."
    ):

        result = generate_advisory(
            sequence=test_sequence,
            activity=activity,
            duration_minutes=duration_minutes,
            location=location,
            ventilation=ventilation,
            age=age,
            sensitivity=sensitivity
        )


    # ======================================================
    # RESULTS
    # ======================================================

    predicted_pm25 = result[
        "predicted_pm25"
    ]

    risk_result = result[
        "risk_result"
    ]

    recommendations = result[
        "recommendations"
    ]


    # ======================================================
    # PERSONALIZED RISK VALUES
    # ======================================================

    base_risk_score = risk_result.get(
        "base_risk_score",
        risk_result.get("risk_score", 0)
    )

    base_risk_level = risk_result.get(
        "risk_level",
        "Unknown"
    )

    personalized_risk_score = risk_result.get(
        "personalized_risk_score",
        base_risk_score
    )

    personalized_risk_level = risk_result.get(
        "personalized_risk_level",
        base_risk_level
    )


    # ======================================================
    # PERSONALIZED SUMMARY
    # ======================================================

    st.markdown("## 🎯 Personalized Advisory")

    st.info(
        f"""
        **Your question:**  
        {user_query}

        **Profile:** {age} years old, {sensitivity.lower()} environmental sensitivity.

        **Context:** {activity} for {duration_minutes} minutes,
        {location}, {ventilation} ventilation.
        """
    )


    # ======================================================
    # KEY RESULTS
    # ======================================================

    st.markdown("### 📊 Environmental Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted PM2.5",
            f"{predicted_pm25:.2f} µg/m³"
        )

    with col2:

        st.metric(
            "Personalized Risk Score",
            f"{personalized_risk_score:.0f}/100"
        )

    with col3:

        st.metric(
            "Personalized Risk Level",
            personalized_risk_level
        )


    # ======================================================
    # BASE VS PERSONALIZED RISK
    # ======================================================

    st.markdown("### 🔄 Risk Personalization")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:

        st.metric(
            "Base Environmental Risk",
            f"{base_risk_score:.0f}/100",
            base_risk_level
        )

    with comparison_col2:

        risk_difference = (
            personalized_risk_score
            - base_risk_score
        )

        st.metric(
            "Personalized Risk",
            f"{personalized_risk_score:.0f}/100",
            f"{risk_difference:+.0f} points"
        )


    # ======================================================
    # PERSONALIZATION DETAILS
    # ======================================================

    if "personalization" in risk_result:

        personalization = risk_result[
            "personalization"
        ]

        st.markdown("### 👤 Personalization Factors")

        p1, p2, p3, p4 = st.columns(4)

        with p1:

            st.metric(
                "Age",
                personalization.get(
                    "age",
                    age
                )
            )

        with p2:

            st.metric(
                "Sensitivity",
                personalization.get(
                    "sensitivity",
                    sensitivity
                )
            )

        with p3:

            st.metric(
                "Age Adjustment",
                f"+{personalization.get('age_adjustment', 0)}"
            )

        with p4:

            st.metric(
                "Sensitivity Adjustment",
                f"+{personalization.get('sensitivity_adjustment', 0)}"
            )


    # ======================================================
    # RISK MESSAGE
    # ======================================================

    if personalized_risk_level == "Low":

        st.success(
            "Your current personalized environmental "
            "exposure risk is low."
        )

    elif personalized_risk_level == "Moderate":

        st.warning(
            "Your personalized environmental exposure "
            "risk is moderate. Consider reducing "
            "prolonged exposure."
        )

    elif personalized_risk_level == "High":

        st.warning(
            "Your personalized environmental exposure "
            "risk is high. Reducing exposure is recommended."
        )

    else:

        st.error(
            "Your personalized environmental exposure "
            "risk is very high. Avoid unnecessary exposure "
            "where possible."
        )


    # ======================================================
    # USER CONTEXT
    # ======================================================

    st.markdown("### 👤 Exposure Context")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.write("**Age**")
        st.write(age)

    with c2:

        st.write("**Activity**")
        st.write(activity.title())

    with c3:

        st.write("**Environment**")
        st.write(location.title())

    with c4:

        st.write("**Duration**")
        st.write(f"{duration_minutes} min")


    # ======================================================
    # RISK BREAKDOWN
    # ======================================================

    st.markdown("### ⚠️ Risk Factor Breakdown")

    if "factor_breakdown" in risk_result:

        breakdown = risk_result[
            "factor_breakdown"
        ]

        cols = st.columns(
            len(breakdown)
        )

        for col, (factor, value) in zip(
            cols,
            breakdown.items()
        ):

            with col:

                st.metric(
                    factor.replace(
                        "_",
                        " "
                    ).title(),
                    value
                )


    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    st.markdown(
        "### 💡 Personalized Recommendations"
    )

    if recommendations:

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div style="
                    padding:15px;
                    margin:10px 0;
                    border-radius:10px;
                    background-color:#262626;
                    border-left:5px solid #4CAF50;
                ">
                    {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.success(
            "No additional recommendations are required."
        )


    # ======================================================
    # QUERY-SPECIFIC RESPONSE
    # ======================================================

    st.markdown("### 💬 Response to Your Question")

    query_lower = user_query.lower()

    if (
        "exercise" in query_lower
        or "workout" in query_lower
        or "run" in query_lower
    ):

        if personalized_risk_level in [
            "High",
            "Very High"
        ]:

            st.error(
                "Based on the predicted exposure and "
                "your personalized risk level, strenuous "
                "outdoor exercise is not recommended."
            )

        elif personalized_risk_level == "Moderate":

            st.warning(
                "Exercise may be possible, but consider "
                "reducing intensity or moving indoors."
            )

        else:

            st.success(
                "The personalized predicted exposure risk "
                "is low. Normal exercise precautions can "
                "be followed."
            )

    elif (
        "cook" in query_lower
        or "cooking" in query_lower
    ):

        if ventilation == "poor":

            st.warning(
                "Since you are cooking indoors with poor "
                "ventilation, improve ventilation using "
                "an exhaust fan or open windows."
            )

        else:

            st.success(
                "Maintain good ventilation while cooking "
                "to reduce indoor exposure."
            )

    elif (
        "safe" in query_lower
        or "outside" in query_lower
        or "outdoor" in query_lower
    ):

        if personalized_risk_level in [
            "High",
            "Very High"
        ]:

            st.error(
                "Outdoor exposure should be reduced "
                "under the current predicted conditions."
            )

        else:

            st.info(
                "The current personalized exposure risk "
                "does not indicate a high environmental risk."
            )

    else:

        st.info(
            "The advisory above is based on your "
            "environmental prediction, exposure context, "
            "user profile and selected activity."
        )


    # ======================================================
    # SHAP EXPLAINABILITY
    # ======================================================

    st.markdown(
        "### 🧠 AI Model Explainability"
    )

    shap_image = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "shap",
        "shap_feature_importance.png"
    )

    if os.path.exists(shap_image):

        st.image(
            shap_image,
            caption=(
                "LSTM feature importance using "
                "mean absolute SHAP values"
            ),
            use_container_width=True
        )

        st.caption(
            "Higher mean absolute SHAP values indicate "
            "greater contribution to the model prediction."
        )

    else:

        st.warning(
            "SHAP visualization is not available."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "AI-Based Advisory Framework for Personalized "
    "Recommendations Based on Environmental Exposure"
)
