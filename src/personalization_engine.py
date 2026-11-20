# src/personalization_engine.py


def apply_personalization(
    risk_result,
    age,
    sensitivity
):
    """
    Adjust the exposure risk based on the user's
    profile.

    IMPORTANT:
    These are project-defined personalization rules,
    not medical guidelines.
    """

    base_score = risk_result["risk_score"]

    # ------------------------------------------------------
    # SENSITIVITY ADJUSTMENT
    # ------------------------------------------------------

    sensitivity_adjustment = {
        "General": 0,
        "Sensitive": 8,
        "Highly Sensitive": 15
    }

    sensitivity_score = sensitivity_adjustment.get(
        sensitivity,
        0
    )

    # ------------------------------------------------------
    # AGE ADJUSTMENT
    # ------------------------------------------------------

    # Keep age adjustment intentionally small.
    # This is a project-defined personalization rule.

    if age < 18:
        age_score = 5

    elif age >= 65:
        age_score = 5

    else:
        age_score = 0

    # ------------------------------------------------------
    # PERSONALIZED SCORE
    # ------------------------------------------------------

    personalized_score = (
        base_score
        + sensitivity_score
        + age_score
    )

    # Cap score at 100

    personalized_score = min(
        personalized_score,
        100
    )

    # ------------------------------------------------------
    # PERSONALIZED RISK LEVEL
    # ------------------------------------------------------

    if personalized_score <= 25:

        personalized_level = "Low"

    elif personalized_score <= 50:

        personalized_level = "Moderate"

    elif personalized_score <= 75:

        personalized_level = "High"

    else:

        personalized_level = "Very High"

    # ------------------------------------------------------
    # ADD PERSONALIZATION INFORMATION
    # ------------------------------------------------------

    risk_result["base_risk_score"] = base_score

    risk_result["personalized_risk_score"] = (
        personalized_score
    )

    risk_result["personalized_risk_level"] = (
        personalized_level
    )

    risk_result["personalization"] = {
        "age": age,
        "sensitivity": sensitivity,
        "age_adjustment": age_score,
        "sensitivity_adjustment": sensitivity_score
    }

    return risk_result