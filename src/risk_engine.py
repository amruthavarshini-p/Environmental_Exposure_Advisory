# ============================================================
# EXPOSURE RISK ENGINE
# ============================================================

"""
Converts environmental exposure conditions and user context
into an Exposure Risk Score and Risk Level.

This module does NOT train a machine-learning model.

It is the decision layer between:
    LSTM prediction
            ↓
    Exposure Risk
            ↓
    Recommendation Engine
"""


# ============================================================
# PM2.5 RISK LEVELS
# ============================================================

def get_pm25_risk(pm25):
    """
    Assign a base environmental risk score based on PM2.5.

    Returns:
        score: 0-100
        level: Low / Moderate / High / Very High
    """

    if pm25 <= 30:
        return 10, "Low"

    elif pm25 <= 60:
        return 35, "Moderate"

    elif pm25 <= 90:
        return 60, "High"

    else:
        return 85, "Very High"


# ============================================================
# ACTIVITY FACTOR
# ============================================================

def get_activity_factor(activity):
    """
    Higher-intensity activities generally increase exposure
    because breathing rate can increase.
    """

    activity = activity.lower()

    factors = {
        "sleeping": 0,
        "sitting": 0,
        "working": 5,
        "cooking": 10,
        "walking": 10,
        "exercising": 20
    }

    return factors.get(
        activity,
        5
    )


# ============================================================
# EXPOSURE DURATION FACTOR
# ============================================================

def get_duration_factor(duration_minutes):
    """
    Longer exposure duration increases cumulative exposure.
    """

    if duration_minutes <= 30:
        return 0

    elif duration_minutes <= 60:
        return 5

    elif duration_minutes <= 120:
        return 10

    else:
        return 15


# ============================================================
# LOCATION FACTOR
# ============================================================

def get_location_factor(location_type):

    location_type = location_type.lower()

    if location_type == "outdoor":
        return 5

    return 0


# ============================================================
# VENTILATION FACTOR
# ============================================================

def get_ventilation_factor(ventilation):

    ventilation = ventilation.lower()

    factors = {
        "good": 0,
        "moderate": 5,
        "poor": 10
    }

    return factors.get(
        ventilation,
        5
    )


# ============================================================
# CALCULATE EXPOSURE RISK
# ============================================================

def calculate_exposure_risk(
    predicted_pm25,
    activity,
    duration_minutes,
    location_type,
    ventilation
):

    # --------------------------------------------------------
    # Base environmental risk
    # --------------------------------------------------------

    pm25_score, pm25_level = get_pm25_risk(
        predicted_pm25
    )

    # --------------------------------------------------------
    # Context factors
    # --------------------------------------------------------

    activity_factor = get_activity_factor(
        activity
    )

    duration_factor = get_duration_factor(
        duration_minutes
    )

    location_factor = get_location_factor(
        location_type
    )

    ventilation_factor = get_ventilation_factor(
        ventilation
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    risk_score = (
        pm25_score
        + activity_factor
        + duration_factor
        + location_factor
        + ventilation_factor
    )

    # Limit to 100
    risk_score = min(
        risk_score,
        100
    )

    # --------------------------------------------------------
    # Final risk category
    # --------------------------------------------------------

    if risk_score <= 25:

        risk_level = "Low"

    elif risk_score <= 50:

        risk_level = "Moderate"

    elif risk_score <= 75:

        risk_level = "High"

    else:

        risk_level = "Very High"

    return {
        "risk_score": round(
            risk_score,
            2
        ),

        "risk_level": risk_level,

        "pm25": round(
            predicted_pm25,
            2
        ),

        "pm25_risk_level": pm25_level,

        "activity_factor": activity_factor,

        "duration_factor": duration_factor,

        "location_factor": location_factor,

        "ventilation_factor": ventilation_factor
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EXPOSURE RISK ENGINE TEST")
    print("=" * 60)

    result = calculate_exposure_risk(
        predicted_pm25=80,
        activity="exercising",
        duration_minutes=120,
        location_type="outdoor",
        ventilation="moderate"
    )

    print("\nPredicted PM2.5:")
    print(
        f"{result['pm25']} µg/m³"
    )

    print("\nPM2.5 risk:")
    print(
        result["pm25_risk_level"]
    )

    print("\nExposure risk score:")
    print(
        result["risk_score"]
    )

    print("\nFinal exposure risk:")
    print(
        result["risk_level"]
    )

    print("\nRisk factors:")

    print(
        "Activity factor:",
        result["activity_factor"]
    )

    print(
        "Duration factor:",
        result["duration_factor"]
    )

    print(
        "Location factor:",
        result["location_factor"]
    )

    print(
        "Ventilation factor:",
        result["ventilation_factor"]
    )

    print("\n")
    print("=" * 60)
    print("RISK ENGINE TEST COMPLETED")
    print("=" * 60)