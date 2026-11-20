# src/recommendation_engine.py

def generate_recommendations(risk_result):
    """
    Generate personalized environmental exposure recommendations
    based on the output of the Exposure Risk Engine.
    """

    risk_level = risk_result["risk_level"]
    pm25 = risk_result["pm25"]
    activity = risk_result["context"]["activity"]
    location = risk_result["context"]["location"]
    duration = risk_result["context"]["duration_minutes"]
    ventilation = risk_result["context"]["ventilation"]

    recommendations = []

    # Risk-based recommendations
    if risk_level == "Low":
        recommendations.append(
            "Environmental exposure risk is low. Normal outdoor activities can be continued."
        )

    elif risk_level == "Moderate":
        recommendations.append(
            "Environmental exposure is moderate. Consider reducing prolonged outdoor exposure."
        )

    elif risk_level == "High":
        recommendations.append(
            "Environmental exposure is high. Reduce prolonged or strenuous outdoor activities."
        )

    elif risk_level == "Very High":
        recommendations.append(
            "Environmental exposure is very high. Avoid unnecessary outdoor exposure and consider moving activities indoors."
        )

    # Activity-specific recommendations
    if activity == "exercising":
        if risk_level in ["High", "Very High"]:
            recommendations.append(
                "Avoid strenuous outdoor exercise during this exposure period."
            )
        elif risk_level == "Moderate":
            recommendations.append(
                "Consider reducing exercise intensity or moving exercise indoors."
            )

    elif activity == "cooking":
        if ventilation == "poor":
            recommendations.append(
                "Improve ventilation while cooking by opening windows or using an exhaust fan."
            )

    # Location-specific recommendations
    if location == "outdoor" and risk_level in ["High", "Very High"]:
        recommendations.append(
            "Consider moving the activity indoors to reduce environmental exposure."
        )

    # Duration-specific recommendations
    if duration > 120 and risk_level in ["Moderate", "High", "Very High"]:
        recommendations.append(
            "Your exposure duration is relatively long. Consider taking breaks and reducing total exposure time."
        )

    # PM2.5-specific recommendation
    if pm25 > 60:
        recommendations.append(
            "Elevated PM2.5 is contributing significantly to the exposure risk."
        )

    return {
        "risk_level": risk_level,
        "pm25": pm25,
        "recommendations": recommendations
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    sample_risk_result = {
        "risk_score": 100,
        "risk_level": "Very High",
        "pm25": 80,
        "pm25_risk_level": "High",

        "context": {
            "activity": "exercising",
            "duration_minutes": 120,
            "location": "outdoor",
            "ventilation": "moderate"
        },

        "factor_breakdown": {
            "pm25_base_score": 60,
            "activity_score": 20,
            "duration_score": 10,
            "location_score": 5,
            "ventilation_score": 5
        }
    }

    result = generate_recommendations(sample_risk_result)

    print("\n===== PERSONALIZED ADVISORY =====")
    print(f"PM2.5: {result['pm25']} µg/m³")
    print(f"Risk Level: {result['risk_level']}")

    print("\nRecommendations:")

    for i, recommendation in enumerate(
        result["recommendations"], start=1
    ):
        print(f"{i}. {recommendation}")