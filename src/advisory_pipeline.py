import os
import sys
import numpy as np
import joblib
import tensorflow as tf


# ==========================================================
# MAKE SRC DIRECTORY IMPORTABLE
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

from risk_engine import calculate_exposure_risk
from recommendation_engine import generate_recommendations
from personalization_engine import apply_personalization


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm_best.keras"
)

TARGET_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "target_scaler.pkl"
)

TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_test.npy"
)


# ==========================================================
# LOAD LSTM MODEL
# ==========================================================

print("Loading LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("LSTM model loaded successfully.")


# ==========================================================
# LOAD TARGET SCALER
# ==========================================================

print("Loading target scaler...")

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

print("Target scaler loaded successfully.")


# ==========================================================
# PREDICT PM2.5
# ==========================================================

def predict_pm25(sequence):
    """
    Predict the next PM2.5 concentration.

    Input:
        24-hour sequence with 10 features.

    Shape:
        (24, 10)

    Output:
        PM2.5 prediction in µg/m³
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # ------------------------------------------------------
    # CHECK INPUT SHAPE
    # ------------------------------------------------------

    if sequence.shape != (24, 10):

        raise ValueError(
            f"Expected sequence shape (24, 10), "
            f"but received {sequence.shape}"
        )

    # ------------------------------------------------------
    # ADD BATCH DIMENSION
    # ------------------------------------------------------

    # (24, 10)
    #     ↓
    # (1, 24, 10)

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    # ------------------------------------------------------
    # LSTM PREDICTION
    # ------------------------------------------------------

    scaled_prediction = model.predict(
        sequence,
        verbose=0
    )

    # ------------------------------------------------------
    # CONVERT BACK TO ORIGINAL PM2.5 SCALE
    # ------------------------------------------------------

    predicted_pm25 = target_scaler.inverse_transform(
        scaled_prediction
    )[0][0]

    return float(predicted_pm25)


# ==========================================================
# COMPLETE ADVISORY PIPELINE
# ==========================================================

def generate_advisory(
    sequence,
    activity,
    duration_minutes,
    location,
    ventilation,
    age,
    sensitivity
):
    """
    Complete personalized environmental advisory pipeline.

    Pipeline:

    24-hour environmental sequence
                ↓
              LSTM
                ↓
        PM2.5 prediction
                ↓
        Exposure Risk Engine
                ↓
          User Context
                ↓
      Personalization Engine
                ↓
      Personalized Risk
                ↓
     Recommendation Engine
                ↓
      Personalized Advisory
    """

    # ======================================================
    # STEP 1: PREDICT PM2.5
    # ======================================================

    predicted_pm25 = predict_pm25(
        sequence
    )

    # ======================================================
    # STEP 2: CALCULATE BASE EXPOSURE RISK
    # ======================================================

    # Positional arguments are deliberately used
    # to remain compatible with the existing
    # risk_engine.py parameter names.

    risk_result = calculate_exposure_risk(
        predicted_pm25,
        activity,
        duration_minutes,
        location,
        ventilation
    )

    # ======================================================
    # STEP 3: ADD USER EXPOSURE CONTEXT
    # ======================================================

    risk_result["context"] = {
        "activity": activity,
        "duration_minutes": duration_minutes,
        "location": location,
        "ventilation": ventilation
    }

    # ======================================================
    # STEP 4: APPLY USER PERSONALIZATION
    # ======================================================

    risk_result = apply_personalization(
        risk_result=risk_result,
        age=age,
        sensitivity=sensitivity
    )

    # ======================================================
    # STEP 5: GENERATE RECOMMENDATIONS
    # ======================================================

    advisory = generate_recommendations(
        risk_result
    )

    # ======================================================
    # STEP 6: RETURN COMPLETE RESULT
    # ======================================================

    return {
        "predicted_pm25": predicted_pm25,

        "risk_result": risk_result,

        "recommendations": advisory[
            "recommendations"
        ]
    }


# ==========================================================
# TEST THE COMPLETE PIPELINE
# ==========================================================

if __name__ == "__main__":

    print("\n==========================================")
    print(" PERSONALIZED ENVIRONMENTAL ADVISORY")
    print("==========================================")

    # ======================================================
    # LOAD TEST DATA
    # ======================================================

    print("\nLoading test data...")

    X_test = np.load(
        TEST_DATA_PATH
    )

    print(
        f"Test data shape: {X_test.shape}"
    )

    # ------------------------------------------------------
    # SELECT FIRST TEST SEQUENCE
    # ------------------------------------------------------

    test_sequence = X_test[0]

    print(
        f"Selected sequence shape: "
        f"{test_sequence.shape}"
    )

    # ======================================================
    # USER PROFILE
    # ======================================================

    age = 20

    sensitivity = "General"

    # ======================================================
    # EXPOSURE CONTEXT
    # ======================================================

    activity = "exercising"

    duration_minutes = 120

    location = "outdoor"

    ventilation = "moderate"

    # ======================================================
    # DISPLAY INPUT
    # ======================================================

    print("\nUser Profile")
    print("--------------------------------")
    print(f"Age            : {age}")
    print(f"Sensitivity    : {sensitivity}")

    print("\nExposure Context")
    print("--------------------------------")
    print(f"Activity       : {activity}")
    print(f"Duration       : {duration_minutes} minutes")
    print(f"Location       : {location}")
    print(f"Ventilation    : {ventilation}")

    # ======================================================
    # RUN COMPLETE PIPELINE
    # ======================================================

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
    # EXTRACT RESULTS
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
    # DISPLAY PM2.5 PREDICTION
    # ======================================================

    print("\n==========================================")
    print(" PM2.5 PREDICTION")
    print("==========================================")

    print(
        f"Predicted PM2.5 : "
        f"{predicted_pm25:.2f} µg/m³"
    )

    # ======================================================
    # DISPLAY BASE RISK
    # ======================================================

    print("\n==========================================")
    print(" BASE EXPOSURE RISK")
    print("==========================================")

    print(
        f"Base Risk Score : "
        f"{risk_result['base_risk_score']:.2f}"
    )

    print(
        f"Base Risk Level : "
        f"{risk_result['risk_level']}"
    )

    # ======================================================
    # DISPLAY PERSONALIZATION
    # ======================================================

    print("\n==========================================")
    print(" PERSONALIZATION")
    print("==========================================")

    personalization = risk_result[
        "personalization"
    ]

    print(
        f"Age             : "
        f"{personalization['age']}"
    )

    print(
        f"Sensitivity     : "
        f"{personalization['sensitivity']}"
    )

    print(
        f"Age Adjustment  : "
        f"+{personalization['age_adjustment']}"
    )

    print(
        f"Sensitivity Adj.: "
        f"+{personalization['sensitivity_adjustment']}"
    )

    # ======================================================
    # DISPLAY PERSONALIZED RISK
    # ======================================================

    print("\n==========================================")
    print(" PERSONALIZED EXPOSURE RISK")
    print("==========================================")

    print(
        f"Personalized Risk Score : "
        f"{risk_result['personalized_risk_score']:.2f}"
    )

    print(
        f"Personalized Risk Level : "
        f"{risk_result['personalized_risk_level']}"
    )

    # ======================================================
    # DISPLAY RECOMMENDATIONS
    # ======================================================

    print("\n==========================================")
    print(" PERSONALIZED RECOMMENDATIONS")
    print("==========================================")

    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{i}. {recommendation}"
        )

    # ======================================================
    # SUCCESS
    # ======================================================

    print("\n==========================================")
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("==========================================")