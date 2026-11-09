import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

best_xgb = joblib.load("models/best_xgb_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def classify_rul(rul):
    if rul > 15000:
        return "🟢 Healthy"
    elif rul > 8000:
        return "🔵 Good"
    elif rul > 3000:
        return "🟡 Warning"
    elif rul > 500:
        return "🟠 Critical"
    else:
        return "🔴 Failure Imminent"

def get_recommendation(feature, value):

    if feature == "Temperature":

        if value <= 56.56:
            return "Normal temperature. Continue routine monitoring."

        elif value <= 70:
            return "Temperature is high. Inspect cooling system during next maintenance."

        else:
            return "Critical temperature. Immediate cooling system inspection required."

    elif feature == "Speed":

        if 1300 <= value <= 1441:
            return "Motor speed is within normal operating range."

        else:
            return "Motor speed is outside normal range. Inspect bearings and mechanical load."

    elif feature == "Frequency":

        if 47.80 <= value <= 49.34:
            return "Frequency is stable."

        else:
            return "Frequency deviation detected. Check inverter and supply."

    elif feature == "Output_Voltage":

        if 210.86 <= value <= 218.68:
            return "Output voltage is normal."

        else:
            return "Output voltage deviation detected. Inspect inverter."

    elif feature == "DC_Bus_Voltage":

        if 312.32 <= value <= 317.30:
            return "DC Bus voltage is stable."

        else:
            return "DC Bus voltage abnormal. Inspect DC link."

    elif feature == "Output_Current":

        if 5.77 <= value <= 8.12:
            return "Output current is normal."

        else:
            return "Output current abnormal. Inspect motor loading."

    elif feature == "High_Resolution_Output_Current":

        if 57.60 <= value <= 81.60:
            return "Current waveform is normal."

        else:
            return "Current deviation detected. Inspect winding and load."

    elif feature == "Power":

        if 1244.87 <= value <= 1734.86:
            return "Power consumption is normal."

        else:
            return "Abnormal power consumption. Inspect mechanical load."

    elif feature == "Thermal_Load":

        if value <= 430.12:
            return "Thermal load is normal."

        elif value <= 600:
            return "Thermal load is high. Reduce operating load."

        else:
            return "Critical thermal load. Immediate maintenance recommended."

    elif feature == "Load_Index":

        if value <= 0.00607:
            return "Load index is normal."

        else:
            return "High load index detected. Inspect mechanical loading."

    else:
        return "Continue routine monitoring."

def predict_motor_health(input_data):
    """
    Predict motor RUL and generate the maintenance report.
    """

    predicted_rul = best_xgb.predict(input_data)[0]
    status = classify_rul(predicted_rul)
    motor_shap = explainer(input_data)
    shap_scores = motor_shap.values[0]
    feature_names = input_data.columns
    top_indices = np.argsort(np.abs(shap_scores))[::-1][:3]
    top_features = feature_names[top_indices]
    top_shap_scores = shap_scores[top_indices]
    top_feature_values = input_data.iloc[0][top_features]

    feature_analysis = []
    for feature, score, value in zip(top_features, top_shap_scores, top_feature_values):

        # SHAP Interpretation
        if score < 0:
            impact = "Negative"
            ai_interpretation = "This feature contributed negatively to the predicted RUL."
        else:
            impact = "Positive"
            ai_interpretation = "This feature contributed positively to the predicted RUL."

        # Engineering Recommendation
        recommendation = get_recommendation(feature, value)

        feature_analysis.append({

            "Feature": feature,
            "Actual_Value": round(value, 2),
            "SHAP_Value": f"{float(score):.2f}",
            "AI_Impact": impact,
            "AI_Interpretation": ai_interpretation,
            "Engineering_Recommendation": recommendation

        })

    return {

        "Predicted_RUL": f"{float(predicted_rul):.2f}",
        "Status": status,
        "Feature_Analysis": feature_analysis,
        "SHAP_Explanation": motor_shap

    }

st.title("AI Predictive Maintenance System")
st.write("Induction Motor Remaining Life Prediction")
st.header("Motor Operating Parameters")

dc_bus_voltage = st.number_input("DC Bus Voltage", value=315.08)
frequency = st.number_input("Frequency", value=48.58)
high_resolution_output_current = st.number_input("High Resolution Output Current", value=70.17)
output_current = st.number_input("Output Current", value=7.01)
output_voltage = st.number_input("Output Voltage", value=214.60)
speed = st.number_input("Speed", value=1369.34)
temperature = st.number_input("Temperature", value=46.78)
load_index = st.number_input("Load Index", value=0.00517)
power = st.number_input("Power", value=1500.24)
thermal_load = st.number_input("Thermal Load", value=338.60)

input_data = pd.DataFrame([[
    dc_bus_voltage,
    frequency,
    high_resolution_output_current,
    output_current,
    output_voltage,
    speed,
    temperature,
    load_index,
    power,
    thermal_load
]], columns=feature_names)

if st.button("Predict Motor Health"):

    report = predict_motor_health(input_data)

    st.subheader("Motor Health Report")

    st.write(
        f"Predicted RUL: {report['Predicted_RUL']} hours"
    )

    st.write(
        f"Status: {report['Status']}"
    )
    st.subheader("Top 3 Feature Analysis")

    for item in report["Feature_Analysis"]:

       st.markdown(f"### {item['Feature']}")

       st.write(f"**Actual Value:** {item['Actual_Value']}")
       st.write(f"**SHAP Value:** {item['SHAP_Value']}")
       st.write(f"**AI Impact:** {item['AI_Impact']}")

       st.write(
        f"**AI Interpretation:** {item['AI_Interpretation']}"
       )

       st.write(
        f"**Maintenance Recommendation:** "
        f"{item['Engineering_Recommendation']}"
       )

       st.divider()

    st.subheader("SHAP Explanation")
    shap.plots.waterfall(
       report["SHAP_Explanation"][0],
       show=False
       )
    st.pyplot(plt.gcf())
    plt.close()


#cd "C:\Users\geeta\Downloads\Ming\Project\AI_PM" 
# .\venv\Scripts\Activate.ps1
#python -m pip show xgboost
#python -m streamlit run apps\app.py