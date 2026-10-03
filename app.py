import json
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Health Risk Predictor", layout="wide")
st.title("🩺 Diabetes Risk Prediction Dashboard")
st.markdown("A machine learning dashboard that predicts diabetes risk from health parameters, trained on a 100,000-record clinical dataset.")

model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')
le_gender = joblib.load('le_gender.pkl')
le_smoking = joblib.load('le_smoking.pkl')
df = pd.read_csv('data/diabetes_prediction_dataset.csv')

tab1, tab2 = st.tabs(["Predict Your Risk", "Dataset Insights"])

with tab1:
    st.subheader("Enter your health parameters")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 1, 100, 30)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease History", [0, 1])

    with col2:
        smoking_history = st.selectbox("Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
        hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
        glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

    if st.button("Predict Risk"):
        gender_enc = le_gender.transform([gender])[0]
        smoking_enc = le_smoking.transform([smoking_history])[0]

        input_data = pd.DataFrame([[gender_enc, age, hypertension, heart_disease,
                                      smoking_enc, bmi, hba1c, glucose]],
                                    columns=columns)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Higher Diabetes Risk Detected ({probability*100:.1f}% probability)")
        else:
            st.success(f"✅ Lower Diabetes Risk ({probability*100:.1f}% probability)")

    st.caption("Portfolio project only. Not a medical diagnostic tool.")

with tab2:
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x='age', color='diabetes',
                             title='Age Distribution by Diabetes Status', barmode='overlay')
        st.plotly_chart(fig1, width='stretch')

    with col2:
        fig2 = px.box(df, x='diabetes', y='bmi', title='BMI Distribution by Diabetes Status')
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                   title='What Drives the Prediction Most')
    st.plotly_chart(fig3, width='stretch')

    st.subheader("Model Performance")
    m = json.load(open('metrics.json'))
    st.markdown(f"""
    - **Accuracy:** {m['accuracy']*100:.2f}%
    - **Precision (diabetic class):** {m['precision']:.2f}
    - **Recall (diabetic class):** {m['recall']:.2f}
    - **F1 (diabetic class):** {m['f1']:.2f}
    - **Model:** Random Forest Classifier (100 trees, balanced class weights)
    """)
