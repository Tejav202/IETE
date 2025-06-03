import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 2px 5px;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_model():
    """Load or train the churn prediction model"""
    try:
        # Try to load existing model
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        # If model doesn't exist, train a new one
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 200, n_samples),
            'TotalCharges': np.random.uniform(100, 10000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        # Create target variable (churn)
        data['Churn'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Prepare features
        X = pd.get_dummies(df.drop('Churn', axis=1))
        y = df['Churn']
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model and scaler
        joblib.dump(model, 'churn_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        return model, scaler

def get_retention_suggestions(prediction, features):
    """Generate retention suggestions based on prediction and features"""
    suggestions = []
    
    if prediction > 0.5:  # High churn risk
        if features['Contract'] == 'Month-to-month':
            suggestions.append("Consider offering a discount for longer-term contracts")
        if features['MonthlyCharges'] > 100:
            suggestions.append("Review pricing strategy and consider promotional offers")
        if features['TechSupport'] == 'No':
            suggestions.append("Offer free tech support trial period")
        if features['OnlineSecurity'] == 'No':
            suggestions.append("Promote security features and offer free security upgrade")
        if features['tenure'] < 12:
            suggestions.append("Implement early-stage customer engagement program")
    else:  # Low churn risk
        suggestions.append("Maintain current service quality")
        suggestions.append("Consider upselling opportunities")
        suggestions.append("Request customer referrals")
    
    return suggestions

def main():
    st.markdown('<div class="title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.write("Predict customer churn and get retention suggestions!")

    # Load model
    model, scaler = load_model()

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        # Numerical inputs
        tenure = st.slider("Tenure (months)", 1, 72, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 100.0, 10000.0, 1000.0)
        
        # Categorical inputs
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        st.subheader("Additional Services")
        # Service-related inputs
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    # Prepare input data
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Dependents': dependents,
        'PhoneService': phone_service
    }

    # Convert to DataFrame and get dummies
    input_df = pd.DataFrame([input_data])
    input_processed = pd.get_dummies(input_df)

    # Ensure all columns from training are present
    model_columns = joblib.load('churn_model.pkl').feature_names_in_
    for col in model_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[model_columns]

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_processed[numerical_cols] = scaler.transform(input_processed[numerical_cols])

    if st.button("Predict Churn Risk"):
        # Make prediction
        churn_probability = model.predict_proba(input_processed)[0][1]
        
        # Display prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Churn Risk Prediction")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            title={'text': "Churn Risk (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 30], 'color': "green"},
                       {'range': [30, 70], 'color': "yellow"},
                       {'range': [70, 100], 'color': "red"}
                   ]}))
        st.plotly_chart(fig)
        
        # Display risk level
        if churn_probability > 0.7:
            risk_level = "High"
            color = "red"
        elif churn_probability > 0.3:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "Low"
            color = "green"
        
        st.markdown(f"### Risk Level: <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # Get and display retention suggestions
        st.subheader("Retention Suggestions")
        suggestions = get_retention_suggestions(churn_probability, input_data)
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        st.subheader("Important Factors")
        feature_importance = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), 
                    x='Importance', 
                    y='Feature',
                    title="Top 10 Important Features",
                    orientation='h')
        st.plotly_chart(fig)

    # Add information
    st.sidebar.header("About")
    st.sidebar.info("""
    This application predicts customer churn risk using machine learning.
    
    Features:
    - Churn risk prediction
    - Retention suggestions
    - Feature importance analysis
    - Interactive visualizations
    
    The model considers various customer attributes to predict churn probability.
    """)
    
    # Add usage instructions
    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1. Enter customer information
    2. Select service details
    3. Click 'Predict Churn Risk'
    4. View prediction and suggestions
    
    Note: This is a demonstration model using synthetic data.
    """)

if __name__ == "__main__":
    main() 