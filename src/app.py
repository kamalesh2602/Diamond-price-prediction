
import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("../models/ridge_model.pkl")

st.title("💎 Diamond Price Prediction System")

page = st.sidebar.selectbox("Select Page",
                           ["Home","Prediction", "EDA", "Results", "Workflow"])

# ---------------- Home ----------------
if page == "Home":
    st.title("💎 Diamond Price Prediction System")

    st.markdown("""
    ### About the Project

    This machine learning application predicts the price of a diamond
    based on its physical and quality characteristics.

    The model has been trained on the popular Diamonds Dataset and uses
    Ridge Regression to estimate market prices accurately.

    ### Features Used

    - Carat
    - Cut
    - Color
    - Clarity
    - Depth
    - Table
    - Length (x)
    - Width (y)
    - Height (z)
    - Volume (Engineered Feature)

    ### Tech Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Streamlit

    ### Workflow

    Data Collection → Preprocessing → Feature Engineering →
    Model Training → Evaluation → Deployment

    Use the sidebar to navigate through the application.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Ridge")

    with col2:
        st.metric("Features", "10")

    with col3:
        st.metric("Deployment", "Streamlit")

    st.info("Navigate to the Prediction page to estimate a diamond's price.")
    
# ---------------- Prediction ----------------
elif page == "Prediction":
    st.header("Predict Price")

    carat = st.slider("Carat", min_value=0.1, max_value=5.5, value=1.0, step=0.01)
    cut_options = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4
}

    cut = st.selectbox("Cut", list(cut_options.keys()))
    cut = cut_options[cut]

    color_options = {
    "J": 0,
    "I": 1,
    "H": 2,
    "G": 3,
    "F": 4,
    "E": 5,
    "D": 6
}

    color = st.selectbox(
        "Color",
        list(color_options.keys()),
        index=3
    )

    color = color_options[color]
    clarity_options = {
        "I1": 0,
        "SI2": 1,
        "SI1": 2,
        "VS2": 3,
        "VS1": 4,
        "VVS2": 5,
        "VVS1": 6,
        "IF": 7
    }

    clarity = st.selectbox(
        "Clarity",
        list(clarity_options.keys()),
        index=3
    )

    clarity = clarity_options[clarity]
    depth = st.slider("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.slider("Table (%)", min_value=40.0, max_value=95.0, value=57.0, step=0.1)
    x = st.slider("Length (mm)", min_value=0.0, max_value=12.0, value=6.0, step=0.01)
    y = st.slider("Width (mm)", min_value=0.0, max_value=12.0, value=6.0, step=0.01)
    z = st.slider("Height (mm)", min_value=0.0, max_value=10.0, value=3.5, step=0.01)

    volume = x * y * z

    if st.button("Predict"):
        features = np.array([[carat, cut, color, clarity,
                              depth, table, x, y, z, volume]])
        prediction = model.predict(features)
        st.success(f"Predicted Price: ${prediction.item():.2f}")

# ---------------- EDA ----------------
elif page == "EDA":
    st.header("EDA Visualizations")
    st.image("../outputs/carat_price.png")
    st.image("../outputs/heatmap.png")

# ---------------- Results ----------------
elif page == "Results":
    st.header("Model Comparison")
    
    with open("../outputs/results.txt") as f:
        lines = f.readlines()
        
    if len(lines) > 1:
        columns = lines[0].split()
        data = []
        for line in lines[1:]:
            parts = line.split()
            model_name = " ".join(parts[:-3])
            metrics = [float(p) for p in parts[-3:]]
            data.append([model_name] + metrics)
            
        df = pd.DataFrame(data, columns=columns)
        st.table(df)
    else:
        st.text("No results available.")

# ---------------- Workflow ----------------
elif page == "Workflow":
    st.header("Project Workflow")
    st.write("""
    1. Data Preprocessing  
    2. Feature Engineering  
    3. Model Training  
    4. Model Evaluation  
    5. Deployment using Streamlit  
    """)