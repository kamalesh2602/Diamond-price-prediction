
import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("../models/ridge_model.pkl")

st.title("💎 Diamond Price Prediction System")

page = st.sidebar.selectbox("Select Page",
                           ["Prediction", "EDA", "Results", "Workflow"])

# ---------------- Prediction ----------------
if page == "Prediction":
    st.header("Predict Price")

    carat = st.slider("Carat", min_value=0.1, max_value=5.5, value=1.0, step=0.01)
    cut = st.slider("Cut (0-4: Fair to Ideal)", min_value=0, max_value=4, value=2)
    color = st.slider("Color (0-6: J to D)", min_value=0, max_value=6, value=3)
    clarity = st.slider("Clarity (0-6: I1 to IF)", min_value=0, max_value=6, value=3)
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