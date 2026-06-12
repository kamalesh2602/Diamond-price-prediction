# Diamond Price Prediction and Analysis

A machine learning web application that predicts diamond prices based on their physical and quality characteristics. The project uses Ridge Regression and is deployed using Streamlit.

## 🚀 Live Demo

https://diamond-price-prediction-tcbh8p7t8xax3jn4ugquhe.streamlit.app/

## Features

* Predict diamond prices using trained ML models
* Interactive user interface with Streamlit
* Exploratory Data Analysis (EDA) visualizations
* Model performance comparison
* Feature engineering using diamond volume

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

## Project Workflow

1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Streamlit Deployment

## Dataset

The project uses the Diamonds Dataset containing attributes such as:

* Carat
* Cut
* Color
* Clarity
* Depth
* Table
* Dimensions (x, y, z)

## Model Performance

* Model: Ridge Regression
* R² Score: 0.886
* RMSE: 1342

## How To Run

```
git clone https://github.com/kamalesh2602/Diamond-price-prediction
cd Diamond-price-prediction
```

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run modules in order
```bash
cd src
python preprocessing.py
python eda.py
python model_building.py
python evaluation.py
```

### 3. Run final app
```bash
streamlit run app.py
```
