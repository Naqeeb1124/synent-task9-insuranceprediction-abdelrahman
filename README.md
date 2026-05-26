# Synent Technologies — Data Science Internship Program
## Task 9: End-to-End Data Science Project (Medical Insurance Premium Prediction)

**Live Demo:** [HealthSure Insurance Predictor](https://naqeeb1124-synent-task9-insuranceprediction-abdelrah-app-xiynip.streamlit.app/)

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/deployment-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/visualization-Plotly-3F4F75.svg)](https://plotly.com/)

An interactive, production-grade **Medical Insurance Premium Prediction** application built using **Python, Scikit-Learn, Plotly, and Streamlit**. This application utilizes predictive analytics models trained on historical insurance patient databases to estimate personal annual medical insurance premiums, calculate customized lifestyle-risk cost drivers, and present interactive exploratory data insights.

---

## 📌 Problem Statement
Medical insurance costs fluctuate widely based on individual health parameters, demographics, and lifestyle factors. For insurance providers, accurately calculating risk premiums is critical for maintaining actuarial solvency. For individuals, understanding how personal health conditions (like high Body Mass Index or tobacco use) affect insurance premiums can inspire positive lifestyle adjustments.

The goal of this project is to build an **end-to-end Machine Learning system** that:
1. Cleans and preprocesses client databases.
2. Conducts multi-dimensional **Exploratory Data Analysis (EDA)**.
3. Benchmarks multiple regressor models (Linear Regression, Random Forest, and Gradient Boosting) to select the most accurate estimator.
4. Deploys a premium **Streamlit web application** featuring a personalized premium calculator, live cost-driver waterfalls, interactive demographic clustering charts, and model diagnostic specifications.

---

## 📊 Dataset Details
The model is trained on the standard **Medical Cost Personal Datasets** sourced from Kaggle/GitHub. It consists of **1,338 individual patient profiles** with the following features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **age** | Numerical | Age of the primary beneficiary (18 to 64 years) |
| **sex** | Categorical | Insurance contractor gender (`female`, `male`) |
| **bmi** | Numerical | Body Mass Index ($kg/m^2$), ideal index is 18.5 to 24.9 |
| **children** | Numerical | Number of dependent children/dependents covered |
| **smoker** | Categorical | Regular smoking habits (`yes`, `no`) |
| **region** | Categorical | The beneficiary's residential area in the US |
| **charges** | Numerical | **[TARGET]** Individual medical costs billed by health insurance annually |

---

## 🛠️ Project Architecture & Workflow
This repository follows standard software engineering best practices with modular, decoupled Python scripts:

```text
synent-task9-insuranceprediction-abdelrahman/
├── data/
│   ├── raw/                  # Raw insurance.csv dataset
│   └── processed/            # Cleaned, duplicate-free dataset
├── models/
│   └── best_model.pkl        # Fitted preprocessor pipeline, model, & feature importances
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Preprocessing pipeline (imputation, scaling, encoding)
│   ├── train.py              # Model benchmarking, training, evaluation, saving
│   └── predict.py            # Inference API & marginal impact cost driver calculation
├── app.py                    # Multi-page interactive Streamlit dashboard
├── requirements.txt          # Python library dependencies
└── README.md                 # Technical project documentation
```

### Decoupled Components
* **`src/data_preprocessing.py`**: Conducts data cleaning (removing duplicate records, casting column formats, stripping string whitespaces) and compiles a Scikit-Learn `ColumnTransformer` preprocessing pipeline.
* **`src/train.py`**: Splits records (80% train / 20% test), applies the preprocessor, trains Linear Regression, Random Forest, and Gradient Boosting models, evaluates accuracy performance, and serializes the champion model package.
* **`src/predict.py`**: Loads the serialized package, handles new profile inferences, and runs **Ceteris Paribus (marginal impact analysis)** to break down surcharge parameters (e.g., exact premium increase added solely due to smoking or high BMI).
* **`app.py`**: The Streamlit user interface crafted with tailored Google Fonts, custom CSS card stylings, interactive BMI calculators, and animated Plotly graphs.

---

## 📈 Model Performance & Comparison
Three candidate models were evaluated during compiling. The **Random Forest Regressor** achieved outstanding accuracy, outperforming the baseline Linear Regression model:

| Model Name | $R^2$ Score (Accuracy) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **89.87%** 🏆 | **$2,426.28** | **$4,315.15** |
| **Gradient Boosting Regressor** | 89.84% | $2,536.32 | $4,321.25 |
| **Linear Regression** | 80.69% | $4,177.05 | $5,956.34 |

### Key Actuarial Insights
* **Smoking is the #1 Cost Driver**: According to the Random Forest model feature weights, **regular tobacco use accounts for nearly 49% of the premium determination**. Actuarial calculations reveal that smoking regularly surcharges the premium by an average of **$15,000+** per year!
* **BMI & Obesity Threshold**: Normal weight BMI (<25) incurs zero BMI surcharge. However, as BMI climbs over the normal threshold (especially past 30), a dynamic surcharge is applied. Obesity compounded by smoking places a patient in the absolute highest risk tier (charges exceeding $35k–$50k).
* **Age Progression**: Aging adds a steady linear risk adjustment factor of approximately **$250–$300** to the premium for every single year of age increase.

---

## 🖥️ Dashboard Interface Showcase
The web dashboard features a premium dark theme and is split into three interactive centers:
1. **🎯 Premium Calculator**:
   * Interactive input forms for demographics, including a dynamic **Height/Weight BMI Calculator**.
   * Instant estimated charges computed dynamically by the machine learning engine.
   * **Actuarial Cost Driver Breakdown**: Interactive horizontal Plotly chart showing exact surcharge contributions (Base Coverage, Age surcharge, BMI surcharge, Smoking surcharge).
2. **📊 Dynamic Insights (EDA)**:
   * Key demographic metric totals.
   * Interactive histograms for charge distribution layers.
   * Box plots comparing premium scales by smoking profiles.
   * **Risk Demographics Cluster Map**: Interactive scatter plots showing distinct risk "stripes" by age, BMI, and smoker status.
3. **🧠 Model Arena & Analytics**:
   * Accuracy ($R^2$ Score) and Error metrics (MAE & RMSE) benchmarks.
   * Relative feature weights demonstrating what drives risk decisions in artificial neural/decision tree configurations.

---

## 🚀 How to Run the Project Locally

### 1. Clone & Set Up Workspace
Navigate to your repository directory and ensure Python 3.10+ is installed:
```bash
# Navigate to workspace
cd synent-task9-insuranceprediction-abdelrahman
```

### 2. Install Required Dependencies
Install the required packages in one command:
```bash
pip install -r requirements.txt
```

### 3. Run Data Preprocessing & Model Training
Execute the training script to process the raw dataset, benchmark models, and generate the model artifact:
```bash
# 1. Clean the dataset
python src/data_preprocessing.py

# 2. Train the ML models and save the best model payload
python src/train.py
```

### 4. Deploy the Streamlit Application
Start the interactive dashboard locally:
```bash
streamlit run app.py
```
A browser tab will automatically launch at `http://localhost:8501/` displaying the active engine!

---

## 🎓 Evaluation & Submission Guidelines Checklist
- [x] Repository correctly named as `synent-task9-insuranceprediction-abdelrahman`
- [x] Preprocessing pipeline implemented (`src/data_preprocessing.py`)
- [x] High-accuracy machine learning model trained and comparative benchmarking logged (`src/train.py`)
- [x] Fully functional deployment ready with Streamlit UI dashboard (`app.py`)
- [x] Thorough markdown documentation in `README.md`
