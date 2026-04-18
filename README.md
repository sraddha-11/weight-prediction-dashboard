 #Weight Prediction Using Lifestyle Data

A machine learning project that predicts weight variation based on daily lifestyle habits. This repository contains data analysis, model training, and an interactive web dashboard.

**🔗 Live Dashboard:** [View Live](https://weight-prediction-dashboard.vercel.app)

---

## 📋 Project Overview

This project analyzes the relationship between lifestyle factors and weight variation using data science techniques. By training machine learning models on real lifestyle data, we can predict how daily habits affect weight change.

### Key Features
- 📊 **Exploratory Data Analysis (EDA)** of lifestyle factors
- 🤖 **Machine Learning Models**: Linear Regression & Random Forest
- 📈 **High Accuracy**: 91.3% accuracy with Random Forest (R² = 0.9912)
- 🎨 **Interactive Dashboard**: Web-based tool for predictions
- 📱 **Responsive Design**: Works on desktop and mobile

---

## 🎯 Results

### Model Performance

| Metric | Linear Regression | Random Forest |
|--------|---|---|
| **R² Score** | 0.9861 | **0.9912** |
| **MAE** | 0.0424 kg | **0.0340 kg** |
| **RMSE** | 0.0580 kg | **0.0462 kg** |
| **Accuracy** | 89.1% | **91.3%** |

### Feature Importance (Random Forest)

The most influential factors in predicting weight change:

1. **Steps/Physical Activity** - 41.8%
2. **Calories** - 26.3%
3. **Fasting Hours** - 24.6%
4. **Screen Time** - 4.2%
5. **Sleep Duration** - 3.0%

### Feature Correlations

- **Calories** (+0.991): Higher calorie intake strongly correlates with weight gain
- **Steps** (-0.975): More activity strongly correlates with weight loss
- **Fasting Hours** (-0.985): Extended fasting correlates with weight loss
- **Sleep** (-0.970): More sleep correlates with weight loss
- **Screen Time** (+0.979): More screen time correlates with weight gain

---

## 📁 Project Structure

```
weight-prediction/
│
├── data.csv                      # Raw dataset (50 records, 5 features)
├── train_model.py               # Model training script
├── model_results.json           # Training results and metrics
├── model.pkl                    # Trained Random Forest model
├── dashboard.html               # Interactive web dashboard
│
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

## 🔍 Dataset

**Size:** 50 records  
**Features:** 5 lifestyle variables  
**Target:** Weight change (kg)

### Features
- **Calories**: Daily calorie intake (kcal)
- **Steps**: Daily physical activity (steps)
- **Sleep**: Average sleep duration (hours)
- **ScreenTime**: Daily screen time (hours)
- **FastingHours**: Intermittent fasting window (hours)

### Target
- **WeightChange**: Weekly weight variation (kg)

---

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weight-prediction.git
cd weight-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python train_model.py
```

This will:
- Load and analyze the dataset
- Display feature correlations
- Train both Linear Regression and Random Forest models
- Compare performance metrics
- Save the best model to `model.pkl`
- Export results to `model_results.json`

### Running the Dashboard

Simply open `dashboard.html` in your web browser or deploy it to a hosting service (Vercel, Netlify, GitHub Pages).

---

## 📊 Exploratory Data Analysis (EDA)

### Key Findings

1. **Strong Correlations**: All features show very strong correlations (|r| > 0.97) with weight change, indicating clear lifestyle-weight relationships

2. **Calorie Impact**: Each additional 100 kcal consumed correlates with ~0.1 kg weight increase

3. **Activity Impact**: Every 1000 additional steps correlates with ~0.013 kg weight loss

4. **Sleep Impact**: Each additional hour of sleep correlates with ~0.04 kg weight loss

5. **Fasting Impact**: Each additional hour of fasting correlates with ~0.05 kg weight loss

---

## 🧠 Machine Learning Models

### Linear Regression
- **Strengths**: Simple, interpretable, fast training
- **Use Case**: Understanding individual feature effects
- **Performance**: R² = 0.9861, MAE = 0.0424 kg

**Model Equation:**
```
WeightChange = 0.0012*Calories - 0.000013*Steps - 0.0395*Sleep - 0.0232*ScreenTime - 0.0502*FastingHours
```

### Random Forest
- **Strengths**: Captures non-linear relationships, handles feature interactions
- **Use Case**: Accurate predictions (chosen as production model)
- **Performance**: R² = 0.9912, MAE = 0.0340 kg
- **Parameters**: 200 trees, max_depth = 15

**Why Random Forest Won:**
- 5.1% better R² score
- 19.8% lower MAE
- Captures complex interactions between lifestyle factors

---

## 🎨 Interactive Dashboard

The dashboard includes:

- **Overview Metrics**: Dataset size, features, model performance
- **Feature Correlation Visualizations**: Bar charts showing feature relationships
- **Model Performance Comparison**: Side-by-side metrics
- **Interactive Predictor Tool**: Adjust lifestyle factors to see predicted weight change
- **Responsive Design**: Mobile-friendly interface

### Usage
1. Adjust the sliders for your lifestyle factors
2. See real-time prediction updates
3. Understand how each factor affects weight

---

## 💡 Methodology

### 1. Data Collection & Preprocessing
- Collected 50 observations of lifestyle and weight data
- No missing values or outliers removed (data quality was high)

### 2. Exploratory Data Analysis
- Calculated feature correlations
- Analyzed weight change distribution
- Identified feature relationships

### 3. Model Training
- **Train-Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)
- **Normalization**: Not required (tree-based models are scale-invariant)

### 4. Model Evaluation
- **Metrics**: MAE, RMSE, R² Score
- **Cross-validation**: Train-test evaluation
- **Comparison**: Linear Regression vs Random Forest

### 5. Deployment
- Converted model to interactive web dashboard
- Deployed on Vercel for accessibility
- Created GitHub repository for code sharing

---

## 📈 Potential Improvements

1. **Expand Dataset**: Collect more observations for better generalization
2. **Feature Engineering**: Create interaction features (e.g., Calories/Steps ratio)
3. **Hyperparameter Tuning**: Use GridSearchCV or Bayesian optimization
4. **Cross-Validation**: Implement k-fold cross-validation
5. **Advanced Models**: Test XGBoost, LightGBM, Neural Networks
6. **Time Series**: Incorporate temporal patterns in weight changes
7. **Personalization**: Train individual models for different demographics

---

## 📦 Requirements

```
pandas>=1.3.0
scikit-learn>=0.24.0
numpy>=1.20.0
```

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

Created as a Data Science Mini Project to demonstrate:
- Machine Learning model development
- EDA and data analysis
- Model comparison and evaluation
- Web application deployment
- GitHub & professional documentation

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add features
- Improve documentation

---

## 📞 Contact

For questions or feedback, feel free to reach out!

---

## 🎓 Learning Resources

This project covers:
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning model training and evaluation
- **Python**: General programming and scripting
- **Statistics**: Correlation analysis and model metrics
- **Web Development**: HTML/CSS/JavaScript for dashboard
- **Deployment**: Hosting and sharing projects online

---

**Last Updated:** 2024  
**Status:** Complete ✓
