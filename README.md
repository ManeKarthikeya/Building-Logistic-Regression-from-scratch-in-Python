---

# Logistic Regression from Scratch in Python

## 📌 Overview
This project implements a **Logistic Regression classifier from scratch** using only NumPy, without relying on high-level machine learning libraries like scikit-learn. The model is trained and tested on the PIMA Diabetes Dataset.

## 📖 Theory Summary

### 1. Hypothesis (Sigmoid Function)
$$
h_\theta(x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \theta^T x
$$

### 2. Cost Function (Log Loss)
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i)) \right]
$$

### 3. Gradient Descent Update Rule
$$
\theta_j := \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
$$

### 4. Gradient Derivative
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_{ij}
$$

### 5. Vectorized Update Formula
$$
\theta := \theta - \alpha \cdot \frac{1}{m} \cdot X^T (h - y)
$$

### 6. Learning Rate
$$
0 < \alpha \le 1 \quad (\text{Typical values: } 0.01, 0.001, 0.1)
$$

### 7. Sigmoid Derivative
$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

---

## 🧠 Implementation Details

### Class: `Logistic_Regression`
- **Initialization**: Sets learning rate and number of iterations.
- **fit(X, Y)**: Trains the model using gradient descent.
- **update_weight()**: Performs weight and bias updates.
- **predict(X)**: Returns binary predictions based on a 0.5 threshold.

### Key Steps:
1. **Data Preprocessing**: Standardization using `StandardScaler`.
2. **Train-Test Split**: 80% training, 20% testing.
3. **Model Training**: Gradient descent optimization.
4. **Evaluation**: Accuracy score on training and test sets.
5. **Prediction**: Example prediction for a new data point.

---

## 📊 Dataset: PIMA Diabetes
- **Rows**: 768
- **Columns**: 8 features + 1 target (`Outcome`)
- **Target**:
  - `0` → Non-Diabetic
  - `1` → Diabetic

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn
```

### 3. Run the script
Ensure `diabetes.csv` is in the same directory or update the path in the script.

---

## 📈 Results
- **Training Accuracy**: ~79%
- **Test Accuracy**: ~75%

Example prediction:
```
Input: (6,148,72,35,0,33.6,0.627,50)
Output: The person is diabetic
```

---

## 🛠️ Dependencies
- NumPy
- Pandas
- Scikit-learn (for preprocessing and evaluation only)

---

## 📝 Files
- `logistic_regression_from_scratch.py` – Main implementation
- `diabetes.csv` – Dataset (included in repo, download from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))
- `README.md` – This file

---

## 📌 Key Learnings
- Understanding the math behind logistic regression
- Implementing gradient descent manually
- Preprocessing data for ML models
- Evaluating classification performance

---

*This project is for educational purposes to demonstrate the inner workings of logistic regression.*

---
