# Stock Prediction Engine Using LSTM & MLP

---

## 📝 Overview
This project implements a **stock price prediction engine** using **LSTM** and **MLP** models. The focus is on testing various configurations, documenting the **best hyperparameters**, and presenting model performance.  

Key points:  
- Data preprocessing to avoid leakage and ensure correct time-series handling  
- Training with **early stopping**, **L2 regularization**, and **K-Fold Cross Validation**  
- Hyperparameter tuning to find optimal configurations  
- Evaluation and comparison between LSTM and MLP models  

---

## ⚙️ Model Training & Evaluation
- Both **LSTM** and **MLP** models were tested extensively  
- **Training functions** include preprocessing, evaluation, and hyperparameter search  
- Cross-validation ensures robustness of results  
- **Metrics computed after proper preprocessing** to avoid inconsistencies  

---

## 📊 Results Visualization

### Predicted vs Target
<p align="center">
  <img src="assests/stock_prediction_vs_target.png" width="500"/>
</p>
*Figure 1 – Stock predictions vs actual target values.*

### Scatter Plot of Predictions
<p align="center">
  <img src="assests/scatter_plot_predictions.png" width="500"/>
</p>
*Figure 2 – Scatter plot comparing predicted values against target.*

### Corrected MLP Predictions
<p align="center">
  <img src="assests/mlp_corrected.png" width="500"/>
</p>
*Figure 3 – Corrected MLP predictions using best hyperparameters.*

### Zoomed-in View of Corrected MLP
<p align="center">
  <img src="assests/mlp_corrected_zoomed.png" width="500"/>
</p>
*Figure 4 – Zoomed-in view highlighting details of corrected MLP predictions.*

---

## 🛠 Utilities
- Dynamic preprocessing functions to select proper transformations  
- Training and evaluation functions handling **early stopping**, **regularization**, and **cross-validation**  
- Hyperparameter tuning implemented for optimal model settings  

---

## 📁 Repository Structure

| Path         | Description |
|--------------|-------------|
| `notebooks/` | Jupyter notebooks for testing and experiments (work-in-progress) |
| `utils/`     | Preprocessing and helper functions |
| `assests/`   | Figures and visual results |
| `README.md`  | Project documentation |
| `.gitignore` | Ignored files configuration |

---

## 📄 Notes
- Notebooks are **work-in-progress**  
- The main goal is **documenting best hyperparameters and model testing results**, rather than finished notebooks
