# 💻 Amazon Laptop Price Prediction (Machine Learning Project)

🔗 Repository: https://github.com/Saikoushik14/amazon_laptop_price_prediction_tekworks_task

## 📌 Project Overview

This project focuses on predicting laptop prices using Machine Learning.  
The dataset contains laptop specifications such as Brand, Processor, RAM, Storage, GPU, Operating System, Rating, and Price.

A Linear Regression model is built to predict laptop prices in INR after proper data preprocessing and feature encoding.

---

## 📂 Dataset Information

Dataset File: `amazon_laptop_price_dataset.csv`

### Features:
- Brand
- Processor
- RAM_GB
- Storage_GB
- Operating_System
- GPU
- Rating
- Price_USD

### 🎯 Target Variable:
- Price_INR (Converted from USD)

---

## ⚙️ Data Preprocessing Steps

1. Converted Price from USD to INR.
2. Removed outliers using the IQR (Interquartile Range) method.
3. Filtered only Windows and macOS laptops.
4. Encoded categorical features:
   - Processor → Numerical Encoding
   - GPU → Numerical Encoding
   - Brand → One Hot Encoding
   - Operating_System → Numerical Encoding
5. Split dataset into Training (80%) and Testing (20%) sets.

---

## 🤖 Model Used

- Linear Regression

The model was trained on the processed dataset and evaluated using standard regression metrics.

---

## 📊 Model Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R² Score)

### 🚀 Final R² Score:
**0.94**

This means the model explains approximately **94% of the variance** in laptop prices, indicating strong predictive performance.

---

## 📈 Visualization

An **Actual vs Predicted Price** scatter plot was generated to visually evaluate model performance.

---

## 🛠 Technologies Used

- Python (3.x)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🚀 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Saikoushik14/amazon_laptop_price_prediction_tekworks_task.git
   ```

2. Install required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. Open the notebook:

   ```bash
   jupyter notebook amazon_laptop_price_prediction1.ipynb
   ```

4. Run all cells sequentially.

---

## 📁 Project Structure

```
amazon_laptop_price_prediction_tekworks_task/
│
├── amazon_laptop_price_dataset.csv
├── amazon_laptop_price_prediction1.ipynb
└── README.md
```

---

## 👨‍💻 Author

**Sai Koushik Kasula**  
Machine Learning & Data Science Enthusiast  

---

## 📌 Future Improvements

- Implement Random Forest Regressor
- Perform Hyperparameter Tuning
- Apply Cross-Validation
- Analyze Feature Importance
- Deploy the model using Flask or Streamlit

---

⭐ If you found this project useful, consider giving it a star!
