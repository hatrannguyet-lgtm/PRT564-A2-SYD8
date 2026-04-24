The repository contains 5 files:
1. README.txt
2. QAW2223DC (2).xlsx (2022 data) 
3. Qualifications and work 2018-19 Data Tables (1).xlsx (2018 data)
4. Minh Tran Part 1 Assignment 2.ipynb
5.
6. abs-qualification-and-work part 3-4 Muhibul Azim.ipynb


Income Prediction with Ridge Regularization
This project utilizes Ridge Regularized Linear Regression to predict weekly income based on an individual’s qualifications. 
The primary goal is to model how educational qualifications impact income, helping policymakers and organizations make informed decisions about workforce planning and educational investment.
Objective
The objective of this analysis is to develop a predictive model that estimates an individual’s income using the following features:
Age
Gender
Qualification Level Completed
Number of Qualifications
Libraries Used
pandas: Data manipulation and preprocessing
numpy: Numerical operations
matplotlib, seaborn: Visualization
scikit-learn: Machine learning (Ridge regression, model evaluation)
Process Overview
Data Loading:
Load and preprocess the data for model training and testing.
Model Training:
Train the Ridge Regularized Linear Regression model using the training data.
Model Evaluation:
Evaluate the model's performance using metrics like RMSE, MAE, and R².
Hyperparameter Tuning:
Perform hyperparameter tuning to find the best regularization strength (alpha).
Model Inference:
Use the trained model to predict income based on new data.
Results
The final model, Ridge Regression with alpha = 0.175, was selected after testing multiple models.
Metrics:
RMSE: 0.245
MAE: 0.213
R²: 0.713
The model successfully predicted income trends based on educational qualifications but showed some overestimation for certain predictions, which could be due to under-regularization or missing features.
Visualizations
Bar Plot: Comparison of RMSE, MAE, and R² for different models.
Box Plot & Violin Plot: Distribution of RMSE for models to compare stability.
Heatmap: Visualization of 95% confidence intervals for Bootstrap RMSE.
The Ridge regression model effectively captures the relationship between education and income.
Further model tuning and feature engineering may improve prediction precision.
License
This project is licensed under the MIT License.



