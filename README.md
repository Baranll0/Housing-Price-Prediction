# Housing Price Prediction

![House Price Prediction](https://i.imgur.com/8w29JyO.jpg)

## Overview

This project focuses on predicting housing prices based on various features such as longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, and ocean proximity. It involves data analysis, visualization, and machine learning to build predictive models.

## Dataset Description

The project uses a dataset with the following columns:

- `longitude`: The longitude of the housing location.
- `latitude`: The latitude of the housing location.
- `housing_median_age`: The median age of the housing units in a specific area.
- `total_rooms`: The total number of rooms in the housing units.
- `total_bedrooms`: The total number of bedrooms in the housing units.
- `population`: The total population in the housing units.
- `households`: The total number of households in the housing units.
- `median_income`: The median income of the residents in the area.
- `median_house_value`: The median value of housing units in dollars.
- `ocean_proximity`: The proximity of the housing units to the ocean.

## Contents

### 1. Correlation Analysis
- Analyzed correlations between different features and the target variable (median house value) to understand their relationships.

### 2. Feature Selection
- Performed feature selection to identify the most relevant features for the prediction models.

### 3. Data Visualization
- Created visualizations to explore data distributions, relationships between variables, and other patterns.

### 4. Data Preparation
- Prepared the data for machine learning by handling missing values, scaling features, and encoding categorical variables.

### 5. Machine Learning
Implemented a range of regression models to predict housing prices:
#### Linear Regression
- Applied simple linear regression as a baseline model.

#### Ridge Regression
- Utilized Ridge regression for regularization and to reduce the risk of overfitting.

#### Elastic Net Regression
- Employed Elastic Net regression, which combines L1 and L2 regularization.

#### K-Nearest Neighbors (KNN)
- Utilized the KNN algorithm for regression.

#### Support Vector Regression (SVR)
- Implemented Support Vector Regression to capture non-linear relationships.

#### Decision Trees
- Employed decision trees for regression analysis.

#### Random Forests
- Used ensemble learning with random forests to improve prediction accuracy.

#### Gradient Boosting
- Implemented Gradient Boosting, which combines multiple decision trees.

#### XGBoost
- Utilized XGBoost, a gradient boosting framework known for its predictive power.

#### Artificial Neural Networks (MLP)
- Applied Multi-Layer Perceptron (MLP) for deep learning.

#### Polynomial Regression
- Employed polynomial regression to capture non-linear relationships.

### 6. Model Comparison
- Compared the performance of various models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.

## Dependencies
- Python 3.11.6
- Jupyter Notebook (for running the code)
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, tensorflow (for MLP)

## How to Run
1. Clone the repository to your local machine.
2. Open the Jupyter Notebook and run the project code cells.

Feel free to explore the project, run the code, and evaluate the performance of different regression models in predicting housing prices.

Happy coding!


