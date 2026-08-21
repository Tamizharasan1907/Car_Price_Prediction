# Car Price Prediction :

## Problem :
The price of the Car was depends on multiple factors like Manufactured Year, Fuel type, Engine, Max_Power, No. of Seats, Owners, Driven KM, Mileage.

The goal of this project was to build a machine learning model that could predicts the selling price of the Car based on the data collected from Cardekho.com which helps the customer to buy the car with better values.

## Tasks :
- Analyze and clean the car price dataset
- Handle outliers, skewness, and categorical features
- Remove leakage-prone and redundant features

## Planning & execution :
To solve the problem, I designed an end-to-end machine learning pipeline.

### 1) Data Understanding & EDA
Loaded a dataset with 8128 rows and 12 features
Features included:
- Name 
- Year
- KM_Driven
- Fuel
- Seller_Type
- Transmission
- Owner
- Mileage
- Engine
- Max Power
- Seats
- Selling Price

### 2) Explored Data Quality and Distribution
Checked dataset shape, info, and summary statistics and identified that,
- Selling Price was heavily right-skewed
- KM_Driven had extreme outliers
- Some rows had duplicated which is removed

### 3) Target Transformation
Since Selling Price was highly skewed, I applied log transformation and that made the target distribution more normal and easier for the model to learn. Final predictions can later be converted back using np.expm1().

### 4) Encoding Categorical Variables
Since the dataset had categorical columns, I used the encoding strategies like One Hot Encoder for Fuel, seller_type, transmission, owner.
This helped convert categorical variables into numeric form while preserving useful yield patterns.

### 5) Train-Test Split
- Defined:
    - X = all features except selling price
    - y = selling price
- Split the data into:
    - 80% training
    - 20% testing

### 6) Model Building

I've trained the Linear Regression Model with X_train.

### 7) Model Evaluation

I evaluated the model using:

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Python Libraries :
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Linear Regression
- Random Forest Regression

## Core Competencies :
- Data Collection
- Data Cleaning
- EDA
- Feature Selection
- Data Preprocessing
- Model training
- Prediction system
- Model Evaluation Metrics (MSE, MAE, R2, RMSE)