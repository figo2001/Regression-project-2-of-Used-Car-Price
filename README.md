# Used Car Price Prediction

This project aims to predict the price of used cars based on various features such as the brand, model year, mileage, fuel type, and more. The goal is to develop a machine learning model that can accurately estimate car prices using historical data.

## Project Overview

The notebook includes the following key steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
2. **Feature Engineering**: Selecting relevant features such as car brand, model year, mileage, fuel type, and accident history.
3. **Model Building**: Using XGBoost to build a regression model to predict car prices.
4. **Model Evaluation**: Measuring the model’s performance using metrics like RMSE.
5. **Price Prediction**: Providing an interface to predict the price of a used car based on input parameters.

## Features

- **Brand**: Car brand identifier.
- **Model**: Specific model of the car.
- **Model Year**: Year in which the car model was released.
- **Mileage**: Total distance covered by the car.
- **Fuel Type**: Type of fuel used by the car (e.g., petrol, diesel).
- **Engine**: Engine size of the car.
- **Transmission**: Type of transmission (manual or automatic).
- **Exterior Color**: Code representing the exterior color of the car.
- **Interior Color**: Code representing the interior color of the car.
- **Accident History**: Binary variable indicating if the car has been involved in an accident.
- **Price**: The target variable representing the price of the car.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required libraries: `xgboost`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`

You can install the required libraries by running:
```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/used-car-price-prediction.git
    ```
2. Navigate to the project folder:
    ```bash
    cd used-car-price-prediction
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook regression-of-used-car-prices.ipynb
    ```
4. Run the cells in the notebook to preprocess the data, train the model, and make predictions.

### Example Usage

Once the model is trained, you can predict the price of a used car by inputting the following parameters:

```python
brand = 31
model = 495
model_year = 2007
mileage = 213000
fuel_type = 2
engine = 116
transmission = 38
ext_col = 312
int_col = 71
accident = 1

price = predict_price(brand, model, model_year, mileage, fuel_type, engine, transmission, ext_col, int_col, accident)
print(f"Predicted price: {price}")
```

## Results

The model predicts the price of a used car based on the input features. In the example above, the predicted price was:

```
Predicted price: [5973.90]
```

## Model Evaluation

The model is evaluated using common regression metrics such as:
- RMSE (Root Mean Squared Error)
- R² (R-squared)

## Screenshots
![Screenshot 2024-10-01 at 11-04-16 app · Streamlit](https://github.com/user-attachments/assets/2b38a020-cf7a-48ed-bb8d-e45c2faa5fe5)

![Screenshot 2024-10-01 at 11-04-05 app · Streamlit](https://github.com/user-attachments/assets/97c2e7b2-4cf2-4777-ae6f-af458a166292)

