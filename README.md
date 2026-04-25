# Rainfall Prediction Project

This project focuses on predicting rainfall using various meteorological parameters. The project includes data loading, preprocessing, exploratory data analysis, model training with a Random Forest Classifier, evaluation, and a deployment-ready model saving mechanism.

## Data Source

The dataset used for this project is `Rainfall.csv`, which contains daily weather observations including pressure, temperature, humidity, cloud cover, sunshine, wind direction, and wind speed, along with a 'rainfall' target variable (yes/no).

## Project Structure

The notebook walks through the following steps:

1.  **Data Loading and Initial Inspection**: Loading the `Rainfall.csv` dataset into a pandas DataFrame and examining its basic properties (shape, head, tail, unique values, info).
2.  **Data Cleaning and Preprocessing**: 
    *   Stripping whitespace from column names.
    *   Dropping the 'day' column as it's not relevant for prediction.
    *   Handling missing values in 'winddirection' (imputed with mode) and 'windspeed' (imputed with median).
    *   Converting the 'rainfall' target variable from categorical ('yes', 'no') to numerical (1, 0).
3.  **Exploratory Data Analysis (EDA)**:
    *   Generating descriptive statistics.
    *   Visualizing distributions of numerical features using histograms.
    *   Visualizing the distribution of the target variable 'rainfall' using a countplot.
    *   Creating a correlation heatmap to understand relationships between features.
    *   Identifying and visualizing outliers using box plots.
4.  **Feature Selection**: Dropping highly correlated features like 'maxtemp', 'temparature', and 'mintemp' to reduce multicollinearity and improve model performance.
5.  **Handling Imbalanced Data**: The 'rainfall' target variable showed an imbalance. Downsampling the majority class ('rainfall' = 1) to match the minority class ('rainfall' = 0) was performed using `sklearn.utils.resample` to create a balanced dataset.
6.  **Model Training and Evaluation**: 
    *   Splitting the preprocessed data into training and testing sets.
    *   Training a Random Forest Classifier (`RandomForestClassifier`) after performing hyperparameter tuning using `GridSearchCV` to find the best model parameters.
    *   Evaluating the best model's performance using cross-validation scores, test set accuracy, confusion matrix, and a classification report.
7.  **Prediction with a New Sample**: Demonstrating how to make predictions on new, unseen data using the trained model.
8.  **Model Persistence**: Saving the trained Random Forest model and its feature names using `pickle` for future use, and demonstrating how to load and use the saved model.

## Installation

To run this notebook, you'll need the following libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1.  **Clone the repository (if applicable)**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Upload the `Rainfall.csv` file** to your Colab environment or ensure it's in the correct path.
3.  **Run all cells** in the Jupyter/Colab notebook (`Rainfall_Prediction.ipynb` or similar).
4.  The notebook will guide you through the data processing, model training, and evaluation. The final model will be saved as `rainfall_prediction_model.pkl`.

To make a new prediction after the model is saved:

```python
import pickle
import pandas as pd

# Load the saved model
with open('rainfall_prediction_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
feature_names = model_data['feature_names']

# Prepare your input data (ensure it matches the feature order)
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7) # Example values
input_df = pd.DataFrame([input_data], columns=feature_names)

# Make a prediction
prediction = model.predict(input_df)

# Interpret the result
print("Prediction result: ", "Rainfall" if prediction[0] == 1 else "No rainfall")
```
