import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load datasets
dataset1 = pd.read_csv(r"C:\Users\Dell\Downloads\classified_groundwater_data_yearwise.csv")  # First dataset
water_data = pd.read_csv(r"C:\Users\Dell\Downloads\water_dataset.csv")  # Water dataset
merged_dataset = pd.read_csv(r"C:\Users\Dell\Downloads\merged_dataset1.csv")  # Dataset for groundwater prediction

# Step 1: Create a Synthetic Target Variable
def create_synthetic_target(row):
    """
    Create a synthetic target variable based on rainfall, permeability, and extraction trends.
    """
    if row["Rainfall_Classification_2023"] == "Low":
        if row["Extraction_Classification_2023"] == "High":
            return "Recharge Wells"
        elif row["Permeability"] == "Low":
            return "Check Dams"
        else:
            return "Percolation Pits"
    elif row["Rainfall_Classification_2023"] == "Medium":
        if row["Extraction_Classification_2023"] == "Medium":
            return "Farm Ponds"
        elif row["Permeability"] == "Low to Moderate":
            return "Nala Bunds"
        else:
            return "Recharge Wells"
    elif row["Rainfall_Classification_2023"] == "High":
        if row["Permeability"] == "Low":
            return "Nala Bunds"
        elif row["Extraction_Classification_2023"] == "Low":
            return "Farm Ponds"
        else:
            return "Check Dams"
    else:
        return "Percolation Pits"  # Default structure

# Add synthetic target to dataset1
dataset1["Structure_Type"] = dataset1.apply(create_synthetic_target, axis=1)

# Step 2: Synthesize Number_of_Structures
def synthesize_num_structures(row):
    """
    Synthesize the number of structures based on Rainfall_Classification.
    """
    if row["Rainfall_Classification_2023"] == "Low":
        return 10  # More structures if rainfall is low
    elif row["Rainfall_Classification_2023"] == "Medium":
        return 7
    elif row["Rainfall_Classification_2023"] == "High":
        return 5
    else:
        return 5  # Default number of structures

# Add synthesized Number_of_Structures to dataset1
dataset1["Number_of_Structures"] = dataset1.apply(synthesize_num_structures, axis=1)

# Step 3: Train a Random Forest Model for Structure Prediction
features = ["Hilly_Classification", "Permeability", "Extraction_Classification_2023", "Rainfall_Classification_2023"]
target = "Structure_Type"

# Encode categorical variables
dataset1_encoded = pd.get_dummies(dataset1[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(dataset1_encoded, dataset1[target], test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Train Random Forest with best parameters
structure_model = grid_search.best_estimator_
structure_model.fit(X_train, y_train)

# Evaluate model
y_pred = structure_model.predict(X_test)
print("Structure Prediction Accuracy:", accuracy_score(y_test, y_pred))

# Step 4: Predict Groundwater Availability Using Linear Extrapolation
def linear_extrapolation(values, future_years):
    """
    Predict future groundwater availability using linear extrapolation.
    """
    x = np.arange(len(values))  # Time steps (0, 1, 2)
    slope, intercept = np.polyfit(x, values, 1)  # Fit a linear trend
    predictions = {}
    for year in future_years:
        time_step = len(values) + (year - 2023)  # Time step for the future year
        predictions[year] = slope * time_step + intercept  # Predict the value
    return predictions

# Prepare for groundwater predictions
availability_columns = [
    "Net Annual Ground Water Availability for Future Use (ham)_2019",
    "Net Annual Ground Water Availability for Future Use (ham)_2021",
    "Net Annual Ground Water Availability for Future Use (ham)_2022"
]
prediction_years = [2023, 2024, 2025, 2026]

# Convert data to numeric, forcing errors to NaN
merged_dataset[availability_columns] = merged_dataset[availability_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
merged_dataset = merged_dataset.dropna(subset=availability_columns)

# Standardize the data
scaler = StandardScaler()
merged_dataset_scaled = merged_dataset.copy()
merged_dataset_scaled[availability_columns] = scaler.fit_transform(merged_dataset[availability_columns])

# Predict groundwater availability for each district
groundwater_predictions = {}
for index, row in merged_dataset_scaled.iterrows():
    availability_values = row[availability_columns].astype(float).values
    predicted_values = linear_extrapolation(availability_values, prediction_years)
    groundwater_predictions[row['DISTRICT']] = predicted_values

# Step 5: Combine Predictions
# Add structure predictions to dataset1
dataset1["Predicted_Structure"] = structure_model.predict(dataset1_encoded)

# Add groundwater predictions to dataset1
for year in prediction_years:
    dataset1[f"Predicted_Groundwater_{year}"] = dataset1["DISTRICT"].map(
        lambda x: groundwater_predictions.get(x, {}).get(year, np.nan)
    )

# Step 6: Categorize Numerical Data
def classify_groundwater(availability):
    """
    Categorize groundwater availability into High, Medium, or Low.
    """
    if availability > 1000:  # Adjust thresholds as needed
        return "High"
    elif 500 <= availability <= 1000:
        return "Medium"
    else:
        return "Low"

# Add categorized groundwater predictions to dataset1
for year in prediction_years:
    dataset1[f"Groundwater_Category_{year}"] = dataset1[f"Predicted_Groundwater_{year}"].apply(classify_groundwater)

# Step 7: Save Results
dataset1.to_csv("combined_predictions_with_categories.csv", index=False)
print("Combined predictions with categories saved to 'combined_predictions_with_categories.csv'.")

# Step 8: Extract Data for Predictions
# Example: Extract data for a specific district
district_to_extract = "Bagalkot"
district_data = dataset1[dataset1["DISTRICT"] == district_to_extract]

print(f"Data for {district_to_extract}:")
print(district_data[["DISTRICT", "Predicted_Structure", "Predicted_Groundwater_2023", "Groundwater_Category_2023"]])