import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    file_path = r"C:\Users\Dell\OneDrive\Documents\GitHub\Groundwater-Recharge-Mapping\datasets\classified_groundwater_data_yearwise.csv"
    df = pd.read_csv(file_path)
    return df

dataset1 = load_data()

# District coordinates
district_coords = {
    "Bagalkot": (16.1867, 75.6961), "Ballari": (15.1394, 76.9214), "Belagavi": (15.8497, 74.4977),
    "Bengaluru (Rural)": (13.1979, 77.7066), "Bengaluru (Urban)": (12.9716, 77.5946), "Bidar": (17.9100, 77.5199),
    "Chamarajanagar": (11.9236, 76.9391), "Chikkamagaluru": (13.3153, 75.7754), "Chikkaballapur": (13.4356, 77.7315),
    "Chitradurga": (14.2306, 76.3980), "Dakshina Kannada": (12.8436, 75.2479), "Davanagere": (14.4644, 75.9210),
    "Dharwad": (15.4589, 75.0078), "Gadag": (15.4297, 75.6370), "Hassan": (13.0072, 76.0960),
    "Haveri": (14.7956, 75.3998), "Kalaburagi": (17.3297, 76.8343), "Kodagu": (12.4208, 75.7397),
    "Kolar": (13.1351, 78.1325), "Koppal": (15.3476, 76.1548), "Mandya": (12.5242, 76.8958),
    "Mysuru": (12.2958, 76.6394), "Raichur": (16.2012, 77.3566), "Ramanagara": (12.7138, 77.2814),
    "Shivamogga": (13.9299, 75.5681), "Tumakuru": (13.3379, 77.1173), "Udupi": (13.3409, 74.7421),
    "Uttara Kannada": (14.8197, 74.1340), "Vijayapura": (16.8350, 75.7154), "Yadgir": (16.7706, 77.1376)
}

# Step 1: Create a Synthetic Target Variable (Recharge Structures)
def create_synthetic_target(row, year):
    """
    Create a synthetic target variable based on rainfall, permeability, and extraction trends for a specific year.
    """
    rainfall_col = f"Rainfall_Classification_{year}"
    extraction_col = f"Extraction_Classification_{year}"

    if row[rainfall_col] == "Low":
        if row[extraction_col] == "High":
            return "Recharge Wells"
        elif row["Permeability"] == "Low":
            return "Check Dams"
        else:
            return "Percolation Pits"
    elif row[rainfall_col] == "Medium":
        if row[extraction_col] == "Medium":
            return "Farm Ponds"
        elif row["Permeability"] == "Low to Moderate":
            return "Nala Bunds"
        else:
            return "Recharge Wells"
    elif row[rainfall_col] == "High":
        if row["Permeability"] == "Low":
            return "Nala Bunds"
        elif row[extraction_col] == "Low":
            return "Farm Ponds"
        else:
            return "Check Dams"
    else:
        return "Percolation Pits"  # Default structure

# Step 2: Define Risk Levels based on Extraction_Classification for a specific year
def define_risk_level(row, year):
    """
    Define risk levels based on Extraction_Classification for a specific year.
    """
    extraction_col = f"Extraction_Classification_{year}"
    extraction_level = row[extraction_col]
    if extraction_level == "High":
        return "High Risk"
    elif extraction_level == "Medium":
        return "Medium Risk"
    elif extraction_level == "Low":
        return "Low Risk"
    else:
        return "Unknown Risk"

# Step 3: Create Folium Map for a specific year
def create_map(year):
    """
    Create a Folium map displaying risk levels and recharge structures for a specific year.
    """
    # Initialize map
    m = folium.Map(location=[15.3173, 75.7139], zoom_start=7)

    # Define colors for risk levels
    risk_colors = {
        "High Risk": "#FF0000",  # Red color for High Risk
        "Medium Risk": "#FFA500",  # Orange color for Medium Risk
        "Low Risk": "#008000",  # Green color for Low Risk
        "Unknown Risk": "#808080"  # Grey color for Unknown Risk
    }

    # Add district markers with risk levels and recharge structures
    for _, row in dataset1.iterrows():
        district = row["DISTRICT"]
        coordinates = district_coords.get(district)
        risk_level = define_risk_level(row, year)
        structure_type = create_synthetic_target(row, year)

        if coordinates:
            # Create popup text
            popup_text = f"""
            <b>District:</b> {district}<br>
            <b>Risk Level:</b> {risk_level}<br>
            <b>Recharge Structure:</b> {structure_type}
            """

            # Add marker to the map with increased visibility
            folium.CircleMarker(
                location=coordinates,
                radius=10,  # Increased marker radius for better visibility
                color=risk_colors[risk_level],
                fill=True,
                fill_color=risk_colors[risk_level],
                fill_opacity=0.7,  # Slight transparency for better map visibility
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

    return m


# Step 4: Sidebar for Risk Level Counts for a specific year
def create_sidebar(year):
    """
    Create a sidebar showing the count of high, medium, and low-risk areas for a specific year.
    """
    st.sidebar.header(f"Risk Level Summary ({year})")
    high_risk_count = dataset1[dataset1[f"Extraction_Classification_{year}"] == "High"].shape[0]
    medium_risk_count = dataset1[dataset1[f"Extraction_Classification_{year}"] == "Medium"].shape[0]
    low_risk_count = dataset1[dataset1[f"Extraction_Classification_{year}"] == "Low"].shape[0]

    st.sidebar.write(f"*High Risk Areas:* {high_risk_count}")
    st.sidebar.write(f"*Medium Risk Areas:* {medium_risk_count}")
    st.sidebar.write(f"*Low Risk Areas:* {low_risk_count}")

# Step 5: Graphs for Predictions Over Years
def create_graphs():
    """
    Create graphs for predicted groundwater levels and recharge structures over multiple years.
    """
    st.write("### Predicted Groundwater Levels Over Years")
    # Example: Line chart for predicted groundwater levels
    years = [2019, 2021, 2022, 2023]
    groundwater_data = {
        "Year": years,
        "Predicted_Groundwater": [np.random.uniform(1000, 2000) for _ in years]  # Replace with actual predictions
    }
    groundwater_df = pd.DataFrame(groundwater_data)

    fig, ax = plt.subplots()
    groundwater_df.plot(kind="line", x="Year", y="Predicted_Groundwater", ax=ax, marker="o")
    ax.set_ylabel("Predicted Groundwater (ham)")
    ax.set_xlabel("Year")
    ax.set_title("Predicted Groundwater Levels Over Years")
    st.pyplot(fig)

    st.write("### Recharge Structures Distribution Over Years")
    # Example: Bar chart for recharge structures distribution
    recharge_data = {
        "Year": years,
        "Recharge Wells": [np.random.randint(5, 15) for _ in years],
        "Check Dams": [np.random.randint(3, 10) for _ in years],
        "Percolation Pits": [np.random.randint(2, 8) for _ in years]
    }
    recharge_df = pd.DataFrame(recharge_data)

    fig, ax = plt.subplots()
    recharge_df.plot(kind="bar", x="Year", ax=ax)
    ax.set_ylabel("Number of Structures")
    ax.set_xlabel("Year")
    ax.set_title("Recharge Structures Distribution Over Years")
    st.pyplot(fig)

# Streamlit App
st.title("Groundwater Risk and Recharge Structures Map")

# Dropdown to select year
years = [2019, 2021, 2022, 2023]
selected_year = st.selectbox("Select Year", years)

# Create sidebar for the selected year
create_sidebar(selected_year)

# Display the map for the selected year
st.write(f"### Risk Areas Based on Extraction Classification ({selected_year})")
st.write("*High Risk (Red), **Medium Risk (Orange), **Low Risk (Green)*")
folium_map = create_map(selected_year)
st_folium(folium_map, width=800, height=500)

# Display dataset for the selected year
st.write(f"### Dataset with Risk Levels and Recharge Structures ({selected_year})")
dataset1["Risk_Level"] = dataset1.apply(lambda row: define_risk_level(row, selected_year), axis=1)
dataset1["Structure_Type"] = dataset1.apply(lambda row: create_synthetic_target(row, selected_year), axis=1)
st.dataframe(dataset1[["DISTRICT", "Risk_Level", "Structure_Type"]])

# Display graphs
create_graphs()

# Load dataset for predictions
file_path = r"C:\Users\Dell\OneDrive\Documents\GitHub\Groundwater-Recharge-Mapping\datasets\merged_dataset1.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Selecting relevant columns
years = [2019, 2021, 2022]
prediction_years = [2023, 2024, 2025, 2026]  # Years to predict
availability_columns = [
    "Net Annual Ground Water Availability for Future Use (ham)_2019",
    "Net Annual Ground Water Availability for Future Use (ham)_2021",
    "Net Annual Ground Water Availability for Future Use (ham)_2022"
]
actual_2023_col = "Net Annual Ground Water Availability for Future Use (ham)_2023"

# Convert data to numeric, forcing errors to NaN
df[availability_columns + [actual_2023_col]] = df[availability_columns + [actual_2023_col]].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=availability_columns + [actual_2023_col])

# Standardize the data
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[availability_columns] = scaler.fit_transform(df[availability_columns])

# Define a linear extrapolation function
def linear_extrapolation(values, future_years):
    x = np.arange(len(values))  # Time steps (0, 1, 2)
    slope, intercept = np.polyfit(x, values, 1)  # Fit a linear trend
    predictions = {}
    for year in future_years:
        time_step = len(values) + (year - prediction_years[0])  # Time step for the future year
        predictions[year] = slope * time_step + intercept  # Predict the value
    return predictions

# Prepare for predictions
predictions = {}
actual_values = []
predicted_values_scaled = {year: [] for year in prediction_years}

for index, row in df_scaled.iterrows():
    availability_values = row[availability_columns].astype(float).values
    
    # Predict using linear extrapolation for multiple years
    predicted_values = linear_extrapolation(availability_values, prediction_years)
    
    # Store predictions and actual values
    predictions[row['DISTRICT']] = predicted_values
    actual_values.append(row[actual_2023_col])
    for year in prediction_years:
        predicted_values_scaled[year].append(predicted_values[year])

# Reshape the scaled predictions for inverse transform
for year in prediction_years:
    predicted_values_scaled[year] = np.array(predicted_values_scaled[year]).reshape(-1, 1)

# Add dummy columns to match the original feature count (3 columns)
dummy_columns = np.zeros((len(predicted_values_scaled[prediction_years[0]]), 2))  # Add 2 dummy columns

# Inverse transform the predictions to original scale
predicted_values = {}
for year in prediction_years:
    predicted_values_scaled_with_dummy = np.hstack((predicted_values_scaled[year], dummy_columns))
    predicted_values[year] = scaler.inverse_transform(predicted_values_scaled_with_dummy)[:, 0]  # Extract the first column

# Scale down the units to 10^-3
scale_factor = 1e-3
actual_values_scaled = np.array(actual_values) * scale_factor
for year in prediction_years:
    predicted_values[year] = predicted_values[year] * scale_factor

# Function to classify rainfall
def classify_rainfall(rainfall):
    if rainfall > 1000:
        return "High"
    elif 500 <= rainfall <= 1000:
        return "Medium"
    else:
        return "Low"

# Classify the predicted groundwater availability for 2024, 2025, and 2026
classified_data = []

for index, row in df.iterrows():
    district = row['DISTRICT']
    
    # Classify predicted values for 2024, 2025, and 2026
    predicted_classes = {}
    for year in prediction_years[1:]:  # Skip 2023 (index 0)
        predicted_value = predicted_values[year][index]
        predicted_class = classify_rainfall(predicted_value)
        predicted_classes[f"Rainfall_Classification_{year}"] = predicted_class
    
    # Append the classified data
    classified_data.append({
        'DISTRICT': district,
        **predicted_classes
    })

# Convert the classified data to a DataFrame
classified_df = pd.DataFrame(classified_data)

# Save the classified data to a CSV file
classified_df.to_csv("classified_groundwater_data_yearwise.csv", index=False)

# Display the first few rows of the classified data
st.write("### Classified Groundwater Data")
st.dataframe(classified_df.head())

# Compute error metrics with scaled values for 2023
mae_scaled = mean_absolute_error(actual_values_scaled, predicted_values[2023])
mse_scaled = mean_squared_error(actual_values_scaled, predicted_values[2023])
rmse_scaled = np.sqrt(mse_scaled)

st.write("### Error Metrics for 2023 Predictions")
st.write(f"Mean Absolute Error (MAE) (scaled): {mae_scaled}")
st.write(f"Mean Squared Error (MSE) (scaled): {mse_scaled}")
st.write(f"Root Mean Squared Error (RMSE) (scaled): {rmse_scaled}")

# Plot actual vs. predicted values for 2023
st.write("### Actual vs Predicted Groundwater Availability (2023)")
fig, ax = plt.subplots()
ax.scatter(actual_values_scaled, predicted_values[2023], alpha=0.7, label='Predicted vs. Actual (2023)')
ax.plot([min(actual_values_scaled), max(actual_values_scaled)], [min(actual_values_scaled), max(actual_values_scaled)], 'r--', label='Ideal Prediction')
ax.set_xlabel('Actual 2023 Groundwater Availability (scaled to 10^-3)')
ax.set_ylabel('Predicted 2023 Groundwater Availability (scaled to 10^-3)')
ax.set_title('Actual vs Predicted Groundwater Availability (2023) (Scaled to 10^-3)')
ax.legend()
st.pyplot(fig)

# Plot predictions for all future years
st.write("### Predicted Groundwater Availability for Future Years")
fig, ax = plt.subplots()
for year in prediction_years:
    ax.plot([year] * len(predicted_values[year]), predicted_values[year], 'o', label=f'Predictions for {year}')
ax.set_xlabel('Year')
ax.set_ylabel('Predicted Groundwater Availability (scaled to 10^-3)')
ax.set_title('Predicted Groundwater Availability for Future Years')
ax.legend()
st.pyplot(fig)