# Groundwater Recharge Mapping Project

## Overview

This project aims to map and predict groundwater recharge structures and their associated risk levels for various districts. The tool leverages historical data, including rainfall, soil permeability, and extraction trends, to determine suitable recharge structures (e.g., recharge wells, check dams, percolation pits) and highlight high-risk areas. By analyzing this data, the project can suggest appropriate solutions to address groundwater depletion in specific regions.
This project is intended for use in regions where groundwater depletion is a concern and serves as a guide to optimizing water resources using predictive models and geographical visualization.

## Key Features

- **Risk Level Classification**: The project classifies districts based on their groundwater extraction levels into "High Risk", "Medium Risk", and "Low Risk" areas.
- **Recharge Structure Prediction**: Based on rainfall, permeability, and extraction levels, the system predicts suitable groundwater recharge structures.
- **Geographical Visualization**: The project uses **Folium** and **Streamlit** to display an interactive map showing risk levels and recharge structures.
- **Predictive Models**: Linear extrapolation models predict future groundwater availability and classify areas for 2024, 2025, and 2026.
- **Data Analysis and Error Metrics**: The project includes a detailed analysis of groundwater availability and predictions, with error metrics (MAE, MSE, RMSE) for model accuracy.

## Technologies Used

- **Python**: The main programming language used for data processing, modeling, and visualization.
- **Streamlit**: Used for building the interactive web application.
- **Folium**: A Python library used for creating interactive maps.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and model predictions.
- **Scikit-learn**: For implementing machine learning algorithms and error metrics.
- **Matplotlib**: For data visualization and plotting graphs.
- **Machine Learning Models**: 
   - **Linear Regression**: Used for predicting future groundwater availability and classifying areas based on historical trends.
   - **Decision Tree**: Used for classifying districts into risk levels (High, Medium, Low) based on various environmental and groundwater extraction factors.

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.6 or later
- Streamlit: For interactive web interfaces
- Pandas: For handling and analyzing data
- NumPy: For numerical computations
- Scikit-learn: For machine learning models and error metrics
- Folium: For mapping functionality
- Matplotlib: For graph plotting

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/groundwater-recharge-mapping.git
   cd groundwater-recharge-mapping
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Start Streamlit
   ```bash
   streamlit run app.py
![image](https://github.com/user-attachments/assets/a74de249-8490-4d7e-8da8-37ddee669147)
![image](https://github.com/user-attachments/assets/fe5e5ca7-ee13-4292-be54-6b7d8934876e)
![image](https://github.com/user-attachments/assets/9c604714-8f61-4265-8730-8c6a51202b4c)
![image](https://github.com/user-attachments/assets/45225bf9-4009-41f4-8cfc-26299ac70fda)



