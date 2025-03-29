import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Load the dataset
file_path = r"C:\Users\saima\OneDrive\Desktop\gwd\data\selected_groundwater_data.csv" # Ensure this file is in your project folder
df = pd.read_csv(file_path)

# Define district coordinates
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

# Convert percentage columns to numeric for plotting
percentage_columns = [
    "Stage of Groundwater Extraction %_2019", "Stage of Ground Water Extraction (%)_2021",
    "Stage of Groundwater Extraction %_2022", "Stage of Groundwater Extraction %_2023"
]
df[percentage_columns] = df[percentage_columns].apply(pd.to_numeric, errors='coerce')

# Plot groundwater extraction trends
plt.figure(figsize=(12, 6))
for year in percentage_columns:
    sns.lineplot(x=df["DISTRICT"], y=df[year], label=year.split("_")[-1])
plt.xticks(rotation=90)
plt.ylabel("Groundwater Extraction (%)")
plt.xlabel("District")
plt.title("Groundwater Extraction Over the Years")
plt.legend(title="Year")
plt.tight_layout()
plt.savefig("groundwater_extraction_chart.png")
plt.show()

# Create interactive map
m = folium.Map(location=[14.5, 75.5], zoom_start=7)
for _, row in df.iterrows():
    district = row["DISTRICT"]
    coordinates = district_coords.get(district)
    if coordinates:
        folium.CircleMarker(
            location=coordinates,
            radius=row["Stage of Groundwater Extraction %_2023"] / 10,
            color="red" if row["Stage of Groundwater Extraction %_2023"] > 100 else "orange",
            fill=True,
            fill_color="red" if row["Stage of Groundwater Extraction %_2023"] > 100 else "orange",
            fill_opacity=0.7,
            popup=f"{district}: {row['Stage of Groundwater Extraction %_2023']}%"
        ).add_to(m)
m.save("groundwater_map.html")

print("Analysis completed! View 'groundwater_extraction_chart.png' and 'groundwater_map.html'.")
