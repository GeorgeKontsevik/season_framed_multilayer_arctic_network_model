"""берем нижнюю и верхнюю температуры"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
from shapely.geometry import Point
import json
import requests

from scripts.preprocesser.constants import data_path, start_date, end_date, parameters, MERCATOR_CRS

def refine_data(df_climate):
    climate_records = []

    for _, row in df_climate.iterrows():
        name = row["name"]
        lon = row["lon"]
        lat = row["lat"]
        try:
            temp_dict = eval(row["temperature"])
        except Exception:
            temp_dict = row["temperature"]

        # Convert dict to Series → DataFrame
        temp_series = pd.Series(temp_dict).astype(float)
        temp_series.index = pd.to_datetime(temp_series.index, format="%Y%m%d")
        temp_df = temp_series.rename("temperature").reset_index()
        temp_df = temp_df.rename(columns={"index": "date"})

        # Compute month
        temp_df["month"] = temp_df["date"].dt.to_period("M")

        # Group and convert to list
        monthly_avg = (
            temp_df.groupby("month")["temperature"]
            .mean()
            .sort_index()
            .tolist()  # 👈 collapse to list
        )

        climate_records.append(
            {
                "name": name,
                "lon": lon,
                "lat": lat,
                "monthly_temperature": [int(round(x)) for x in monthly_avg],  # round to int
            }
        )
    return pd.DataFrame(climate_records)


def call_nasa(blocks_gdf, CLIMATE_DATA_FILE_NAME):

    # Папка для кэша
    output_dir = "../data/climate_data_blocks"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if CLIMATE_DATA_FILE_NAME in os.listdir(data_path):
        df_climate = pd.read_csv(data_path + CLIMATE_DATA_FILE_NAME, index_col=0)

    else:
        blocks_gdf = blocks_gdf.to_crs(4326)
        # Будем собирать результаты
        climate_results = []

        for idx, row in tqdm(blocks_gdf.iterrows(), total=len(blocks_gdf)):
            name = row["name"]
            centroid: Point = row["geometry"].centroid
            lon, lat = centroid.x, centroid.y

            out_file = os.path.join(output_dir, f"{name}_{start_date}_{end_date}.json")

            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    data = json.load(f)
            else:
                url = (
                    f"https://power.larc.nasa.gov/api/temporal/daily/point"
                    f"?parameters={parameters}"
                    f"&community=RE"
                    f"&longitude={lon}&latitude={lat}"
                    f"&start={start_date}&end={end_date}"
                    f"&format=JSON"
                )
                r = requests.get(url)
                r.raise_for_status()
                data = r.json()
                with open(out_file, "w") as f:
                    json.dump(data, f)

            temperature = data["properties"]["parameter"]["T2M"]

            climate_results.append(
                {"name": name, "lon": lon, "lat": lat, "temperature": temperature}
            )

        df_climate = pd.DataFrame(climate_results)
        df_climate.to_csv(data_path + CLIMATE_DATA_FILE_NAME)

        blocks_gdf = blocks_gdf.to_crs(MERCATOR_CRS)

    return refine_data(df_climate)
