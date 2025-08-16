import geopandas as gpd
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def add_parent_to_path():
    """Add parent directory to Python path to enable imports from scripts/"""
    module_path = str(Path.cwd().parent)
    if module_path not in sys.path:
        sys.path.append(module_path)


def load_time_data(main_data_path: str, settl_name: str, transport_cols_warm: List[str], 
                   transport_cols_cold: List[str]) -> pd.DataFrame:
    """Load and merge warm and cold time data"""
    # Load warm time data
    df_time_warm = pd.read_csv(
        main_data_path + f"python/Расчёт времени/warm_time_{settl_name}.csv",
        sep=";",
        header=None,
    )
    df_time_warm.columns = ["edge1", "edge2"] + transport_cols_warm
    
    # Load cold time data
    df_time_cold = pd.read_csv(
        main_data_path + f"python/Расчёт времени/cold_time_{settl_name}.csv",
        sep=";",
        header=None,
    )
    df_time_cold.columns = ["edge1", "edge2"] + transport_cols_cold
    
    # Merge warm and cold time data
    df_time = pd.merge(
        df_time_warm[["edge1", "edge2"] + transport_cols_warm],
        df_time_cold[["edge1", "edge2"] + transport_cols_cold],
        on=["edge1", "edge2"],
        how="outer",
        suffixes=("_warm", "_cold"),
    ).fillna(0)
    
    return df_time


def clean_time_data(df_time: pd.DataFrame, transport_cols_warm: List[str], 
                    transport_cols_cold: List[str]) -> pd.DataFrame:
    """Clean and process time data"""
    # Clean and convert time data
    df_time.loc[:, transport_cols_warm + transport_cols_cold] = df_time[
        transport_cols_warm + transport_cols_cold
    ].apply(lambda x: x.str.replace(",", ".").astype("float32"), axis=1)
    df_time.fillna(0, inplace=True)

    # Calculate mean time for each row
    df_time["mean_time"] = df_time[transport_cols_warm + transport_cols_cold].apply(
        lambda row: row[(row > 0)].mean(), axis=1
    )
    
    return df_time


def check_reversed_edges(df_time: pd.DataFrame) -> None:
    """Check for reversed edge pairs and print results"""
    df_reversed = df_time.copy()
    df_reversed[["edge1", "edge2"]] = df_reversed[["edge2", "edge1"]]
    matches = pd.merge(df_time, df_reversed, on=["edge1", "edge2"], how="inner")

    if not matches.empty:
        print("Строки с обратным порядком найдены:")
    else:
        print("Строк с обратным порядком не найдено.")


def load_settlement_data(main_data_path: str, settl_name: str, 
                        service_amenities_count_column: str) -> gpd.GeoDataFrame:
    """Load and clean settlement data"""
    df_all_data = gpd.read_file(main_data_path + f"{settl_name}_settl.geojson")
    df_all_data = df_all_data[
        ["name", "population", service_amenities_count_column, "geometry"]
    ]
    df_all_data = df_all_data.sort_values("name")
    df_all_data["name"] = df_all_data["name"].str.replace("ё", "е")
    df_all_data["population"] = (df_all_data["population"]).astype("Int64")
    
    return df_all_data


def update_population_data(df_all_data: gpd.GeoDataFrame, settl_name: str) -> None:
    """Update population data for specific settlements"""
    if settl_name == "nao":
        population_updates = {
            "Архангельск": 296665,
            "Воркута": 54223,
            "Искателей": 7253,
            "Красное": 1625,
            "Куя": 152,
            "Мезень": 2832,
            "Нарьян-Мар": 24266,
            "Усинск": 31200,
            "Харьягинский": 544,
        }
        for name, population in population_updates.items():
            df_all_data.loc[df_all_data["name"] == name, "population"] = population


def update_service_data(df_all_data: gpd.GeoDataFrame, service_name: str, 
                       settl_name: str) -> None:
    """Update service data for specific settlements"""
    if service_name == "hospital" and settl_name == "nao":
        df_all_data[service_name] = [
            0, 0, 12, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        ]


def update_capacity_data(df_all_data: gpd.GeoDataFrame) -> None:
    """Update capacity data for major cities"""
    if "Архангельск" in df_all_data["name"].values:
        capacity_updates = {
            "Архангельск": 77021,
            "Воркута": 20034,
            "Нарьян-Мар": 14002,
            "Усинск": 18057,
        }
        for name, capacity in capacity_updates.items():
            df_all_data.loc[df_all_data["name"] == name, "capacity"] = capacity


def calculate_capacity(df_all_data: gpd.GeoDataFrame, service_name: str, 
                      service_amenities_count_column: str, 
                      default_capacity: int) -> None:
    """Calculate capacity based on service count"""
    try:
        df_all_data["capacity"] = df_all_data[service_name] * default_capacity
    except KeyError:
        print(f"Column {service_name} not found in df_all_data. Using default capacity.")
        df_all_data["capacity"] = (
            df_all_data[service_amenities_count_column] * default_capacity
        )


def filter_time_data(df_time: pd.DataFrame, df_all_data: gpd.GeoDataFrame, 
                     settl_name: str, service_name: str) -> pd.DataFrame:
    """Filter time data and remove specific connections if needed"""
    # Remove specific connection for NAO hospital data
    if settl_name == "nao" and service_name == "hospital":
        df_time = df_time[
            ~((df_time["edge1"] == "Коткино") & (df_time["edge2"] == "Лабожское"))
        ]

    # Filter time data to only include settlements in all_data
    df_time = df_time[
        (df_time["edge1"].isin(df_all_data["name"].unique()))
        & (df_time["edge2"].isin(df_all_data["name"].unique()))
    ]
    
    return df_time


def merge_with_geometry(df_time: pd.DataFrame, main_data_path: str, 
                       settl_name: str) -> gpd.GeoDataFrame:
    """Merge time data with link geometry"""
    df_link = gpd.read_file(
        main_data_path + f"qgis/visualisation connections/connections_{settl_name}.gpkg"
    )
    
    # Rename columns in link data if needed
    if "depart" in df_link.columns and "arrival" in df_link.columns:
        df_link.rename(columns={"depart": "settl1", "arrival": "settl2"}, inplace=True)

    # Merge time data with link geometry
    df_time = pd.merge(
        df_time,
        df_link.loc[:, [col for col in df_link.columns if col not in df_time.columns]],
        left_on=["edge1", "edge2"],
        right_on=["settl1", "settl2"],
        how="left",
    ).drop(columns=["settl1", "settl2"])

    # Convert mean time to minutes
    df_time["mean_time"] = (df_time["mean_time"] * 60).round().astype("Int64")
    
    return gpd.GeoDataFrame(df_time, geometry="geometry")


# def create_adjacency_matrix(df_time: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
#     """Create adjacency matrix from time data"""
    

#     adjacency_matrix = pd.DataFrame(
#         0, index=range(len(unique_settlements)), columns=range(len(unique_settlements))
#     )

#     for _, row in df_time.iterrows():
#         edge1, edge2 = row["edge1"], row["edge2"]
#         id1, id2 = settlement_id_map[edge1], settlement_id_map[edge2]
#         adjacency_matrix.at[id1, id2] = row["mean_time"]
#         adjacency_matrix.at[id2, id1] = row["mean_time"]  # Undirected graph

#     return adjacency_matrix, settlement_id_map


def create_output_dataframes(settlement_id_map: Dict[str, int], 
                           df_all_data: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create settlements and service dataframes"""
    settlements_df = pd.DataFrame([
        {"id": idx, "settlement": name} 
        for name, idx in settlement_id_map.items()
    ])

    # Ensure capacity column exists
    if "capacity" not in df_all_data.columns:
        df_all_data["capacity"] = 600  # Default capacity

    # Create service dataframe
    service_df = pd.merge(
        settlements_df,
        df_all_data[["name", "population", "geometry", "capacity"]],
        left_on=["settlement"],
        right_on=["name"],
        how="left",
    )
    service_df = service_df[["id", "name", "population", "geometry", "capacity"]]

    # Merge settlements with all data
    settlements_df = pd.merge(
        settlements_df,
        df_all_data[["name", "population", "geometry"]],
        left_on=["settlement"],
        right_on=["name"],
        how="left",
    )
    settlements_df = settlements_df[["id", "name", "population", "geometry"]]

    return (
        gpd.GeoDataFrame(settlements_df, geometry="geometry"),
        gpd.GeoDataFrame(service_df, geometry="geometry")
    )


def save_processed_data(df_time: gpd.GeoDataFrame, settlements_df: gpd.GeoDataFrame, 
                       service_df: gpd.GeoDataFrame, settl_name: str, 
                       service_name: str) -> None:
    """Save processed data to files"""
    # Convert to projected CRS and save files
    df_time = df_time.to_crs(3857)
    df_time.to_file(f"../data/processed/df_time_{settl_name}.geojson", driver="GeoJSON")

    settlements_df = settlements_df.to_crs(3857)
    settlements_df.to_file(
        f"../data/processed/df_settlements_{settl_name}.geojson", driver="GeoJSON"
    )

    service_df = service_df.to_crs(3857)
    service_df.to_file(
        f"../data/processed/df_{service_name}_{settl_name}.geojson", driver="GeoJSON"
    )


def process_settlement_data(
    default_capacity_if_not_found: int = 600,
    service_amenities_count_column: str = "shop",
    settl_name: str = "mezen",
    transport_cols_warm: List[str] = ["car_warm", "plane_warm", "water_ship", "water_boat"],
    transport_cols_cold: List[str] = ["car_cold", "plane_cold", "winter_tr"],
    main_data_path: str = "../data/unzipped_data/initial_data/Арктика/"
):
    service_name = service_amenities_count_column
    """
    Main function to process settlement data
    
    Args:
        default_capacity_if_not_found: Default capacity value if service not found
        service_amenities_count_column: Column name for service amenities count

        settl_name: Name of the settlement
        transport_cols_warm: List of warm transport columns
        transport_cols_cold: List of cold transport columns
        main_data_path: Path to main data directory
    """

    # Add parent directory to path
    add_parent_to_path()
    
    # Load and process time data
    df_time = load_time_data(main_data_path, settl_name, transport_cols_warm, transport_cols_cold)
    df_time = clean_time_data(df_time, transport_cols_warm, transport_cols_cold)
    check_reversed_edges(df_time)
    
    # Load and process settlement data
    df_all_data = load_settlement_data(main_data_path, settl_name, service_amenities_count_column)
    if 'pristan' in df_all_data.columns:
        df_all_data.rename(columns={'pristan': 'marina'}, inplace=True)
        service_name = 'marina'
        # print(df_all_data.head())
    update_population_data(df_all_data, settl_name)
    update_service_data(df_all_data, service_name, settl_name)
    update_capacity_data(df_all_data)
    calculate_capacity(df_all_data, service_name, service_amenities_count_column, default_capacity_if_not_found)
    
    # Filter and merge data
    df_time = filter_time_data(df_time, df_all_data, settl_name, service_name)
    df_time = merge_with_geometry(df_time, main_data_path, settl_name)
    df_all_data.reset_index(inplace=True, drop=True)
    
    # Create adjacency matrix and output dataframes
    # adjacency_matrix, settlement_id_map = create_adjacency_matrix(df_time)
    unique_settlements = sorted(set(df_time["edge1"]).union(set(df_time["edge2"])))
    settlement_id_map = {name: idx for idx, name in enumerate(unique_settlements)}
    settlements_df, service_df = create_output_dataframes(settlement_id_map, df_all_data)
    
    # Save processed data
    save_processed_data(df_time, settlements_df, service_df, settl_name, service_name)
