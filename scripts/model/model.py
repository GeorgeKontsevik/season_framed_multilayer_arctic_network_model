import math
from itertools import product
import random
import warnings


import numpy as np
from tqdm import tqdm, trange
# from lscp import LSCP
import pulp
import geopandas as gpd
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD


warnings.filterwarnings("ignore", category=FutureWarning)


def create_blocks(
    not_yet_blocks_gdf: gpd.GeoDataFrame, const_demand: int, epsg: int
) -> list[dict]:
    """
    Converts a GeoDataFrame of blocks into a list of block dictionaries.
    """
    blocks = []
    for _, row in not_yet_blocks_gdf.iterrows():
        pops = int(x) if (x := row.get("population")) else const_demand
        # demand = math.ceil(pops / 1000) if pops > const_demand else const_demand
        # capacity = row.get("capacity", 0)
        block = {
            "id": row["id"],
            "name": row["name"],
            "geometry": row["geometry"],
            "population": pops,
            # "demand": demand,
            # "capacities": capacity,
            "epsg": epsg,
        }
        blocks.append(block)
    
    # Create a GeoDataFrame for blocks
    blocks_gdf = gpd.GeoDataFrame(blocks).set_geometry("geometry").set_crs(epsg=epsg)

    return blocks_gdf


# def create_graph(matrix: pd.DataFrame) -> nx.DiGraph:
#     """
#     Creates a directed graph from a distance matrix and blocks.
#     """
#     #blocks: list[dict]

#     graph = nx.DiGraph()
#     # block_by_id = {block["id"]: block for block in blocks}

#     for i in matrix.index:
#         for j in matrix.columns:
#             # Используем id блоков как узлы графа
#             graph.add_edge(
#                 i,  # id первого блока
#                 j,  # id второго блока
#                 weight=matrix.loc[i, j],  # вес ребра
#             )

#     return graph


def update_blocks_with_services(
    blocks_gdf: gpd.GeoDataFrame,
    services_gdf: gpd.GeoDataFrame,
    service_type: str,
) -> gpd.GeoDataFrame:
    """
    Updates blocks GeoDataFrame with service capacities.
    """
    # Check CRS match
    assert blocks_gdf.crs.to_epsg() == services_gdf.crs.to_epsg(), "CRS mismatch"
    if 'capacity' in services_gdf.columns:
        services_gdf.rename(columns={"capacity": f"capacity_{service_type}"}, inplace=True)

    # First join to identify which services are in which blocks
    joined = gpd.sjoin(services_gdf, blocks_gdf, how="inner", predicate="intersects")
    
    
    # Then dissolve to aggregate capacities within each block
    aggregated = joined.dissolve(
        by="index_right",
        aggfunc={f"capacity_{service_type}": "sum"}
    )

    # Create capacity column name 
    capacity_col = f"capacity_{service_type}"
    
    # Update blocks with capacities
    blocks_gdf[capacity_col] = 0  # Initialize with zeros
    blocks_gdf.loc[aggregated.index, capacity_col] = aggregated[f"capacity_{service_type}"]

    return blocks_gdf

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from itertools import product
import pandas as pd
import geopandas as gpd

def lp_provision(
    city_model: dict, gdf: gpd.GeoDataFrame, service_type: str
) -> gpd.GeoDataFrame:
    """
    Linear programming assessment method for resource allocation using block names.
    """
    accessibility = city_model["service_types"][service_type]["accessibility"]
    gdf = gdf.copy()

    delta = gdf["demand"].sum() - gdf[f"capacity_{service_type}"].sum()
    fictive_block_id = "FICTIVE_BLOCK"

    if delta > 0:
        # Добавляем фиктивную строку с capacity
        gdf.loc[fictive_block_id, f"capacity_{service_type}"] = delta
        gdf.loc[fictive_block_id, "demand"] = 0
    elif delta < 0:
        # Добавляем фиктивную строку с demand
        gdf.loc[fictive_block_id, "demand"] = -delta
        gdf.loc[fictive_block_id, f"capacity_{service_type}"] = 0

    gdf["capacity_left"] = gdf[f"capacity_{service_type}"]

    def _get_weight(name1, name2):
        if name1 == name2:
            return 0
        if fictive_block_id in [name1, name2]:
            return 0
        try:
            return int(city_model['graph'][name1][name2]['weight'])
        except Exception as e:
            try:
                return int(city_model['graph'][name2][name1]['weight'])
            except Exception as e:
                # print(f"Error getting weight between {name1} and {name2}: {e}, {city_model['graph']}")
                return int(1e3)

    demand = gdf[gdf["demand"] > 0]
    capacity = gdf[gdf[f"capacity_{service_type}"] > 0]

    prob = LpProblem("Transportation", LpMinimize)
    x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None)
    prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in x)

    for n in demand.index:
        prob += lpSum(x[n, m] for m in capacity.index) == demand.loc[n, "demand"]
    for m in capacity.index:
        prob += lpSum(x[n, m] for n in demand.index) <= capacity.loc[m, f"capacity_{service_type}"]

    prob.solve(PULP_CBC_CMD(msg=False))

    # Before processing results, initialize new columns
    gdf["closest_service"] = None
    gdf["distance_to_service"] = float('inf')
    gdf["assigned_to"] = None

    # Process the results
    assignments = pd.DataFrame(0, index=gdf.index, columns=gdf.index)

    # ensure columns exist
    if "demand_within" not in gdf.columns:
        gdf["demand_within"] = 0
    if "demand_without" not in gdf.columns:
        gdf["demand_without"] = 0

    # Find closest service and process assignments
    for (a, b), var in x.items():
        value = var.varValue
        if value == 0 or (a == fictive_block_id or b == fictive_block_id):
            continue

        weight = _get_weight(a, b)
        assignments.loc[a, b] = value

        # Update closest service if this is closer
        if weight != 1e3:
            if  weight < gdf.loc[a, "distance_to_service"]:
                # print(f"Updating closest service for {a} to {b} with weight {weight}, value {value}, {gdf.loc[a, 'distance_to_service']}")
                gdf.loc[a, "closest_service"] = b
                gdf.loc[a, "distance_to_service"] = weight

            # Update assignment information
            if value > 0:
                if gdf.loc[a, "assigned_to"] is None:
                    gdf.loc[a, "assigned_to"] = b
                else:
                    gdf.loc[a, "assigned_to"] = f"{gdf.loc[a, 'assigned_to']},{b}"

            if weight <= accessibility:
                gdf.loc[a, "demand_within"] += value
            else:
                gdf.loc[a, "demand_without"] += value

            gdf.loc[b, "capacity_left"] -= value

    if fictive_block_id in gdf.index:
        gdf.drop(index=fictive_block_id, inplace=True)
        assignments.drop(index=fictive_block_id, errors="ignore", inplace=True)
        assignments.drop(columns=fictive_block_id, errors="ignore", inplace=True)

    return gdf, assignments

def calculate_provision(
    city_model: dict,
    service_type: str,
    # blocks_gdf: gpd.GeoDataFrame,
    update_df: pd.DataFrame = None,
    method: str = "lp",
) -> gpd.GeoDataFrame:
    """
    Provision assessment using a certain method for the given city and service type.
    Can be used with updated blocks GeoDataFrame.
    """

    # Prepare the GeoDataFrame for provision assessment
    def _get_blocks_gdf(city_model):
        return gpd.GeoDataFrame(city_model["blocks"]).set_index("name").set_crs(epsg=city_model["epsg"])

    gdf = _get_blocks_gdf(city_model)

    # Update the GeoDataFrame if update_df is provided
    if update_df is not None:
        gdf = gdf.join(update_df)
        gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
        if "population" not in gdf:
            gdf["population"] = 0
        gdf["demand"] += gdf["population"].apply(
            lambda pop: math.ceil(pop / 1000 * service_type["demand"])
        )
        gdf["demand_left"] = gdf["demand"]
        if service_type["name"] not in gdf:
            gdf[service_type["name"]] = 0
        gdf["capacity"] += gdf[service_type["name"]]
        gdf["capacity_left"] += gdf[service_type["name"]]

    # Choose the method for provision assessment
    if method == "lp":
        gdf, assignments = lp_provision(city_model, gdf, service_type)
    elif method == "iterative":
        raise NotImplementedError("Iterative method is not implemented yet.")
    else:
        raise ValueError("Invalid method specified. Use 'lp' or 'iterative'.")

    # Calculate the provision ratio
    gdf["provision"] = gdf["demand_within"] / gdf["demand"]

    return gdf.fillna(0), assignments
