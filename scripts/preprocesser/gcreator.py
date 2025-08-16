import networkx as nx
import pandas as pd

def make_g(transport_df, transport_modes, blocks_gdf, settl):# Convert G to an undirected graph
    G_undirected = nx.MultiGraph()

    # Add edges from transport_df
    for _, row in transport_df.iterrows():
        for mode in transport_modes:
            if row[mode] != 0:
                G_undirected.add_edge(
                    row["edge1"],
                    row["edge2"],
                    weight=round(row[mode], 2),
                    label=mode,
                )

                G_undirected.add_edge(
                    row["edge2"],
                    row["edge1"],
                    weight=round(row[mode], 2),
                    label=mode,
                )

    # Add node properties from blocks_gdf dataframe
    for _, row in blocks_gdf.iterrows():
        node = row["name"]
        if node in G_undirected.nodes:
            for col in blocks_gdf.columns:
                # if col != "geometry":  # Skip geometry column
                G_undirected.nodes[node][col] = row[col]

                if col == "geometry":
                    # Convert geometry to WKT format
                    geom = row[col]
                    if geom is not None:
                        G_undirected.nodes[node]["x"] = geom.centroid.x
                        G_undirected.nodes[node]["y"] = geom.centroid.y
                    else:
                        G_undirected.nodes[node]["x"] = None
                        G_undirected.nodes[node]["y"] = None

    G_undirected.graph["crs"] = settl.crs.to_epsg()

    return G_undirected

def add_temp_to_g(G_undirected, df_monthly_list):
    for node in G_undirected.nodes:
        if (
            G_undirected.nodes[node]["x"] is not None
            and G_undirected.nodes[node]["y"] is not None
        ):
            name = G_undirected.nodes[node]["name"]

            # Extract temperature data for the node's coordinates
            city_temp = df_monthly_list[df_monthly_list["name"] == name][
                "monthly_temperature"
            ].squeeze()
            G_undirected.nodes[node]["temperature"] = pd.DataFrame(
                city_temp, columns=["temperature"]
            ).round(0)

            # G_undirected.nodes[node]["season"] = G_undirected.nodes[node][
            #     "temperature"
            # ].map(get_season_by_temperature)
    return G_undirected
