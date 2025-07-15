from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
from transliterate import translit

from constants import MINUTES_IN_HOUR

def get_data(data_path, SETTL_NAME, transport_mode_name_mapper, transport_modes, SERVICE_NAME):
    settl = gpd.read_file(data_path + f"df_settlements_{SETTL_NAME}.geojson")
    settl["name"] = settl["name"].str.replace("ё", "е")
    settl.geometry = settl.geometry.buffer(1e3)

    transport_df = (
        gpd.read_file(data_path + f"df_time_{SETTL_NAME}.geojson")
        .dropna(subset=["geometry"])
        .reset_index(drop=True)
    )


    transport_df.rename(
        columns=transport_mode_name_mapper,
        inplace=True,
    )


    duplicates = list(
        {item for item, count in Counter(transport_df.columns).items() if count > 1}
    )

    for i in duplicates:
        _df = transport_df[i].copy()
        _col1 = _df.iloc[:, 0].astype(float)
        _col2 = _df.iloc[:, 1].astype(float)

        transport_df.drop(columns=[i], inplace=True)

        combined = np.where(
            _col1 != 0,
            np.where(_col2 != 0, np.where(_col1 < _col2, _col1, _col2), _col1),
            np.where(_col2 == 0, 0, _col2),
        )  # if both are zero, just take col1

        transport_df = transport_df.join(pd.DataFrame({i: combined}))

    # # Convert mode values from ',' to '.' and ensure t≠hey are float
    for mode in transport_modes:
        transport_df[mode] = (
            transport_df[mode].astype(str).str.replace(",", ".").astype(float)
        )
    transport_df["edge1"] = transport_df["edge1"].str.replace("ё", "е")
    transport_df["edge2"] = transport_df["edge2"].str.replace("ё", "е")
    transport_df[transport_modes] *= MINUTES_IN_HOUR  # convert from hours to minutes
    transport_df[transport_modes] = transport_df[transport_modes].round(0)

    infr_df = pd.read_csv(data_path + f"infrastructure_{SETTL_NAME}.csv", sep=";")
    infr_df.fillna(0, inplace=True)
    infr_df["name"] = infr_df["name"].str.replace("ё", "е")

    df_service = gpd.read_file(data_path + f"df_{SERVICE_NAME}_{SETTL_NAME}.geojson")
    df_service["name"] = df_service["name"].str.replace("ё", "е")

    transport_df["edge1"] = transport_df["edge1"].apply(
        lambda x: translit(x, "ru", reversed=True)
    )
    transport_df["edge2"] = transport_df["edge2"].apply(
        lambda x: translit(x, "ru", reversed=True)
    )
    transport_df.drop(columns=["mean_time", "winter", "water"], inplace=True)


    return settl, df_service, transport_df, infr_df





