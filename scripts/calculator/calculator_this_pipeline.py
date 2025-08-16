from transliterate import translit
import scripts.model.model as model
from scripts.preprocesser.constants import MERCATOR_CRS, CONST_BASE_DEMAND

def fix_missing_capacity(row, service_name):
    """this is just a very straight forward workaround"""
    if row["population"] > 10e3 and row[f"capacity_{service_name}"] == 0:
        row[f"capacity_{service_name}"] = row["population"] / 2

    return int(row[f"capacity_{service_name}"])


def make_block_scheme(settl, df_service, service_name):
    blocks_gdf = model.create_blocks(
        settl, const_demand=CONST_BASE_DEMAND, epsg=MERCATOR_CRS
    )
    blocks_gdf = model.update_blocks_with_services(
        blocks_gdf, df_service, service_type=service_name
    )
    # Добавим колонку с транслитерированными именами
    blocks_gdf["name"] = blocks_gdf["name"].apply(
        lambda x: translit(x, "ru", reversed=True)
    )

    blocks_gdf[f"capacity_{service_name}"] = blocks_gdf.apply(lambda x:
        fix_missing_capacity(x, service_name), axis=1
    )
    blocks_gdf = blocks_gdf.to_crs(MERCATOR_CRS)
    return blocks_gdf
