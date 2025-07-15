
threshold = 0.55

DEMAND_COL = "demand_without"
KELVIN_TO_CELSIUS = 273.15
CONST_BASE_DEMAND = 120  # if not known (!)
SERVICE_RADIUS_MINUTES = 60 * 13
MINUTES_IN_HOUR = 60
MERCATOR_CRS = 3857
START_YEAR = 1982
MONTHS_IN_YEAR = 12
END_YEAR = 2024
FONT_SIZE = 16


# Параметры
start_date = str(START_YEAR) + "0101"
end_date = str(END_YEAR) + "1201"
parameters = "T2M"

transport_modes_color = {
    "Aviation": "#BF616A",
    "Regular road": "#EBCB8B",
    "Winter road": "#A3BE8C",
    "Water transport": "#B48EAD",
    # "Winter road (regular)": "#A3BE8C",
    # "Water transport (ship)": "#B48EAD",
}

# # Пороговые температуры
# thresholds = {
#     "Aviation": -44.6,
#     "Winter road": -11.2,
#     "Water transport": -2,
#     "Regular road": 7.3,
# }

transport_modes = list(transport_modes_color.keys())

transport_mode_name_mapper = {
    "car_warm": "Regular road",
    "plane_cold": "Aviation",
    "plane_warm": "Aviation",
    "water_ship": "Water transport",
    "water_boat": "Water transport",
    "car_cold": "Winter road",
    "winter_tr": "Winter road",
}

settl_node_color = "#88C0D0"
service_node_color = "#5E81AC"


# Упорядочим месяцы
month_order = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

data_path = "../data/processed/"

service_list = [
        "school",
        "kindergarten",
        "post",
        "library",
        "culture",
        "atm",
        "health",
        "port",
        "airport",
        "shop",
        "pristan",]

settl_list = ["yanao_kras","mezen","yakut_chuk"]