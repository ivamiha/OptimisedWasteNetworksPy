import src.network as network
import src.optimisation as optimisation
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim


# config Nominatim geolocator, user_agent is your IP address to limit API calls
geolocator = Nominatim(user_agent="192.168.1.117")

# read region csv file containing sources
region = pd.read_csv("data/DE_NUTS1.csv")
# extract source coordinates and save them in a list
sources = []
for _, row in region.iterrows():
    city = geolocator.geocode(f"{row['Largest_city']}, Germany")
    sources.append((city.latitude, city.longitude))

# create list with names of chemical parks of interest
chemical_parks = ["Dormagen", "Brunsbüttel", "Stade"]
# extract customer coordinates and save them in a list
customers = []
for name in chemical_parks:
    city = geolocator.geocode(f"{name}, Germany")
    customers.append((city.latitude, city.longitude))

img_path = "icons/DE.png"
region_name = "Germany"
distances = network.region_setup(sources, customers, img_path, region_name)

# specify product set for the value chain
products = [
    "ETICS",
    "compressed_ETICS",
    "pre_concentrate",
    "pyrolysis_oil",
    "styrene",
]

# specify product capacity at sources using population density [tons/day]
ETICS_tons_per_year = 17100  # HBCD-free ETICS in 2017 as per Conversion study
ETICS_tons_per_day = ETICS_tons_per_year / 360
population = region["Population"].sum()
s_list, s_values = [], []
idx = 0
for _, row in region.iterrows():
    s_list.append("S_" + str(idx))
    idx += 1
    for jdx in range(0, len(products)):
        if jdx == 0:
            s_values.append(ETICS_tons_per_day * row["Population"] / population)
        else:
            s_values.append(0)
s_values = np.array(s_values).reshape((len(sources), len(products)))
source_capacity = pd.DataFrame(s_values, columns=products, index=s_list)

# specify demand of the product set [tons/day]
# demand = [0, 0, 0, 0, 0, s_values.sum() / 20 / len(customers)]
demand = [0, 0, 0, 0, 0, 10000000]

# specify market price of the product set [euro/ton]
market_price = {
    "ETICS": 0,
    "compressed_ETICS": 0,
    "pre_concentrate": 0,
    "pyrolysis_oil": 0,
    "styrene": 1650,
}

# specify yield factor of the different technologies
yield_factor = {
    ("ETICS", "OCF"): 0,
    ("compressed_ETICS", "OCF"): 0.95,
    ("pre_concentrate", "OCF"): 0,
    ("pyrolysis_oil", "OCF"): 0,
    ("styrene", "OCF"): 0,
    ("ETICS", "MPF"): 0,
    ("compressed_ETICS", "MPF"): 0,
    ("pre_concentrate", "MPF"): 0.10,
    ("pyrolysis_oil", "MPF"): 0,
    ("styrene", "MPF"): 0,
    ("ETICS", "CPF"): 0,
    ("compressed_ETICS", "CPF"): 0,
    ("pre_concentrate", "CPF"): 0,
    ("pyrolysis_oil", "CPF"): 0.69,
    ("styrene", "CPF"): 0,
    ("ETICS", "DPF"): 0,
    ("compressed_ETICS", "DPF"): 0,
    ("pre_concentrate", "DPF"): 0,
    ("pyrolysis_oil", "DPF"): 0,
    ("styrene", "DPF"): 0.75,
}

# specify maximum facility capacities [tons/day]
facility_capacity = {"OCF": 0.4, "MPF": 41, "CPF": 12, "DPF": 30}

# initiate ``Infrastructure`` class based on distances from building network
scenario = optimisation.Infrastructure(
    distances[0], distances[1], distances[2], distances[3], distances[4]
)

# define the value chain
scenario.define_value_chain(
    products, source_capacity, facility_capacity, demand, yield_factor, market_price
)

# create optimisation model of the value chain
scenario.model_value_chain()

# solve the optimisation problem
scenario.model.optimize()

# process optimisation problem results
# scenario.process_results()

# plot resulting infrastructure
# scenario.plot_resulting_infrastructure()

# plot resulting product flow
# scenario.plot_resulting_product_flow()
