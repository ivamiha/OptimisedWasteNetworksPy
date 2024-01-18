import os
import src.network as network
import src.optimisation as optimisation
import pandas as pd
import numpy as np
import utils.get_coords as gc
import utils.convert_coords as cc

# create the /results directory unless it already exists
try:
    os.makedirs("./results")
except FileExistsError:
    pass

# read region csv file containing sources
region = pd.read_csv("data/DE_NUTS1.csv")
# define country to which the region corresponds
country = "Germany"

# extract & process city coordinates, NOTE: comment out when data processed once
cities = []
distance = []
x_distance = []
y_distance = []
country_coords = gc.get_country_coords(country)
c_lat = country_coords[0]
c_lng = country_coords[1]
for _, row in region.iterrows():
    # obtain coordiantes of the current city
    cities.append(row["Largest_city"])
    city_coords = gc.get_city_coords(city=row["Largest_city"])
    # calculate relevant distances of this city from the centre of the country
    distance.append(cc.coords_to_distance(city_coords, (c_lat, c_lng)))
    distances = cc.coords_to_distances(city_coords, (c_lat, c_lng))
    x_distance.append(distances[0])
    y_distance.append(distances[1])
# create and save the ``city_distances`` dataframe as a csv file
city_distances = {
    "City": cities,
    "Distance": distance,
    "X_distance": x_distance,
    "Y_distance": y_distance,
}
city_distances = pd.DataFrame(city_distances)
city_distances.to_csv("results/city_distances.csv", float_format="%.3f")

# generate the sources array of arrays containing x and y distances
sources = []
for _, row in city_distances.iterrows():
    sources.append([row["X_distance"], row["Y_distance"]])

# define problem customers by specifying indices w.r.t. sources array
customers = [0, 8, 11]

# generate region defined by distances between sources & sinks
distances = network.region_builder(
    sources, customers, country="Germany", img_path="icons/DE.png"
)

# specify product set for the value chain
products = [
    "ETICS",
    "compressed_ETICS",
    "pre_concentrate",
    "pyrolysis_oil",
    "styrene",
]

# specify product capacity at sources using population density [tons/day]
ETICS_tons_per_year = 17100  # HBCD-free ETICS in 2017 as per Conversio study
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
demand = [0, 0, 0, 0, s_values.sum() / 20 / len(customers)]

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
facility_capacity = {"OCF": 0.39, "MPF": 41.67, "CPF": 12, "DPF": 30}

# initiate ``Infrastructure`` class based on distances from building network
scenario = optimisation.Infrastructure(
    distances[0], distances[1], distances[2], distances[3], distances[4]
)

# define the value chain
scenario.define_value_chain(
    products, source_capacity, facility_capacity, demand, yield_factor, market_price
)

# create optimisation model of the value chain
scenario.model_value_chain(weight_economic=0.5, weight_environmental=0.5)

# solve the optimisation problem
scenario.model.optimize()

# process optimisation problem results
# NOTE: this function MUST be called before any further post-processing is done
scenario.process_results()

# plot resulting infrastructure
scenario.plot_infrastructure(country="Germany", img_path="icons/DE.png")

# plot resulting product flow
scenario.plot_product_flow(country="Germany", img_path="icons/DE.png", layered=True)

# plot objective function breakdown
scenario.plot_objective_function_breakdown()

# tabulate the product flows within the network
scenario.tabulate_product_flows()

print("Post-processing of results finished!")
