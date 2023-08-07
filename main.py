import src.network as network
import src.optimisation as optimisation
import random
import numpy as np
import pandas as pd

# number of sources per side of region
n = 5

# generate region defined by distances between sources & sinks
distances = network.region_generator(10, n, [1])

# specify product set for the value chain
products = ["Type1", "Type2", "Briq", "pyrOil", "ANL", "P-TOL"]

# specify product capacity at sources [tons]
s_list, s_values = [], []
for idx in range(0, n * n):
    s_list.append("S_" + str(idx))
    for jdx in range(0, len(products)):
        if jdx == 0 or jdx == 1:
            s_values.append(random.uniform(0, 1))
        else:
            s_values.append(0)
s_values = np.array(s_values).reshape((n * n, len(products)))
source_capacity = pd.DataFrame(s_values, columns=products, index=s_list)

# specify demand of the product set [tons]
demand = [0, 0, 0, 0, 8, 6]

# specify market price of the product set [euro/ton]
market_price = {
    "Type1": 0,
    "Type2": 0,
    "Briq": 0,
    "pyrOil": 0,
    "ANL": 1495,
    "P-TOL": 2400,
}

# specify yield factor of the different technologies
yield_factor = {
    ("Type1", "CF"): 0.95,
    ("Type2", "CF"): 0.95,
    ("Briq", "CF"): 0,
    ("pyrOil", "CF"): 0,
    ("ANL", "CF"): 0,
    ("P-TOL", "CF"): 0,
    ("Type1", "RTF"): 0,
    ("Type2", "RTF"): 0,
    ("Briq", "RTF"): 0.60,
    ("pyrOil", "RTF"): 0,
    ("ANL", "RTF"): 0,
    ("P-TOL", "RTF"): 0,
    ("Type1", "CPF"): 0,
    ("Type2", "CPF"): 0,
    ("Briq", "CPF"): 0,
    ("pyrOil", "CPF"): 0.75,
    ("ANL", "CPF"): 0,
    ("P-TOL", "CPF"): 0,
    ("Type1", "DPF"): 0,
    ("Type2", "DPF"): 0,
    ("Briq", "DPF"): 0,
    ("pyrOil", "DPF"): 0,
    ("ANL", "DPF"): 0.25,
    ("P-TOL", "DPF"): 0.12,
}

# specify maximum facility capacities
# facility_capacity = {"CF": 20, "RTF": 15, "CPF": 35, "DPF": 30}
facility_capacity = {"CF": 2000, "RTF": 1500, "CPF": 3500, "DPF": 3000}

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

# print solution
scenario.getOutput()
