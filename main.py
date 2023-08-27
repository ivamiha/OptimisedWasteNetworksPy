import src.network as network
import src.optimisation as optimisation
import random
import numpy as np
import pandas as pd

# number of sources per side of region
n = 4

# define nodes on which customers are located
customers = [1, 8, 14]

# generate region defined by distances between sources & sinks
distances = network.region_generator(600, n, customers)

# specify product set for the value chain
products = [
    "ETICS",
    "compressed_ETICS",
    "pre_concentrate",
    "pyrolysis_oil",
    "polystyrene",
]

# specify product capacity at sources [tons]
s_list, s_values = [], []
for idx in range(0, n * n):
    s_list.append("S_" + str(idx))
    for jdx in range(0, len(products)):
        if jdx == 0 or jdx == 1:
            s_values.append(random.triangular(0, 1, 10))
        else:
            s_values.append(0)
s_values = np.array(s_values).reshape((n * n, len(products)))
source_capacity = pd.DataFrame(s_values, columns=products, index=s_list)

# specify demand of the product set [tons]
demand = [0, 0, 0, 0, s_values.sum() / len(customers)]

# specify market price of the product set [euro/ton]
market_price = {
    "ETICS": 0,
    "compressed_ETICS": 0,
    "pre_concentrate": 0,
    "pyrolysis_oil": 0,
    "polystyrene": 1650,
}

# specify yield factor of the different technologies
yield_factor = {
    ("ETICS", "OCF"): 0,
    ("compressed_ETICS", "OCF"): 0.95,
    ("pre_concentrate", "OCF"): 0,
    ("pyrolysis_oil", "OCF"): 0,
    ("polystyrene", "OCF"): 0,
    ("ETICS", "MPF"): 0,
    ("compressed_ETICS", "MPF"): 0,
    ("pre_concentrate", "MPF"): 0.10,
    ("pyrolysis_oil", "MPF"): 0,
    ("polystyrene", "MPF"): 0,
    ("ETICS", "CPF"): 0,
    ("compressed_ETICS", "CPF"): 0,
    ("pre_concentrate", "CPF"): 0,
    ("pyrolysis_oil", "CPF"): 0.69,
    ("polystyrene", "CPF"): 0,
    ("ETICS", "DPF"): 0,
    ("compressed_ETICS", "DPF"): 0,
    ("pre_concentrate", "DPF"): 0,
    ("pyrolysis_oil", "DPF"): 0,
    ("polystyrene", "DPF"): 0.75,
}

# specify maximum facility capacities [tons/day]
facility_capacity = {"OCF": 6, "MPF": 10, "CPF": 12, "DPF": 30}

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
scenario.model.solveConcurrent()

# process optimisation problem results
scenario.process_results()

# plot resulting infrastructure
scenario.plot_resulting_infrastructure()

# plot resulting product flow
scenario.plot_resulting_product_flow()
