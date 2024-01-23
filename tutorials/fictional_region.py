import os, sys
import random
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.network as network
import src.optimisation as optimisation

# create the /results directory unless it already exists
try:
    os.makedirs("./results")
except FileExistsError:
    pass

# number of sources per side of region
n = 4

# define nodes on which customers are located
customers = [0, 1, 2, 3]

# generate region defined by distances between sources & sinks
distances = network.region_generator(1000, n, customers)

# specify product set for the value chain
products = [
    "ETICS",
    "compressed_ETICS",
    "pre_concentrate",
    "pyrolysis_oil",
    "styrene",
]

# set a seed for the random number generator
seed_value = 333242
random.seed(seed_value)

# specify product capacity at sources [tons/day]
s_list, s_values = [], []
for idx in range(0, n * n):
    s_list.append("S_" + str(idx))
    for jdx in range(0, len(products)):
        if jdx == 0:
            s_values.append(random.triangular(0, 20, 50))
        else:
            s_values.append(0)
s_values = np.array(s_values).reshape((n * n, len(products)))
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
scenario.model_value_chain(weight_economic=1, weight_environmental=0)

# solve the optimisation problem
scenario.model.Params.MIPFocus = 0
scenario.model.Params.timelimit = 10000
scenario.model.presolve()
scenario.model.optimize()

# process optimisation problem results
scenario.process_results()

# plot resulting infrastructure
scenario.plot_infrastructure()

# plot resulting product flow
scenario.plot_product_flow(layered=True)

# plot objective function breakdown
scenario.plot_objective_function_breakdown()

# tabulate the product flows within the network
scenario.tabulate_product_flows()

print("Post-processing of results finished!")
