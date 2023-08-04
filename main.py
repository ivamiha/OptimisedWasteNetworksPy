import src.network as network
import random
import numpy as np
import pandas as pd

# number of sources per side of region
n = 5

# generate region defined by distances between sources & sinks
distances = network.region_generator(100, n, [1, 23])

# specify product set for the value chain
products = ["Type1", "Type2", "Briq", "pyrOil", "ANL", "P-TOL"]

# specify product capacity at sources [kilotons]
s_list, s_values = [], []
for idx in range(0, n * n):
    s_list.append("S_" + str(idx))
    for jdx in range(0, len(products)):
        if jdx == 0 or jdx == 1:
            s_values.append(random.uniform(0, 10))
        else:
            s_values.append(0)
s_values = np.array(s_values).reshape((n * n, len(products)))
source_capacity = pd.DataFrame(s_values, columns=products, index=s_list)

# specify demand of the product set [kilotons]
demand = [0, 0, 0, 0, 8, 6]

# specify market price of the product set [EUR/kiloton]
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

# specify maximum capacity of the different technologies
capacity = {"CF": 20, "RTF": 15, "CPF": 35, "DPF": 30}
