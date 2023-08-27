import pyscipopt as scip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
import matplotlib.lines as mlines
import re
import math


class Infrastructure:
    """
    ``Infrastructure class`` defines value-chain-specific parameters and builds
    up the optimisation problem utilising pyscipopt
    """

    # define class variables
    max_load = 20  # maximum load for big roll-off [tons]
    max_volume = 99  # maximum volume for big roll-off [m^3]
    max_load_small = 6  # maximum load for small roll-off [tons]
    max_volume_small = 33  # maximum volume for small roll-off [m^3]
    max_volume_small_tanker = 30  # maximum volume for small steel tanker [m^3]
    max_volume_large_tanker = 45  # maximum volume for large steel tanker [m^3]
    fuel_cons = 0.4  # fuel consumption [lt/km]
    fuel_price = 1.79  # fuel price [euro/lt]
    toll_cost = 0.198  # toll cost [euro/km]
    avg_speed = 60  # average driving speed [km/h]
    driver_wage = 45500  # driver wage [euro/year]
    driver_hours = 2070  # driver working hours [hours/year]
    vehicle_cost = 30000 / (15 * 360 * 90 / 14)  # large roll-off [euro/h]
    vehicle_cost_small = 10000 / (15 * 360 * 90 / 14)  # small roll-off [euro/h]
    vehicle_cost_tanker = 50000 / (15 * 360 * 90 / 14)  # steel tanker [euro/h]
    rho_ETICS = 0.014  # density of ETICS [ton/m^3]
    rho_compressed_ETICS = 0.14  # density of compressed ETICS [ton/m^3]
    rho_pre_concentrate = 0.35  # density of pre-concentrate [ton/m^3]
    rho_pyrolysis_oil = 0.80  # density of pyrolysis oil [ton/m^3]
    rho_polystyrene = 0.910  # density of polystyrene [ton/m^3]
    max_time = 100  # maximum transportation between facilities [hours]
    variable_OCF = 15  # operational costs of OCF [euro/ton]
    variable_MPF = 46  # operational cost of MPF [euro/ton]
    variable_CPF = 500  # operational cost of CPF [euro/ton]
    variable_DPF = 500  # operational cost of DPF [euro/ton]
    period = 10  # loan period [years]
    rate = 0.1

    def __init__(self, D1, D2, D3, D4, D5):
        """
        Initialise ``Infrastructure`` object utilising DataFrames for the
        distance between facilities in the network.

        Parameters
        ----------
        D1 (DataFrame): distance matrix for S to OCF

        D2 (DataFrame): distance matrix for OCF to MPF

        D3 (DataFrame): distance matrix for MPF to CPF

        D4 (DataFrame): distance matrix for CPF to DPF

        D5 (DataFrame): distance matrix for DPF to C

        Notes
        -----
        User should be careful in naming facilities. Identical names cannot be
        used to name different facilities at the same location. For example, a
        source and collection facility in Cologne should have different names:
        S_Cologne and OCF_Cologne, for example.
        """

        # initialise instance variables for distance matrices
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4
        self.D5 = D5
        # initialise instance variables for sources, customers & facilities
        self.S = D1.index.tolist()
        self.OCF = D1.columns.tolist()
        self.MPF = D3.index.tolist()
        self.CPF = D3.columns.tolist()
        self.DPF = D5.index.tolist()
        self.C = D5.columns.tolist()

    def define_value_chain(
        self,
        products,
        source_capacity,
        facility_capacity,
        demand,
        yield_factor,
        market_price,
    ):
        """
        Define the value chain by specifying product- and capacity-specific
        parameters.

        Parameters
        ----------
        products (list): contains names of products in value chain

        source_capacity (DataFrame): contains source capacities [tons]

        facility_capacity (dict): contains facilities as keys and capacities as
        values [tons]

        demand (list): contains demand of all products [tons]

        yield_factor (dict): contains product and facility type as keys and
        corresponding yield factors as values

        market_price (dict): contains products as keys and market price as
        values [euro/ton]
        """

        # initialise instance variables from provided parameters
        self.P = products
        self.PP = products  # P subset used when products transformed @ facility
        self.source_cap = source_capacity
        self.yield_factor = yield_factor
        self.market_price = market_price
        self.facility_cap = facility_capacity

        # generate list with product and customer location pairs
        key_list_C = []
        for n in self.C:
            for p in self.P:
                key_list_C.append((p, n))
        self.key_list_C = key_list_C

        # generate instance dict that uses ``self.key_list_C`` as keys and
        # ``demand`` as values
        D = {}
        idx = 0
        for key in range(0, len(self.key_list_C)):
            D[self.key_list_C[key]] = demand[idx]
            idx += 1
            if idx >= len(demand):
                idx = 0
        self.D = D

        # calculate total capital invesment cost wrt capacity [euro]
        self.TCI_OCF = 1.45 * ((self.facility_cap["OCF"] / 100) ** 0.6) * (10**6)
        self.TCI_MPF = 2.88 * ((self.facility_cap["MPF"] / 100) ** 0.6) * (10**6)
        self.TCI_CPF = 100 * ((self.facility_cap["CPF"] / 278) ** 0.6) * (10**6)
        self.TCI_DPF = 250 * ((self.facility_cap["DPF"] / 278) ** 0.6) * (10**6)

        # calculate annualized capital investment cost per day [euro/day]
        self.fixed_OCF = (
            self.rate * self.TCI_OCF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_MPF = (
            self.rate * self.TCI_MPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_CPF = (
            self.rate * self.TCI_CPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_DPF = (
            self.rate * self.TCI_DPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )

        # calculate transportation costs [euro/(km*ton)]
        # assume volume is the limiting factor for ETICS, large roll-off
        self.TC_ETICS = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost)
            / self.avg_speed
        ) / (self.max_volume * self.rho_ETICS)
        # assume load is limiting factor for compressed ETICS, large roll-off
        self.TC_comp_ETICS = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost)
            / self.avg_speed
        ) / self.max_load
        # assume load is limiting factor for pre-concentrate, large roll-off
        self.TC_pre_concentrate = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / self.max_load
        # assume volume is limiting factor pyrolysis oil, using large tanker
        self.TC_pyrolysis_oil = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / (self.max_volume_large_tanker * self.rho_pyrolysis_oil)
        # assume volume is limiting factor for polystyrene, using large tanker
        self.TC_polystyrene = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost)
            / self.avg_speed
        ) / (self.max_volume_large_tanker * self.rho_polystyrene)

    def model_value_chain(self):
        """
        Create optimisation model for the defined value chain utilising the
        SCIPT solver
        """

        # initialise optimisation problem
        model = scip.Model("value_chain")

        # initialise variables whose scope is this function
        b = {}  # binary variable, represents open/close decisions
        x = {}  # continuous variable, represents material flows

        # add solution variables to the optimisation problem
        # product flows from S to OCF
        for p in self.P:
            for i in self.S:
                for j in self.OCF:
                    # Source > Collection Facility
                    x[p, i, j] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, i, j)
                    )
        # product flows from OCF to MPF
        for p in self.P:
            for j in self.OCF:
                for k in self.MPF:
                    x[p, j, k] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, j, k)
                    )
        # product flows from MPF to CPF
        for p in self.P:
            for k in self.MPF:
                for l in self.CPF:
                    x[p, k, l] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, k, l)
                    )
        # product flows from CPF to DPF
        for p in self.P:
            for l in self.CPF:
                for m in self.DPF:
                    x[p, l, m] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, l, m)
                    )
        # product flows from DPF to C
        for p in self.P:
            for m in self.DPF:
                for n in self.C:
                    x[p, m, n] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, m, n)
                    )
        # OCF installation binary outcomes
        for j in self.OCF:
            b[j] = model.addVar(vtype="BINARY", name="b(%s)" % j)
        # MPF installation binary outcomes
        for k in self.MPF:
            b[k] = model.addVar(vtype="BINARY", name="b(%s)" % k)
        # CPF installation binary outcomes
        for l in self.CPF:
            b[l] = model.addVar(vtype="BINARY", name="b(%s)" % l)
        # DPF installation binary outcomes
        for m in self.DPF:
            b[m] = model.addVar(vtype="BINARY", name="b(%s)" % m)

        # add constraint for flow conservation at sources to the model
        for p in self.P:
            for i in self.S:
                model.addCons(
                    scip.quicksum(x[p, i, j] for j in self.OCF)
                    == self.source_cap.loc[i, p],
                    name="Conservation(%s,%s)" % (p, i),
                )

        # add constraints for flow conservation at facilities to the model
        for p in self.P:
            for j in self.OCF:
                # input PU, output PU
                model.addCons(
                    self.yield_factor[(p, "OCF")]
                    * scip.quicksum(x[p, i, j] for i in self.S)
                    == scip.quicksum(x[p, j, k] for k in self.MPF),
                    name="Conservation(%s,%s)" % (p, j),
                )
            for k in self.MPF:
                # input PU, output briquette
                model.addCons(
                    self.yield_factor[(p, "MPF")]
                    * scip.quicksum(x[p, j, k] for j in self.OCF for p in self.PP)
                    == scip.quicksum(x[p, k, l] for l in self.CPF),
                    name="Conservation(%s,%s)" % (p, k),
                )
            for l in self.CPF:
                # input briquette, output pyrolysis oil
                model.addCons(
                    self.yield_factor[(p, "CPF")]
                    * scip.quicksum(x[p, k, l] for k in self.MPF for p in self.PP)
                    == scip.quicksum(x[p, l, m] for m in self.DPF),
                    name="Conservation(%s,%s)" % (p, l),
                )
            for m in self.DPF:
                # input pyrolysis oil, output aniline & toluidine
                model.addCons(
                    self.yield_factor[(p, "DPF")]
                    * scip.quicksum(x[p, l, m] for l in self.CPF for p in self.PP)
                    == scip.quicksum(x[p, m, n] for n in self.C),
                    name="Conservation(%s,%s)" % (p, m),
                )

        # add capacity constraint for the facilities to the model
        for j in self.OCF:
            model.addCons(
                scip.quicksum(x[p, i, j] for i in self.S for p in self.P)
                <= b[j] * self.facility_cap["OCF"],
                name="Capacity(%s)" % j,
            )
        for k in self.MPF:
            model.addCons(
                scip.quicksum(x[p, j, k] for j in self.OCF for p in self.P)
                <= b[k] * self.facility_cap["MPF"],
                name="Capacity(%s)" % k,
            )
        for l in self.CPF:
            model.addCons(
                scip.quicksum(x[p, k, l] for k in self.MPF for p in self.P)
                <= b[l] * self.facility_cap["CPF"],
                name="Capacity(%s)" % l,
            )
        for m in self.DPF:
            model.addCons(
                scip.quicksum(x[p, l, m] for l in self.CPF for p in self.P)
                <= b[m] * self.facility_cap["DPF"],
                name="Capacity(%s)" % m,
            )

        # add demand satisfaction constraint to the model
        for p in self.P:
            for n in self.C:
                model.addCons(
                    scip.quicksum(x[p, m, n] for m in self.DPF) <= self.D[(p, n)],
                    name="Demand(%s,%s)" % (p, n),
                )

        # add driving time constraint for OCFs to the model
        # NOTE: this is a new constraint not mentioned in the paper
        for i in self.S:
            for j in self.OCF:
                model.addCons(
                    scip.quicksum(x[p, i, j] for p in self.P) * self.D1.loc[i, j]
                    <= (scip.quicksum(x[p, i, j] for p in self.P))
                    * self.avg_speed
                    * self.max_time,
                    name="Travel Time(%s,%s)" % (j, i),
                )

        # add objective function to the model
        model.setObjective(
            scip.quicksum(
                self.market_price[p]
                * scip.quicksum(x[p, m, n] for m in self.DPF for n in self.C)
                for p in self.P
            )
            - (
                scip.quicksum(self.fixed_OCF * b[j] for j in self.OCF)
                + scip.quicksum(self.fixed_MPF * b[k] for k in self.MPF)
                + scip.quicksum(self.fixed_CPF * b[l] for l in self.CPF)
                + scip.quicksum(self.fixed_DPF * b[m] for m in self.DPF)
            )
            - (
                scip.quicksum(
                    self.variable_OCF
                    * scip.quicksum(x[p, i, j] for i in self.S for p in self.P)
                    for j in self.OCF
                )
                + scip.quicksum(
                    self.variable_MPF
                    * scip.quicksum(x[p, j, k] for j in self.OCF for p in self.P)
                    for k in self.MPF
                )
                + scip.quicksum(
                    self.variable_CPF
                    * scip.quicksum(x[p, k, l] for k in self.MPF for p in self.P)
                    for l in self.CPF
                )
                + scip.quicksum(
                    self.variable_DPF
                    * scip.quicksum(x[p, l, m] for l in self.CPF for p in self.P)
                    for m in self.DPF
                )
            )
            - (
                scip.quicksum(
                    2 * self.D1.loc[i, j] * self.TC_ETICS * x[p, i, j]
                    for i in self.S
                    for j in self.OCF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D2.loc[j, k] * self.TC_comp_ETICS * x[p, j, k]
                    for j in self.OCF
                    for k in self.MPF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D3.loc[k, l] * self.TC_pre_concentrate * x[p, k, l]
                    for k in self.MPF
                    for l in self.CPF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D4.loc[l, m] * self.TC_pyrolysis_oil * x[p, l, m]
                    for l in self.CPF
                    for m in self.DPF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D5.loc[m, n] * self.TC_polystyrene * x[p, m, n]
                    for m in self.DPF
                    for n in self.C
                    for p in self.P
                )
            ),
            "maximize",
        )

        model.data = x, b

        self.x = x
        self.b = b
        self.model = model

    def process_results(self):
        """
        Process results for the optimisation problem. This function prints out
        key characteristics of the obtained optimised infrastructure. The
        infrastructure is also plotted for visual inspection.
        """

        # access solution and print value for objective function (profit)
        self.OBJ = self.model.getObjVal()
        print("\n-------------------------------------------------------------")
        print("Objective Value (Net Profit) = {:.2f} euro/day".format(self.OBJ))

        # create empty DataFrame ``product_flow`` to which data regarding the
        # flow of products between facilities will be written
        columns = ["Origin", "Destination", "Product", "Amount"]
        self.product_flow = pd.DataFrame(columns=columns)

        # process data related to installed OCFs
        name_list_OCF = []
        print("\n-------------------------------------------------------------")
        print("Optional Compacting Facilities")
        for j in self.OCF:
            if self.model.getVal(self.b[j]) > 0.5:
                print("{} = {:.2f}".format(j, self.model.getVal(self.b[j])))
                name_list_OCF.append(j)
                print(
                    "Total inflow to {} is {:.2f}".format(
                        j,
                        sum(
                            self.model.getVal(self.x[p, i, j])
                            for i in self.S
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        j,
                        sum(
                            self.model.getVal(self.x[p, i, j])
                            for i in self.S
                            for p in self.P
                        )
                        / self.facility_cap["OCF"]
                        * 100,
                    )
                )
            for p in self.P:
                for i in self.S:
                    if self.model.getVal(self.x[p, i, j]) > 0.001:
                        # append flow data from i to j to DataFrame
                        new_data = {
                            "Origin": i,
                            "Destination": j,
                            "Product": p,
                            "Amount": self.model.getVal(self.x[p, i, j]),
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_OCF = name_list_OCF

        # print out installed OCFs
        print(
            "Total number of optional compacting facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[j]) for j in self.OCF)
            )
        )
        print("List of optional compacting facilities:", self.name_list_OCF)

        # process data related to installed MPFs
        name_list_MPF = []
        print("\n-------------------------------------------------------------")
        print("Mechanical Preprocessing Facilities")
        for k in self.MPF:
            if self.model.getVal(self.b[k]) > 0.5:
                print("{} = {:.2f}".format(k, self.model.getVal(self.b[k])))
                name_list_MPF.append(k)
                print(
                    "Total inflow to {} is {:.2f}".format(
                        k,
                        sum(
                            self.model.getVal(self.x[p, j, k])
                            for j in self.OCF
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        k,
                        sum(
                            self.model.getVal(self.x[p, j, k])
                            for j in self.OCF
                            for p in self.P
                        )
                        / self.facility_cap["MPF"]
                        * 100,
                    )
                )
            for p in self.P:
                for j in self.OCF:
                    if self.model.getVal(self.x[p, j, k]) > 0.001:
                        # append flow data from j to k to DataFrame
                        new_data = {
                            "Origin": j,
                            "Destination": k,
                            "Product": p,
                            "Amount": self.model.getVal(self.x[p, j, k]),
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_MPF = name_list_MPF

        # print installed MPFs
        print(
            "Total number of mechanical preprocessing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[k]) for k in self.MPF)
            )
        )
        print("List of mechanical preprocessing facilities:", self.name_list_MPF)

        # process data related to installed CPFs
        name_list_CPF = []
        print("\n-------------------------------------------------------------")
        print("Chemical Processing Facilities")
        for l in self.CPF:
            if self.model.getVal(self.b[l]) > 0.5:
                print("{} = {:.2f}".format(l, self.model.getVal(self.b[l])))
                name_list_CPF.append(l)
                print(
                    "Total inflow to {} is {:.2f}".format(
                        l,
                        sum(
                            self.model.getVal(self.x[p, k, l])
                            for k in self.MPF
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        l,
                        sum(
                            self.model.getVal(self.x[p, k, l])
                            for k in self.MPF
                            for p in self.P
                        )
                        / self.facility_cap["CPF"]
                        * 100,
                    )
                )
            for p in self.P:
                for k in self.MPF:
                    if self.model.getVal(self.x[p, k, l]) > 0.001:
                        # append flow data from k to l to DataFrame
                        new_data = {
                            "Origin": k,
                            "Destination": l,
                            "Product": p,
                            "Amount": self.model.getVal(self.x[p, k, l]),
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_CPF = name_list_CPF

        # print installed CPFs
        print(
            "Total number of chemical processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[l]) for l in self.CPF)
            )
        )
        print("List of chemical processing facilities:", self.name_list_CPF)

        # process data related to installed DPFs
        name_list_DPF = []
        print("\n-------------------------------------------------------------")
        print("Downstream Processing Facilities")
        for m in self.DPF:
            if self.model.getVal(self.b[m]) > 0.5:
                print("{} = {:.2f}".format(m, self.model.getVal(self.b[m])))
                name_list_DPF.append(m)
                print(
                    "Total inflow to {} is {:.2f}".format(
                        m,
                        sum(
                            self.model.getVal(self.x[p, l, m])
                            for l in self.CPF
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        m,
                        sum(
                            self.model.getVal(self.x[p, l, m])
                            for l in self.CPF
                            for p in self.P
                        )
                        / self.facility_cap["DPF"]
                        * 100,
                    )
                )
            for p in self.P:
                for l in self.CPF:
                    if self.model.getVal(self.x[p, l, m]) > 0.001:
                        # append flow data from l to m to DataFrame
                        new_data = {
                            "Origin": l,
                            "Destination": m,
                            "Product": p,
                            "Amount": self.model.getVal(self.x[p, l, m]),
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )

        for m in self.DPF:
            for p in self.P:
                for n in self.C:
                    if self.model.getVal(self.x[p, m, n]) > 0.001:
                        # append flow data from m to n to DataFrame
                        new_data = {
                            "Origin": m,
                            "Destination": n,
                            "Product": p,
                            "Amount": self.model.getVal(self.x[p, m, n]),
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_DPF = name_list_DPF

        # print installed DPFs
        print(
            "Total number of downstream processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[m]) for m in self.DPF)
            )
        )
        print("List of downstream processing facilities:", self.name_list_DPF)

        # print demand satisfaction for all customers in the value chain
        print("\n-------------------------------------------------------------")
        print("Demand Satisfaction")
        demand_satisfaction = {}
        for p in self.P:
            for n in self.C:
                demand_satisfaction[(p, n)] = sum(
                    self.model.getVal(self.x[p, m, n]) for m in self.DPF
                )
                print(
                    "Demand Satisfaction of {} = {:.2f} ton/day {}".format(
                        n, demand_satisfaction[(p, n)], p
                    )
                )
        self.demand_satisfaction = demand_satisfaction

        # process specific components of the objective function, recall that
        # objective function = revenue - CAPEX - OPEX - transportation costs
        print("\n-------------------------------------------------------------")
        print("Objective Function Breakdown")

        # print value chain revenue
        print("\nRevenue")
        self.Revenue = sum(
            sum(self.market_price[p] * self.demand_satisfaction[(p, n)] for n in self.C)
            for p in self.P
        )
        print("The total revenue is {:.2f} euro/day".format(self.Revenue))

        # print transportation costs of the value chain
        print("\nTransportation Costs")
        # transportation costs between S and OCF
        self.transportation_cost_1 = sum(
            2
            * self.D1.loc[i, j]
            * self.TC_ETICS
            * sum(self.model.getVal(self.x[p, i, j]) for p in self.P)
            for i in self.S
            for j in self.OCF
        )
        print(
            "The transportation cost from Sources to Optional Compacting Facilities is {:.2f} euro/day".format(
                self.transportation_cost_1
            )
        )
        # transportation costs between OCF and MPF
        self.transportation_cost_2 = sum(
            2
            * self.D2.loc[j, k]
            * self.TC_comp_ETICS
            * sum(self.model.getVal(self.x[p, j, k]) for p in self.P)
            for j in self.OCF
            for k in self.MPF
        )
        print(
            "The transportation cost from Optional Compacting Facilities to Mechanical Preprocessing Facilities is {:.2f} euro/day".format(
                self.transportation_cost_2
            )
        )
        # transportation cost between MPF and CPF
        self.transportation_cost_3 = sum(
            2
            * self.D3.loc[k, l]
            * self.TC_pre_concentrate
            * sum(self.model.getVal(self.x[p, k, l]) for p in self.P)
            for k in self.MPF
            for l in self.CPF
        )
        print(
            "The transportation cost from Mechanical Preprocessing Facilities to Chemical Processing Facilities is {:.2f} euro/day".format(
                self.transportation_cost_3
            )
        )
        # transportation cost between CPF and DPF
        self.transportation_cost_4 = sum(
            2
            * self.D4.loc[l, m]
            * self.TC_pyrolysis_oil
            * sum(self.model.getVal(self.x[p, l, m]) for p in self.P)
            for l in self.CPF
            for m in self.DPF
        )
        print(
            "The transportation cost from Chemical Processing Facilities to Downstream Processing Facilities is {:.2f} euro/day".format(
                self.transportation_cost_4
            )
        )
        # transportation cost between DPF and C
        self.transportation_cost_5 = sum(
            2
            * self.D5.loc[m, n]
            * self.TC_polystyrene
            * sum(self.model.getVal(self.x[p, m, n]) for p in self.P)
            for m in self.DPF
            for n in self.C
        )
        print(
            "The transportation cost from Downstream Processing Facilities to Customers is {:.2f} euro/day".format(
                self.transportation_cost_5
            )
        )

        # print CAPEX of the value chain per facility
        print("\nCapital Expenditure (CAPEX)")
        self.capex_OCF = sum(
            self.fixed_OCF * self.model.getVal(self.b[j]) for j in self.OCF
        )
        self.capex_MPF = sum(
            self.fixed_MPF * self.model.getVal(self.b[k]) for k in self.MPF
        )
        self.capex_CPF = sum(
            self.fixed_CPF * self.model.getVal(self.b[l]) for l in self.CPF
        )
        self.capex_DPF = sum(
            self.fixed_DPF * self.model.getVal(self.b[m]) for m in self.DPF
        )
        print("CAPEX of OCFs is {:.2f} euro/day".format(self.capex_OCF))
        print("CAPEX of MPFs is {:.2f} euro/day".format(self.capex_MPF))
        print("CAPEX of CPFs is {:.2f} euro/day".format(self.capex_CPF))
        print("CAPEX of DPFs is {:.2f} euro/day".format(self.capex_DPF))

        # print OPEX of the value chain per facility
        print("\nOperating Cost (OPEX)")
        self.opex_OCF = sum(
            self.variable_OCF
            * sum(self.model.getVal(self.x[p, i, j]) for i in self.S for p in self.P)
            for j in self.OCF
        )
        self.opex_MPF = sum(
            self.variable_MPF
            * sum(self.model.getVal(self.x[p, j, k]) for j in self.OCF for p in self.P)
            for k in self.MPF
        )
        self.opex_CPF = sum(
            self.variable_CPF
            * sum(self.model.getVal(self.x[p, k, l]) for k in self.MPF for p in self.P)
            for l in self.CPF
        )
        self.opex_DPF = sum(
            self.variable_DPF
            * sum(self.model.getVal(self.x[p, l, m]) for l in self.CPF for p in self.P)
            for m in self.DPF
        )
        print("OPEX of OCFs is {:.2f} euro/day".format(self.opex_OCF))
        print("OPEX of MPFs is {:.2f} euro/day".format(self.opex_MPF))
        print("OPEX of CPFs is {:.2f} euro/day".format(self.opex_CPF))
        print("OPEX of DPFs is {:.2f} euro/day".format(self.opex_DPF))

        # save the ``self.product_flow`` DataFrame to a csv file
        self.product_flow.to_csv("product_flows.csv")

    def plot_resulting_infrastructure(self):
        """
        Create plot where the nodes are plotted as a scatter plot with the size
        of the node corresponding to the amount of waste sourced from it. The
        installed facilities (and the presence or lack of customers) is then
        indicated by icons which are used to annotate nodes on the plot.

        Notes
        -----
        This is a good way for visually inspecting and checking smaller
        networks.
        """

        # convert source_cap DataFrame to list containing row sums
        source_cap_row_sums = self.source_cap.sum(axis=1).to_list()
        # extract source coordinates to list
        sources = pd.read_csv("coordinates_sources.csv")
        x_coordinates = sources["xcord"].to_list()
        y_coordinates = sources["ycord"].to_list()
        # extract customer coordinates list
        customers = pd.read_csv("coordinates_customers.csv")
        x_coordinates_c = customers["xcord"].to_list()
        y_coordinates_c = customers["ycord"].to_list()
        # convert name_list of installed facilities into an int_list
        int_list_OCF = name_list_to_int_list(self.name_list_OCF)
        int_list_MPF = name_list_to_int_list(self.name_list_MPF)
        int_list_CPF = name_list_to_int_list(self.name_list_CPF)
        int_list_DPF = name_list_to_int_list(self.name_list_DPF)
        # use source row sums to create scatter plot with scaled source size
        fig, ax = plt.subplots(figsize=(8, 8))
        factor = int(300 / max(source_cap_row_sums))
        size = [factor * val for val in source_cap_row_sums]
        ax.scatter(x_coordinates, y_coordinates, c="k", s=size)
        ax.set_xlabel("Horizontal distance [km]")
        ax.set_ylabel("Vertical distance [km]")
        ax.set_xlim(-max(x_coordinates) * 0.3, max(x_coordinates) * 1.3)
        ax.set_ylim(-max(y_coordinates) * 0.3, max(y_coordinates) * 1.3)
        # chosen offset for annotation from the node
        offset = max(25, self.source_cap.values.max() / 30)
        # annotate nodes where OCFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/OCF.png"), zoom=0.03)
        for value in int_list_OCF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates[value], y_coordinates[value]),
                xybox=(-offset, 0),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annotate nodes where MPFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/MPF.png"), zoom=0.03)
        for value in int_list_MPF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates[value], y_coordinates[value]),
                xybox=(-offset * math.sqrt(2) / 2, offset * math.sqrt(2) / 2),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annnotate nodes where CPFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/CPF.png"), zoom=0.03)
        for value in int_list_CPF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates[value], y_coordinates[value]),
                xybox=(0, offset),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annotate nodes where DPFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/DPF.png"), zoom=0.03)
        for value in int_list_DPF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates[value], y_coordinates[value]),
                xybox=(offset * math.sqrt(2) / 2, offset * math.sqrt(2) / 2),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annotate nodes where customers are located
        imagebox = osb.OffsetImage(plt.imread("icons/C.png"), zoom=0.03)
        for idx in range(0, len(x_coordinates_c)):
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates_c[idx], y_coordinates_c[idx]),
                xybox=(offset, 0),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # plot and save the figure
        fig.savefig("results_infrastructure.pdf", dpi=1200)
        plt.show()

    def plot_resulting_product_flow(self):
        """
        Create a plot where product flows are plotted between nodes represented
        by a scatter plot. The nodes of the scatter plot are scaled according to
        the amount of waste available at them. The amount of product exchanged
        between nodes is represented by the width of the line connecting them.
        The lines and nodes are colour and shape coded according to the
        facilities installed at the node and the type of material being
        transported.

        Notes
        -----
        This is a good way for visually inspecting and checking larger networks.
        """

        # define colour scheme used throughout using hex notation
        # colours correspond to: yellow, orange, red, purple, indigo
        colours = ["#fff3ad", "#f7ac78", "#dc636e", "#9e2a7d", "#002287"]

        # convert source_cap DataFrame to list containing row sums
        source_cap_row_sums = self.source_cap.sum(axis=1).to_list()
        # extract source coordinates to list
        sources = pd.read_csv("coordinates_sources.csv")
        x_coordinates = sources["xcord"].to_list()
        y_coordinates = sources["ycord"].to_list()
        # create the figure and define axis labels and limits
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("Horizontal distance [km]")
        ax.set_ylabel("Vertical distance [km]")
        ax.set_xlim(-max(x_coordinates) * 0.2, max(x_coordinates) * 1.2)
        ax.set_ylim(-max(y_coordinates) * 0.2, max(y_coordinates) * 1.4)
        # draw lines representing exchanged products
        product_flows = pd.read_csv("product_flows.csv")
        for _, row in product_flows.iterrows():
            origin_int = name_to_int(row["Origin"])
            destination_int = name_to_int(row["Destination"])
            # draw line only if origin and destination have different node num.
            if origin_int != destination_int:
                # plot line with selected product flow colour
                if row["Product"] == "ETICS":
                    colour = colours[0]
                elif row["Product"] == "compressed_ETICS":
                    colour = colours[1]
                elif row["Product"] == "pre_concentrate":
                    colour = colours[2]
                elif row["Product"] == "pyrolysis_oil":
                    colour = colours[3]
                else:
                    colour = colours[4]
                # draw the corresponding product flow line
                x = (x_coordinates[origin_int], x_coordinates[destination_int])
                y = (y_coordinates[origin_int], y_coordinates[destination_int])
                plt.plot(x, y, c=colour)
        # convert name_list of installed facilities into an int_list
        int_list_OCF = name_list_to_int_list(self.name_list_OCF)
        int_list_MPF = name_list_to_int_list(self.name_list_MPF)
        int_list_CPF = name_list_to_int_list(self.name_list_CPF)
        int_list_DPF = name_list_to_int_list(self.name_list_DPF)
        # loop over the source coordinates (that is, number of nodes)
        factor = int(300 / max(source_cap_row_sums))
        for idx in range(0, len(x_coordinates)):
            # do not plot if the node has no facilities installed
            if (
                idx not in int_list_DPF
                and idx not in int_list_CPF
                and idx not in int_list_MPF
                and idx not in int_list_OCF
            ):
                break
            # otherwise, plot node with selected marker colour
            else:
                if idx in int_list_DPF:
                    colour = colours[4]
                elif idx in int_list_CPF:
                    colour = colours[3]
                elif idx in int_list_MPF:
                    colour = colours[2]
                elif idx in int_list_OCF:
                    colour = colours[1]
                # plot the node
                plt.scatter(
                    x_coordinates[idx],
                    y_coordinates[idx],
                    c=colour,
                    s=source_cap_row_sums[idx] * factor,
                    zorder=2.5,
                )
        # create manual symbols for legend
        point_S = mlines.Line2D(
            [0],
            [0],
            label="only S",
            marker="o",
            markersize=10,
            markeredgecolor=colours[0],
            markerfacecolor=colours[0],
            linestyle="",
        )
        point_OCF = mlines.Line2D(
            [0],
            [0],
            label="at least OCF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[1],
            markerfacecolor=colours[1],
            linestyle="",
        )
        point_MPF = mlines.Line2D(
            [0],
            [0],
            label="at least MPF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[2],
            markerfacecolor=colours[2],
            linestyle="",
        )
        point_CPF = mlines.Line2D(
            [0],
            [0],
            label="at least CPF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[3],
            markerfacecolor=colours[3],
            linestyle="",
        )
        point_DPF = mlines.Line2D(
            [0],
            [0],
            label="at least DPF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[4],
            markerfacecolor=colours[4],
            linestyle="",
        )
        line_ET = mlines.Line2D(
            [0],
            [0],
            label="ETICS flow",
            c=colours[0],
        )
        line_CE = mlines.Line2D(
            [0],
            [0],
            label="compressed ETICS flow",
            c=colours[1],
        )
        line_PC = mlines.Line2D(
            [0],
            [0],
            label="pre-concentrate flow",
            c=colours[2],
        )
        line_PO = mlines.Line2D(
            [0],
            [0],
            label="pyrolysis oil flow",
            c=colours[3],
        )
        line_PS = mlines.Line2D(
            [0],
            [0],
            label="polystyrene flow",
            c=colours[4],
        )
        plt.legend(
            handles=[
                point_S,
                point_OCF,
                point_MPF,
                point_CPF,
                point_DPF,
                line_ET,
                line_CE,
                line_PC,
                line_PO,
                line_PS,
            ],
            loc="upper right",
            ncol=2,
            frameon=False,
        )
        fig.savefig("results_product_flow.pdf", dpi=1200)
        plt.show()


@staticmethod
def name_list_to_int_list(name_list):
    """
    Use regular expressions to extract list of integers from a list of strings
    containing names of facilities that were installed, contained within
    ``name_list``.

    Parameters
    ----------
    name_list (list): list of strings representing installed facilities

    Returns
    -------
    int_list (list): list containing only integers in the provided strings
    """

    # use regular expressions to extract only the numbers in the strings
    int_list = []
    for name in name_list:
        integer = int(re.search(r"\d+", name).group())
        int_list.append(integer)

    return int_list


@staticmethod
def name_to_int(name):
    """
    Use regular expressions to extract an integer from a string representing the
    name of an installed facility contained within ``name``.

    Parameters
    ----------
    name (string): string containing name of installed facility

    Returns
    -------
    integer (int): integer extracted from string representing facility name
    """

    integer = int(re.search(r"\d+", name).group())

    return integer
