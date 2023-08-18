import pyscipopt as scip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
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
    fuel_cons = 0.4  # fuel consumption [lt/km]
    fuel_price = 1.79  # fuel price [euro/lt]
    toll_cost = 0.198  # toll cost [euro/km]
    avg_speed = 60  # average driving speed [km/h]
    driver_wage = 45500  # driver wage [euro/year]
    driver_hours = 2070  # driver working hours [hours/year]
    vehicle_cost = 30000 / (15 * 360 * 90 / 14)  # large roll-off [euro/h]
    vehicle_cost_small = 10000 / (15 * 360 * 90 / 14)  # small roll-off [euro/h]
    vehicle_cost_tanker = 50000 / (15 * 360 * 90 / 14)  # steel tanker [euro/h]
    rho_PU = 0.045  # density of polyurethane [ton/m^3]
    rho_BRIQ = 0.60  # density of briquette [ton/m^3]
    rho_PO = 0.80  # density of pyrolysis oil [ton/m^3]
    rho_ANL = 1.00  # density of aniline [ton/m^3]
    max_time = 100  # maximum transportation between facilities [hours]
    variable_CF = 15  # operational costs of CF [euro/ton]
    variable_RTF = 46  # operational cost of RTF [euro/ton]
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
        D1 (DataFrame): distance matrix for S to CF

        D2 (DataFrame): distance matrix for CF to RTF

        D3 (DataFrame): distance matrix for RTF to CPF

        D4 (DataFrame): distance matrix for CPF to DPF

        D5 (DataFrame): distance matrix for DPF to C

        Notes
        -----
        User should be careful in naming facilities. Identical names cannot be
        used to name different facilities at the same location. For example, a
        source and collection facility in Cologne should have different names:
        S_Cologne and CF_Cologne, for example.
        """

        # initialise instance variables for distance matrices
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4
        self.D5 = D5
        # initialise instance variables for sources, customers & facilities
        self.S = D1.index.tolist()
        self.CF = D1.columns.tolist()
        self.RTF = D3.index.tolist()
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
        # NOTE: previously done as follows, in case encounter downstream probs
        # for p in self.P:
        #    for n in self.C:
        #        key_list_C.append((p, n))

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
        self.TCI_CF = 1.45 * ((self.facility_cap["CF"] / 100) ** 0.6) * (10**6)
        self.TCI_RTF = 2.88 * ((self.facility_cap["RTF"] / 100) ** 0.6) * (10**6)
        self.TCI_CPF = 100 * ((self.facility_cap["CPF"] / 278) ** 0.6) * (10**6)
        self.TCI_DPF = 250 * ((self.facility_cap["DPF"] / 278) ** 0.6) * (10**6)

        # calculate annualized capital investment cost per day [euro/day]
        self.fixed_CF = (
            self.rate * self.TCI_CF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_RTF = (
            self.rate * self.TCI_RTF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_CPF = (
            self.rate * self.TCI_CPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.fixed_DPF = (
            self.rate * self.TCI_DPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )

        # calculate transportation costs [euro/(km*ton)]
        # NOTE: need to use function which actually tests whether load or
        # volume are the limiting factor, not guessing as it currently is
        self.time_penalty = (
            (self.driver_wage / self.driver_hours + self.vehicle_cost_small)
            / self.avg_speed
            / (self.max_volume_small * self.rho_PU)
        )
        # assume volume is the limiting factor
        self.TC_PU = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost)
            / self.avg_speed
        ) / (self.max_volume * self.rho_PU)
        # assume load is limiting factor
        self.TC_BRIQ = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost)
            / self.avg_speed
        ) / self.max_load
        # assume load is limiting factor, using 30 m^3 tanker
        self.TC_PO = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / (self.rho_PO * 30)
        # assume load is limiting factor, using 45 m^3 tanker
        self.TC_ANL = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / (self.rho_ANL * 45)

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
        # product flows from S to CF
        for p in self.P:
            for i in self.S:
                for j in self.CF:
                    # Source > Collection Facility
                    x[p, i, j] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, i, j)
                    )
        # product flows from CF to RTF
        for p in self.P:
            for j in self.CF:
                for k in self.RTF:
                    x[p, j, k] = model.addVar(
                        vtype="CONTINUOUS", lb=0, name="x(%s,%s,%s)" % (p, j, k)
                    )
        # product flows from RTF to CPF
        for p in self.P:
            for k in self.RTF:
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
        # CF installation binary outcomes
        for j in self.CF:
            b[j] = model.addVar(vtype="BINARY", name="b(%s)" % j)
        # RTF installation binary outcomes
        for k in self.RTF:
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
                    scip.quicksum(x[p, i, j] for j in self.CF)
                    == self.source_cap.loc[i, p],
                    name="Conservation(%s,%s)" % (p, i),
                )

        # add constraints for flow conservation at facilities to the model
        for p in self.P:
            for j in self.CF:
                # input PU, output PU
                model.addCons(
                    self.yield_factor[(p, "CF")]
                    * scip.quicksum(x[p, i, j] for i in self.S)
                    == scip.quicksum(x[p, j, k] for k in self.RTF),
                    name="Conservation(%s,%s)" % (p, j),
                )
            for k in self.RTF:
                # input PU, output briquette
                model.addCons(
                    self.yield_factor[(p, "RTF")]
                    * scip.quicksum(x[p, j, k] for j in self.CF for p in self.PP)
                    == scip.quicksum(x[p, k, l] for l in self.CPF),
                    name="Conservation(%s,%s)" % (p, k),
                )
            for l in self.CPF:
                # input briquette, output pyrolysis oil
                model.addCons(
                    self.yield_factor[(p, "CPF")]
                    * scip.quicksum(x[p, k, l] for k in self.RTF for p in self.PP)
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
        for j in self.CF:
            model.addCons(
                scip.quicksum(x[p, i, j] for i in self.S for p in self.P)
                <= b[j] * self.facility_cap["CF"],
                name="Capacity(%s)" % j,
            )
        for k in self.RTF:
            model.addCons(
                scip.quicksum(x[p, j, k] for j in self.CF for p in self.P)
                <= b[k] * self.facility_cap["RTF"],
                name="Capacity(%s)" % k,
            )
        for l in self.CPF:
            model.addCons(
                scip.quicksum(x[p, k, l] for k in self.RTF for p in self.P)
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

        # add driving time constraint for CFs to the model
        # NOTE: this is a new constraint not mentioned in the paper
        for i in self.S:
            for j in self.CF:
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
                scip.quicksum(self.fixed_CF * b[j] for j in self.CF)
                + scip.quicksum(self.fixed_RTF * b[k] for k in self.RTF)
                + scip.quicksum(self.fixed_CPF * b[l] for l in self.CPF)
                + scip.quicksum(self.fixed_DPF * b[m] for m in self.DPF)
            )
            - (
                scip.quicksum(
                    self.variable_CF
                    * scip.quicksum(x[p, i, j] for i in self.S for p in self.P)
                    for j in self.CF
                )
                + scip.quicksum(
                    self.variable_RTF
                    * scip.quicksum(x[p, j, k] for j in self.CF for p in self.P)
                    for k in self.RTF
                )
                + scip.quicksum(
                    self.variable_CPF
                    * scip.quicksum(x[p, k, l] for k in self.RTF for p in self.P)
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
                    2 * self.D1.loc[i, j] * self.time_penalty * x[p, i, j]
                    for i in self.S
                    for j in self.CF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D2.loc[j, k] * self.TC_PU * x[p, j, k]
                    for j in self.CF
                    for k in self.RTF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D3.loc[k, l] * self.TC_BRIQ * x[p, k, l]
                    for k in self.RTF
                    for l in self.CPF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D4.loc[l, m] * self.TC_PO * x[p, l, m]
                    for l in self.CPF
                    for m in self.DPF
                    for p in self.P
                )
                + scip.quicksum(
                    2 * self.D5.loc[m, n] * self.TC_ANL * x[p, m, n]
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

        # process data related to installed CFs
        name_list_CF = []
        print("\n-------------------------------------------------------------")
        print("Collection Facilities")
        for j in self.CF:
            if self.model.getVal(self.b[j]) > 0.5:
                print("{} = {:.2f}".format(j, self.model.getVal(self.b[j])))
                name_list_CF.append(j)
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
                        / self.facility_cap["CF"]
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
        self.name_list_CF = name_list_CF

        # print out installed CFs
        print(
            "Total number of open collection facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[j]) for j in self.CF)
            )
        )
        print("List of open collection facilities:", self.name_list_CF)

        # process data related to installed RTFs
        name_list_RTF = []
        print("\n-------------------------------------------------------------")
        print("Recovery and Treatment Facilities")
        for k in self.RTF:
            if self.model.getVal(self.b[k]) > 0.5:
                print("{} = {:.2f}".format(k, self.model.getVal(self.b[k])))
                name_list_RTF.append(k)
                print(
                    "Total inflow to {} is {:.2f}".format(
                        k,
                        sum(
                            self.model.getVal(self.x[p, j, k])
                            for j in self.CF
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        k,
                        sum(
                            self.model.getVal(self.x[p, j, k])
                            for j in self.CF
                            for p in self.P
                        )
                        / self.facility_cap["RTF"]
                        * 100,
                    )
                )
            for p in self.P:
                for j in self.CF:
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
        self.name_list_RTF = name_list_RTF

        # print installed RTFs
        print(
            "Total number of open recovery and treatment facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[k]) for k in self.RTF)
            )
        )
        print("List of open recovery and treatment facilities:", self.name_list_RTF)

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
                            for k in self.RTF
                            for p in self.P
                        ),
                    )
                )
                print(
                    "Capacity utilization of {} is {:.2f}%".format(
                        l,
                        sum(
                            self.model.getVal(self.x[p, k, l])
                            for k in self.RTF
                            for p in self.P
                        )
                        / self.facility_cap["CPF"]
                        * 100,
                    )
                )
            for p in self.P:
                for k in self.RTF:
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
            "Total number of open chemical processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[l]) for l in self.CPF)
            )
        )
        print("List of open chemical processing facilities:", self.name_list_CPF)

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
            "Total number of open downstream processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[m]) for m in self.DPF)
            )
        )
        print("List of open downstream processing facilities:", self.name_list_DPF)

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
        # transportation costs between S and CF
        self.transportation_cost_1 = sum(
            2
            * self.D1.loc[i, j]
            * self.TC_PU
            * sum(self.model.getVal(self.x[p, i, j]) for p in self.P)
            for i in self.S
            for j in self.CF
        )
        print(
            "The transportation cost from Sources to Collection Facilities is {:.2f} euro/day".format(
                self.transportation_cost_1
            )
        )
        # transportation costs between CF and RTF
        self.transportation_cost_2 = sum(
            2
            * self.D2.loc[j, k]
            * self.TC_PU
            * sum(self.model.getVal(self.x[p, j, k]) for p in self.P)
            for j in self.CF
            for k in self.RTF
        )
        print(
            "The transportation cost from Collection Facilities to Recovery and Treatment Facilities is {:.2f} euro/day".format(
                self.transportation_cost_2
            )
        )
        # transportation cost between RTF and CPF
        self.transportation_cost_3 = sum(
            2
            * self.D3.loc[k, l]
            * self.TC_BRIQ
            * sum(self.model.getVal(self.x[p, k, l]) for p in self.P)
            for k in self.RTF
            for l in self.CPF
        )
        print(
            "The transportation cost from Recovery and Treatment Facilities to Chemical Processing Facilities is {:.2f} euro/day".format(
                self.transportation_cost_3
            )
        )
        # transportation cost between CPF and DPF
        self.transportation_cost_4 = sum(
            2
            * self.D4.loc[l, m]
            * self.TC_PO
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
            * self.TC_ANL
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
        self.capex_CF = sum(
            self.fixed_CF * self.model.getVal(self.b[j]) for j in self.CF
        )
        self.capex_RTF = sum(
            self.fixed_RTF * self.model.getVal(self.b[k]) for k in self.RTF
        )
        self.capex_CPF = sum(
            self.fixed_CPF * self.model.getVal(self.b[l]) for l in self.CPF
        )
        self.capex_DPF = sum(
            self.fixed_DPF * self.model.getVal(self.b[m]) for m in self.DPF
        )
        print("CAPEX of CFs is {:.2f} euro/day".format(self.capex_CF))
        print("CAPEX of RTFs is {:.2f} euro/day".format(self.capex_RTF))
        print("CAPEX of CPFs is {:.2f} euro/day".format(self.capex_CPF))
        print("CAPEX of DPFs is {:.2f} euro/day".format(self.capex_DPF))

        # print OPEX of the value chain per facility
        print("\nOperating Cost (OPEX)")
        self.opex_CF = sum(
            self.variable_CF
            * sum(self.model.getVal(self.x[p, i, j]) for i in self.S for p in self.P)
            for j in self.CF
        )
        self.opex_RTF = sum(
            self.variable_RTF
            * sum(self.model.getVal(self.x[p, j, k]) for j in self.CF for p in self.P)
            for k in self.RTF
        )
        self.opex_CPF = sum(
            self.variable_CPF
            * sum(self.model.getVal(self.x[p, k, l]) for k in self.RTF for p in self.P)
            for l in self.CPF
        )
        self.opex_DPF = sum(
            self.variable_DPF
            * sum(self.model.getVal(self.x[p, l, m]) for l in self.CPF for p in self.P)
            for m in self.DPF
        )
        print("OPEX of CFs is {:.2f} euro/day".format(self.opex_CF))
        print("OPEX of RTFs is {:.2f} euro/day".format(self.opex_RTF))
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
        int_list_CF = name_list_to_int_list(self.name_list_CF)
        int_list_RTF = name_list_to_int_list(self.name_list_RTF)
        int_list_CPF = name_list_to_int_list(self.name_list_CPF)
        int_list_DPF = name_list_to_int_list(self.name_list_DPF)
        # use source row sums to create scatter plot with scaled source size
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x_coordinates, y_coordinates, c="k", s=source_cap_row_sums)
        ax.set_xlabel("Horizontal distance [km]")
        ax.set_ylabel("Vertical distance [km]")
        ax.set_xlim(-max(x_coordinates) * 0.3, max(x_coordinates) * 1.3)
        ax.set_ylim(-max(y_coordinates) * 0.3, max(y_coordinates) * 1.3)
        # chosen offset for annotation from the node
        offset = max(25, self.source_cap.values.max() / 30)
        # annotate nodes where CFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/OCF.png"), zoom=0.03)
        for value in int_list_CF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coordinates[value], y_coordinates[value]),
                xybox=(-offset, 0),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annotate nodes where RTFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/MPF.png"), zoom=0.03)
        for value in int_list_RTF:
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
        fig.savefig("results.pdf", dpi=1200)
        plt.show()

    def plot_resulting_product_flow(self):
        """
        Explain
        """

        print(self.product_flow)


@staticmethod
def name_list_to_int_list(name_list):
    """
    Use regular expressions to convert strings representing names of facilities
    that were installed within ``name_list`` into a list of only the integers
    contained in those same strings.

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
