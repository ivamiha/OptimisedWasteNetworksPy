import pyscipopt as scip
import numpy as np


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
    max_time = 1  # maximum transportation between facilities [hours]
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
        # NOTE: this is a new constraint not mentioned in the paper, ...
        for i in self.S:
            for j in self.CF:
                model.addCons(
                    scip.quicksum(x[p, i, j] for p in self.P) * self.D1.loc[i, j]
                    <= (scip.quicksum(x[p, i, j] for p in self.P))
                    * self.avg_speed
                    * self.max_time,
                    name="Travel Time(%s,%s)" % (j, i),
                )

        # Objective function
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
            - (  # scip.quicksum(2*self.D1.loc[i, j] * self.time_penalty * x[p, i, j] for i in self.S for j in self.CF for p in self.P) +
                scip.quicksum(
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
        # (self.D1.loc[i, j]/(self.avg_speed * self.max_time)-1)

        model.data = x, b

        self.x = x
        self.b = b
        self.model = model

    def getOutput(self):
        self.OBJ = self.model.getObjVal()
        print("\nObjective value (Profit) = {:.2f} euro/day".format(self.OBJ))

        namelistCF = []
        print("\nCollection Facilities")
        for j in self.CF:
            if self.model.getVal(self.b[j]) > 0.5:
                print("\n")
                print("{} = {:.2f}".format(j, self.model.getVal(self.b[j])))
                namelistCF.append(j)
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
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                i, j, self.model.getVal(self.x[p, i, j]), p
                            )
                        )
                for k in self.RTF:
                    if self.model.getVal(self.x[p, j, k]) > 0.001:
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                j, k, self.model.getVal(self.x[p, j, k]), p
                            )
                        )

        self.namelistCF = namelistCF

        print(
            "\nTotal number of open collection facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[j]) for j in self.CF)
            )
        )
        print("\nList of open collection facilities:", self.namelistCF)

        namelistRTF = []
        print("\nRecovery and Treatment Facilities")
        for k in self.RTF:
            if self.model.getVal(self.b[k]) > 0.5:
                print("\n")
                print("{} = {:.2f}".format(k, self.model.getVal(self.b[k])))
                namelistRTF.append(k)
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
                        print(
                            "{} to {} = {:.2f} ton/day of {} material".format(
                                j, k, self.model.getVal(self.x[p, j, k]), p
                            )
                        )
                for l in self.CPF:
                    if self.model.getVal(self.x[p, k, l]) > 0.001:
                        print(
                            "{} to {} = {:.2f} ton/day of {} material".format(
                                k, l, self.model.getVal(self.x[p, k, l]), p
                            )
                        )

        self.namelistRTF = namelistRTF

        print(
            "\nTotal number of open recovery and treatment facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[k]) for k in self.RTF)
            )
        )
        print("\nList of open recovery and treatment facilities:", self.namelistRTF)

        namelistCPF = []
        for l in self.CPF:
            if self.model.getVal(self.b[l]) > 0.5:
                print("\n")
                print("{} = {:.2f}".format(l, self.model.getVal(self.b[l])))
                namelistCPF.append(l)
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
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                k, l, self.model.getVal(self.x[p, k, l]), p
                            )
                        )
                for m in self.DPF:
                    if self.model.getVal(self.x[p, l, m]) > 0.001:
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                l, m, self.model.getVal(self.x[p, l, m]), p
                            )
                        )

        self.namelistCPF = namelistCPF

        print(
            "\nTotal number of open chemical processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[l]) for l in self.CPF)
            )
        )
        print("\nList of open chemical processing facilities:", self.namelistCPF)

        namelistDPF = []
        for m in self.DPF:
            if self.model.getVal(self.b[m]) > 0.5:
                print("\n")
                print("{} = {:.2f}".format(m, self.model.getVal(self.b[m])))
                namelistDPF.append(m)
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
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                l, m, self.model.getVal(self.x[p, l, m]), p
                            )
                        )
                for n in self.C:
                    if self.model.getVal(self.x[p, m, n]) > 0.001:
                        print(
                            "{} to {} = {:.2f} ton/day of {}".format(
                                m, n, self.model.getVal(self.x[p, m, n]), p
                            )
                        )

        self.namelistDPF = namelistDPF

        print(
            "\nTotal number of open downstream processing facilities = {:.2f}".format(
                sum(self.model.getVal(self.b[m]) for m in self.DPF)
            )
        )
        print("\nList of open downstream processing facilities:", self.namelistDPF)

        DSAT = {}
        print("\n")
        for p in self.P:
            for n in self.C:
                DSAT[(p, n)] = sum(self.model.getVal(self.x[p, m, n]) for m in self.DPF)
                print(
                    "\nDemand Satisfaction of {} = {:.2f} ton/day {}".format(
                        n, DSAT[(p, n)], p
                    )
                )
        self.DSAT = DSAT

        print("\nObjective Function Breakdown:")

        print("\nRevenue:")
        self.Revenue = sum(
            sum(self.market_price[p] * self.DSAT[(p, n)] for n in self.C)
            for p in self.P
        )
        print("The total revenue is {:.2f} euro/day".format(self.Revenue))

        print("\nLogistics Cost:")
        # self.logCost1 = sum(2 * self.D1.loc[i, j] * self.time_penalty * sum(self.model.getVal(self.x[p, i, j]) for p in self.P) for i in self.S for j in self.CF)
        # print('The transportation cost from Sources to Collection Facilities is {:.2f} euros per day'.format(self.logCost1))

        self.logCost2 = sum(
            2
            * self.D2.loc[j, k]
            * self.TC_PU
            * sum(self.model.getVal(self.x[p, j, k]) for p in self.P)
            for j in self.CF
            for k in self.RTF
        )
        print(
            "The transportation cost from Collection Facilities to Recovery and Treatment Facilities is {:.2f} euro/day".format(
                self.logCost2
            )
        )

        self.logCost3 = sum(
            2
            * self.D3.loc[k, l]
            * self.TC_BRIQ
            * sum(self.model.getVal(self.x[p, k, l]) for p in self.P)
            for k in self.RTF
            for l in self.CPF
        )
        print(
            "The transportation cost from Recovery and Treatment Facilities to Chemical Processing Facilities is {:.2f} euro/day".format(
                self.logCost3
            )
        )

        self.logCost4 = sum(
            2
            * self.D4.loc[l, m]
            * self.TC_PO
            * sum(self.model.getVal(self.x[p, l, m]) for p in self.P)
            for l in self.CPF
            for m in self.DPF
        )
        print(
            "The transportation cost from Chemical Processing Facilities to Downstream Processing Facilities is {:.2f} euro/day".format(
                self.logCost4
            )
        )

        self.logCost5 = sum(
            2
            * self.D5.loc[m, n]
            * self.TC_ANL
            * sum(self.model.getVal(self.x[p, m, n]) for p in self.P)
            for m in self.DPF
            for n in self.C
        )
        print(
            "The transportation cost from Downstream Processing Facilities to Customers is {:.2f} euro/day".format(
                self.logCost5
            )
        )

        print("\nCapital Investment Cost (CAPEX):")
        self.capexCF = sum(
            self.fixed_CF * self.model.getVal(self.b[j]) for j in self.CF
        )
        self.capexRTF = sum(
            self.fixed_RTF * self.model.getVal(self.b[k]) for k in self.RTF
        )
        self.capexCPF = sum(
            self.fixed_CPF * self.model.getVal(self.b[l]) for l in self.CPF
        )
        self.capexDPF = sum(
            self.fixed_DPF * self.model.getVal(self.b[m]) for m in self.DPF
        )
        print(
            "The total CAPEX of Collection Facilities is {:.2f} euro/day".format(
                self.capexCF
            )
        )
        print(
            "The total CAPEX of Recovery and Treatment Facilities is {:.2f} euro/day".format(
                self.capexRTF
            )
        )
        print(
            "The total CAPEX of Chemical Processing Facilities is {:.2f} euro/day".format(
                self.capexCPF
            )
        )
        print(
            "The total CAPEX of Downstream Processing is {:.2f} euro/day".format(
                self.capexDPF
            )
        )

        print("\nOperating Cost (OPEX):")
        self.opexCF = sum(
            self.variable_CF
            * sum(self.model.getVal(self.x[p, i, j]) for i in self.S for p in self.P)
            for j in self.CF
        )
        self.opexRTF = sum(
            self.variable_RTF
            * sum(self.model.getVal(self.x[p, j, k]) for j in self.CF for p in self.P)
            for k in self.RTF
        )
        self.opexCPF = sum(
            self.variable_CPF
            * sum(self.model.getVal(self.x[p, k, l]) for k in self.RTF for p in self.P)
            for l in self.CPF
        )
        self.opexDPF = sum(
            self.variable_DPF
            * sum(self.model.getVal(self.x[p, l, m]) for l in self.CPF for p in self.P)
            for m in self.DPF
        )
        print(
            "The total OPEX of Collection Facilities is {:.2f} euro/day".format(
                self.opexCF
            )
        )
        print(
            "The total OPEX of Recovery and Treatment Facilities is {:.2f} euro/day".format(
                self.opexRTF
            )
        )
        print(
            "The total OPEX of Chemical Processing Facilities is {:.2f} euro/day".format(
                self.opexCPF
            )
        )
        print(
            "The total OPEX of Downstream Processing is {:.2f} euro/day".format(
                self.opexDPF
            )
        )
