import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
import matplotlib.lines as mlines
import utils.get_coords as gc
import utils.convert_coords as cc
import re
import math
import numpy as np
import seaborn as sns
import subprocess
import sys


class Infrastructure:
    """
    ``Infrastructure class`` defines value-chain-specific parameters and builds
    up the optimisation problem utilising pyscipopt
    """

    # define miscellaneous simulation variables
    env_cost = 170  # cost of the environmental impact [euro/ton CO2e]
    circuit_factor = 1.32  # average circuit factor for Germany []
    working_days = 330  # assume operation 330 day/year (7'920h total) [days]
    # define transportation variables
    rolloff_load = 6  # maximum load for a roll-off [tons]
    rolloff_volume = 33  # maximum volume for a roll-off [m^3]
    tanker_volume = 45  # maximum volume for steel tanker [m^3]
    fuel_cons = 0.4  # fuel consumption [lt/km]
    fuel_price = 1.79  # fuel price [euro/lt]
    toll_cost = 0.198  # toll cost [euro/km]
    avg_speed = 60  # average driving speed [km/h]
    driver_wage = 45500  # driver wage [euro/year]
    driver_hours = 2070  # driver working hours [hours/year]
    vehicle_cost_rollof = 10000 / (15 * 360 * 90 / 14)  # roll-off [euro/h]
    vehicle_cost_tanker = 50000 / (15 * 360 * 90 / 14)  # steel tanker [euro/h]
    max_time = 100  # maximum transportation between facilities [hours]
    # define product physical variables
    rho_ETICS = 0.014  # density of ETICS [ton/m^3]
    rho_compressed_ETICS = 0.14  # density of compressed ETICS [ton/m^3]
    rho_pre_concentrate = 0.35  # density of pre-concentrate [ton/m^3]
    rho_pyrolysis_oil = 0.80  # density of pyrolysis oil [ton/m^3]
    rho_styrene = 0.910  # density of styrene [ton/m^3]
    # define economic variables
    ref_CAPEX_OCF = 0.057  # CAPEX of OCF for reference capacity [Meuro]
    ref_CAPEX_MPF = 8  # CAPEX of MPF for reference capacity [Meuro]
    ref_CAPEX_CPF = 20.2  # CAPEX of CPF for reference capacity [Meuro]
    ref_CAPEX_DPF = 0.9  # CAPEX of DPF for reference capacity [Meuro]
    ref_capacity_OCF = 140 / working_days  # OCF reference capacity [ton/day]
    ref_capacity_MPF = 15000 / working_days  # MPF reference capacity [ton/day]
    ref_capacity_CPF = 40000 / working_days  # CPF reference capacity [ton/day]
    ref_capacity_DPF = 33000 / working_days  # DPF reference capacity [ton/day]
    ref_fOPEX_OCF = 0.012  # reference fixed OPEX of OCF [Meuro/year]
    ref_fOPEX_MPF = 1.1  # reference fixed OPEX of MPF [Meuro/year]
    ref_fOPEX_CPF = 1.7  # reference fixed OPEX of CPF [Meuro/year]
    ref_fOPEX_DPF = 4.7  # reference fixed OPEX of DPF [Meuro/year]
    vOPEX_OCF = 11  # variable OPEX of OCF [euro/ton]
    vOPEX_MPF = 46  # variable OPEX of MPF [euro/ton]
    vOPEX_CPF = 44  # variable OPEX of CPF [euro/ton]
    vOPEX_DPF = 97  # variable OPEX of DPF [euro/ton]
    period = 10  # loan period [years]
    rate = 0.10  # discount rate []
    # define environmental impact variables
    env_rolloff = 5.6402e-4  # environmental impact of roll-off [tons CO2e/km]
    env_tanker = 1.0586e-4  # environmental impact of tanker [tons CO2e/km]
    CI_OCF = 4.463e-03  # construction impact of mech. plant [tons CO2e/ton]
    CI_MPF = 4.463e-03  # construction impact of mech. plant [tons CO2e/ton]
    CI_CPF = 0.651  # construction impact of chem. plant [tons CO2e/ton]
    CI_DPF = 2.813e-02  # construction impact of chem. plant [tons CO2e/ton]
    OI_OCF = 1.250e-2  # operational impact of OCF [tons CO2e/ton]
    OI_MPF = 3.575e-2  # operational impact of MPF [tons CO2e/ton]
    OI_CPF = 1.100  # operational impact of CPF [tons CO2e/ton]
    OI_DPF = 2.055  # operational impact of DPF [tons CO2e/ton]

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

        # initialise instance variables for distance matrices NOTE: multiplied
        # by circuit factor, only needed if use straight-line distance as is
        # currently being used
        self.D1 = D1 * self.circuit_factor
        self.D2 = D2 * self.circuit_factor
        self.D3 = D3 * self.circuit_factor
        self.D4 = D4 * self.circuit_factor
        self.D5 = D5 * self.circuit_factor
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
        self.TCI_OCF = (
            six_tenths_rule(
                self.ref_capacity_OCF, self.facility_cap["OCF"], self.ref_CAPEX_OCF
            )
            * 10**6
        )
        self.TCI_MPF = (
            six_tenths_rule(
                self.ref_capacity_MPF, self.facility_cap["MPF"], self.ref_CAPEX_MPF
            )
            * 10**6
        )
        self.TCI_CPF = (
            six_tenths_rule(
                self.ref_capacity_CPF, self.facility_cap["CPF"], self.ref_CAPEX_CPF
            )
            * 10**6
        )
        self.TCI_DPF = (
            six_tenths_rule(
                self.ref_capacity_DPF, self.facility_cap["DPF"], self.ref_CAPEX_DPF
            )
            * 10**6
        )

        # calculate fixed OPEX wrt capacity [euro/day]
        self.fOPEX_OCF = (
            six_tenths_rule(
                self.ref_capacity_OCF, self.facility_cap["OCF"], self.ref_fOPEX_OCF
            )
            * 10**6
            / self.working_days
        )
        self.fOPEX_MPF = (
            six_tenths_rule(
                self.ref_capacity_MPF, self.facility_cap["MPF"], self.ref_fOPEX_MPF
            )
            * 10**6
            / self.working_days
        )
        self.fOPEX_CPF = (
            six_tenths_rule(
                self.ref_capacity_CPF, self.facility_cap["CPF"], self.ref_fOPEX_CPF
            )
            * 10**6
            / self.working_days
        )
        self.fOPEX_DPF = (
            six_tenths_rule(
                self.ref_capacity_DPF, self.facility_cap["DPF"], self.ref_fOPEX_DPF
            )
            * 10**6
            / self.working_days
        )

        # calculate annualized capital investment cost per day [euro/day]
        self.ACI_OCF = (
            self.rate * self.TCI_OCF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.ACI_MPF = (
            self.rate * self.TCI_MPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.ACI_CPF = (
            self.rate * self.TCI_CPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )
        self.ACI_DPF = (
            self.rate * self.TCI_DPF / (1 - (1 + self.rate) ** (-self.period)) / 360
        )

        # calculate transportation costs [euro/(km*ton)]
        # ETICS transported in roll-off, determine if volume- or load-limit
        self.TC_ETICS = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_rollof)
            / self.avg_speed
        ) / min(self.rolloff_load, self.rolloff_volume * self.rho_ETICS)
        # compressed ETICS transported in roll-off, determine if v- or l-limit
        self.TC_comp_ETICS = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_rollof)
            / self.avg_speed
        ) / min(self.rolloff_load, self.rolloff_volume * self.rho_compressed_ETICS)
        # pre-concentrate transported in roll-off, determine if v- or l-limit
        self.TC_pre_concentrate = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_rollof)
            / self.avg_speed
        ) / min(self.rolloff_load, self.rolloff_volume * self.rho_pre_concentrate)
        # pyrolysis oil transported in tanker, volume-limited
        self.TC_pyrolysis_oil = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / (self.tanker_volume * self.rho_pyrolysis_oil)
        # styrene transported in tanker, volume-limited
        self.TC_styrene = (
            (self.fuel_price * self.fuel_cons + self.toll_cost)
            + (self.driver_wage / self.driver_hours + self.vehicle_cost_tanker)
            / self.avg_speed
        ) / (self.tanker_volume * self.rho_styrene)

        # calculate transportation environmental impact [tons CO2-eq/(km*ton)]
        # ETICS, compressed ETICS and pre-concentrate transported in roll-off
        self.TI_ETICS = self.env_rolloff
        self.TI_comp_ETICS = self.env_rolloff
        self.TI_pre_concentrate = self.env_rolloff
        # pyrolysis oil and styrene transported in tanker
        self.TI_pyrolysis_oil = self.env_tanker
        self.TI_styrene = self.env_tanker

    def model_value_chain(self, weight_economic, weight_environmental):
        """
        Utilise gurobipy to define a multi-objective optimisation model for the
        user-defined value chain. The model is defined using gurobipy's blended
        multi-bojective functions. See note below for more information regarding
        how the weights work.

        Parameters
        ----------
        weight_economic (int): weight corresponding to the economic objective
        function

        weight_environmental (int): weight corresponding to the environmental
        objective function

        NOTE: the resulting multi-objective function will be a linear
        combination of the individual objective functions multiplied by their
        corresponding weights, i.e.: weight_economic * obj_economic +
        weight_environmental * obj_environmental
        """

        # initialise optimisation problem
        model = gp.Model("value_chain")

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
            b[j] = model.addVar(vtype="INTEGER", lb=0, name="b(%s)" % j)
        # MPF installation binary outcomes
        for k in self.MPF:
            b[k] = model.addVar(vtype="INTEGER", lb=0, name="b(%s)" % k)
        # CPF installation binary outcomes
        for l in self.CPF:
            b[l] = model.addVar(vtype="INTEGER", lb=0, name="b(%s)" % l)
        # DPF installation binary outcomes
        for m in self.DPF:
            b[m] = model.addVar(vtype="INTEGER", lb=0, name="b(%s)" % m)

        # add constraint for flow conservation at sources to the model
        for p in self.P:
            for i in self.S:
                model.addConstr(
                    gp.quicksum(x[p, i, j] for j in self.OCF)
                    == self.source_cap.loc[i, p],
                    name="Conservation(%s,%s)" % (p, i),
                )

        # add constraints for flow conservation at facilities to the model
        for p in self.P:
            for j in self.OCF:
                # input ETICS, output compressed ETICS
                model.addConstr(
                    self.yield_factor[(p, "OCF")]
                    * gp.quicksum(x[p, i, j] for i in self.S for p in self.PP)
                    == gp.quicksum(x[p, j, k] for k in self.MPF),
                    name="Conservation(%s,%s)" % (p, j),
                )
            for k in self.MPF:
                # input compressed ETICS, output pre-concentrate
                model.addConstr(
                    self.yield_factor[(p, "MPF")]
                    * gp.quicksum(x[p, j, k] for j in self.OCF for p in self.PP)
                    == gp.quicksum(x[p, k, l] for l in self.CPF),
                    name="Conservation(%s,%s)" % (p, k),
                )
            for l in self.CPF:
                # input pre-concentrate, output pyrolysis oil
                model.addConstr(
                    self.yield_factor[(p, "CPF")]
                    * gp.quicksum(x[p, k, l] for k in self.MPF for p in self.PP)
                    == gp.quicksum(x[p, l, m] for m in self.DPF),
                    name="Conservation(%s,%s)" % (p, l),
                )
            for m in self.DPF:
                # input pyrolysis oil, output styrene
                model.addConstr(
                    self.yield_factor[(p, "DPF")]
                    * gp.quicksum(x[p, l, m] for l in self.CPF for p in self.PP)
                    == gp.quicksum(x[p, m, n] for n in self.C),
                    name="Conservation(%s,%s)" % (p, m),
                )

        # add capacity constraint for the facilities to the model
        for j in self.OCF:
            model.addConstr(
                gp.quicksum(x[p, i, j] for i in self.S for p in self.P)
                <= b[j] * self.facility_cap["OCF"],
                name="Capacity(%s)" % j,
            )
        for k in self.MPF:
            model.addConstr(
                gp.quicksum(x[p, j, k] for j in self.OCF for p in self.P)
                <= b[k] * self.facility_cap["MPF"],
                name="Capacity(%s)" % k,
            )
        for l in self.CPF:
            model.addConstr(
                gp.quicksum(x[p, k, l] for k in self.MPF for p in self.P)
                <= b[l] * self.facility_cap["CPF"],
                name="Capacity(%s)" % l,
            )
        for m in self.DPF:
            model.addConstr(
                gp.quicksum(x[p, l, m] for l in self.CPF for p in self.P)
                <= b[m] * self.facility_cap["DPF"],
                name="Capacity(%s)" % m,
            )

        # add demand satisfaction constraint to the model
        for p in self.P:
            for n in self.C:
                model.addConstr(
                    gp.quicksum(x[p, m, n] for m in self.DPF) <= self.D[(p, n)],
                    name="Demand(%s,%s)" % (p, n),
                )

        # add driving time constraint for OCFs to the model
        for i in self.S:
            for j in self.OCF:
                model.addConstr(
                    gp.quicksum(x[p, i, j] for p in self.P) * self.D1.loc[i, j]
                    <= (gp.quicksum(x[p, i, j] for p in self.P))
                    * self.avg_speed
                    * self.max_time,
                    name="Travel Time(%s,%s)" % (j, i),
                )

        # set Gurobi to maximize all objective functions
        model.ModelSense = gp.GRB.MAXIMIZE

        # add economic objective to the model (maximize)
        model.setObjectiveN(
            gp.quicksum(
                self.market_price[p]
                * gp.quicksum(x[p, m, n] for m in self.DPF for n in self.C)
                for p in self.P
            )
            - (
                gp.quicksum(self.ACI_OCF * b[j] for j in self.OCF)
                + gp.quicksum(self.ACI_MPF * b[k] for k in self.MPF)
                + gp.quicksum(self.ACI_CPF * b[l] for l in self.CPF)
                + gp.quicksum(self.ACI_DPF * b[m] for m in self.DPF)
            )
            - (
                gp.quicksum(
                    (
                        self.fOPEX_OCF * b[j]
                        + self.vOPEX_OCF
                        * gp.quicksum(x[p, i, j] for i in self.S for p in self.P)
                    )
                    for j in self.OCF
                )
                + gp.quicksum(
                    (
                        self.fOPEX_MPF * b[k]
                        + self.vOPEX_MPF
                        * gp.quicksum(x[p, j, k] for j in self.OCF for p in self.P)
                    )
                    for k in self.MPF
                )
                + gp.quicksum(
                    (
                        self.fOPEX_CPF * b[l]
                        + self.vOPEX_CPF
                        * gp.quicksum(x[p, k, l] for k in self.MPF for p in self.P)
                    )
                    for l in self.CPF
                )
                + gp.quicksum(
                    (
                        self.fOPEX_DPF * b[m]
                        + self.vOPEX_DPF
                        * gp.quicksum(x[p, l, m] for l in self.CPF for p in self.P)
                    )
                    for m in self.DPF
                )
            )
            - (
                gp.quicksum(
                    2 * self.D1.loc[i, j] * self.TC_ETICS * x[p, i, j]
                    for i in self.S
                    for j in self.OCF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D2.loc[j, k] * self.TC_comp_ETICS * x[p, j, k]
                    for j in self.OCF
                    for k in self.MPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D3.loc[k, l] * self.TC_pre_concentrate * x[p, k, l]
                    for k in self.MPF
                    for l in self.CPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D4.loc[l, m] * self.TC_pyrolysis_oil * x[p, l, m]
                    for l in self.CPF
                    for m in self.DPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D5.loc[m, n] * self.TC_styrene * x[p, m, n]
                    for m in self.DPF
                    for n in self.C
                    for p in self.P
                )
            ),
            0,
            weight=weight_economic,
            name="obj_economic",
        )

        # add environmental objective to the model (minimize = -1 * maximize)
        # NOTE: convert to estimated economic cost so blended MOO works
        model.setObjectiveN(
            -1
            * self.env_cost
            * (
                gp.quicksum(
                    (self.CI_OCF + self.OI_OCF)
                    * gp.quicksum(x[p, i, j] for i in self.S for p in self.P)
                    for j in self.OCF
                )
                + gp.quicksum(
                    (self.CI_MPF + self.OI_MPF)
                    * gp.quicksum(x[p, j, k] for j in self.OCF for p in self.P)
                    for k in self.MPF
                )
                + gp.quicksum(
                    (self.CI_CPF + self.OI_CPF)
                    * gp.quicksum(x[p, k, l] for k in self.MPF for p in self.P)
                    for l in self.CPF
                )
                + gp.quicksum(
                    (self.CI_DPF + self.OI_DPF)
                    * gp.quicksum(x[p, l, m] for l in self.CPF for p in self.P)
                    for m in self.DPF
                )
                + gp.quicksum(
                    2 * self.D1.loc[i, j] * self.TI_ETICS * x[p, i, j]
                    for i in self.S
                    for j in self.OCF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D2.loc[j, k] * self.TI_comp_ETICS * x[p, j, k]
                    for j in self.OCF
                    for k in self.MPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D3.loc[k, l] * self.TI_pre_concentrate * x[p, k, l]
                    for k in self.MPF
                    for l in self.CPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D4.loc[l, m] * self.TI_pyrolysis_oil * x[p, l, m]
                    for l in self.CPF
                    for m in self.DPF
                    for p in self.P
                )
                + gp.quicksum(
                    2 * self.D5.loc[m, n] * self.TI_styrene * x[p, m, n]
                    for m in self.DPF
                    for n in self.C
                    for p in self.P
                )
            ),
            1,
            weight=weight_environmental,
            name="obj_environmental",
        )

        self.x = x
        self.b = b
        self.model = model

    def process_results(self):
        """
        Process results for the optimised solution. This function accesses
        solution variables and computes other relevant secondary variables.
        NOTE: this function MUST be called if any further post-processing of the
        results is to be done.
        """

        # extract resulting variable values and store them in a dictionary
        vars = {}
        for var in self.model.getVars():
            vars[f"{var.varName}"] = var.x

        # access and store both objective functions (profit & environmental
        # impact expressed as an economic cost)
        self.obj_economic = self.model.getObjective(0).getValue()
        self.obj_environmental = -1 * self.model.getObjective(1).getValue()

        # create ``product_flow`` df for products flowing between facilities
        columns = ["Origin", "Destination", "Product", "Amount"]
        self.product_flow = pd.DataFrame(columns=columns)

        # calculate and store the demand satisfaction
        demand_satisfaction = {}
        for p in self.P:
            for n in self.C:
                demand_satisfaction[(p, n)] = sum(
                    vars[f"x({p},{m},{n})"] for m in self.DPF
                )
        self.demand_satisfaction = demand_satisfaction

        # calculate and store the revenue
        self.Revenue = sum(
            sum(self.market_price[p] * self.demand_satisfaction[(p, n)] for n in self.C)
            for p in self.P
        )

        # process data related to installed OCFs
        name_list_OCF = []
        for j in self.OCF:
            if vars[f"b({j})"] > 0.5:
                name_list_OCF.append(j)
            # add inflowing products to product_flow
            for p in self.P:
                for i in self.S:
                    if vars[f"x({p},{i},{j})"] > 0.001:
                        # append flow data from i to j to DataFrame
                        new_data = {
                            "Origin": i,
                            "Destination": j,
                            "Product": p,
                            "Amount": vars[f"x({p},{i},{j})"],
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_OCF = name_list_OCF

        # process data related to installed MPFs
        name_list_MPF = []
        for k in self.MPF:
            if vars[f"b({k})"] > 0.5:
                name_list_MPF.append(k)
            # add inflowing products to product_flow
            for p in self.P:
                for j in self.OCF:
                    if vars[f"x({p},{j},{k})"] > 0.001:
                        # append flow data from j to k to DataFrame
                        new_data = {
                            "Origin": j,
                            "Destination": k,
                            "Product": p,
                            "Amount": vars[f"x({p},{j},{k})"],
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_MPF = name_list_MPF

        # process data related to installed CPFs
        name_list_CPF = []
        for l in self.CPF:
            if vars[f"b({l})"] > 0.5:
                name_list_CPF.append(l)
            # add inflowing products to product_flow
            for p in self.P:
                for k in self.MPF:
                    if vars[f"x({p},{k},{l})"] > 0.001:
                        # append flow data from k to l to DataFrame
                        new_data = {
                            "Origin": k,
                            "Destination": l,
                            "Product": p,
                            "Amount": vars[f"x({p},{k},{l})"],
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_CPF = name_list_CPF

        # process data related to installed DPFs
        name_list_DPF = []
        for m in self.DPF:
            if vars[f"b({m})"] > 0.5:
                name_list_DPF.append(m)
            # add inflowing & outflowing products to product flow
            for p in self.P:
                # inflowing products
                for l in self.CPF:
                    if vars[f"x({p},{l},{m})"] > 0.001:
                        # append flow data from l to m to DataFrame
                        new_data = {
                            "Origin": l,
                            "Destination": m,
                            "Product": p,
                            "Amount": vars[f"x({p},{l},{m})"],
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
                # outflowing products
                for n in self.C:
                    if vars[f"x({p},{m},{n})"] > 0.001:
                        # append flow data from m to n to DataFrame
                        new_data = {
                            "Origin": m,
                            "Destination": n,
                            "Product": p,
                            "Amount": vars[f"x({p},{m},{n})"],
                        }
                        self.product_flow = self.product_flow._append(
                            new_data, ignore_index=True
                        )
        self.name_list_DPF = name_list_DPF

        # save the ``self.product_flow`` DataFrame to a csv file
        self.product_flow.to_csv("results/product_flow.csv")

        # compute demand satisfaction of all customers in value chain
        demand_satisfaction = {}
        for p in self.P:
            for n in self.C:
                demand_satisfaction[(p, n)] = sum(
                    vars[f"x({p},{m},{n})"] for m in self.DPF
                )
        self.demand_satisfaction = demand_satisfaction

        # compute individual elements of the economic objective function
        # revenue
        self.Revenue = sum(
            sum(self.market_price[p] * self.demand_satisfaction[(p, n)] for n in self.C)
            for p in self.P
        )
        # transportation costs
        # transportation costs between S and OCF
        self.transportation_cost_1 = sum(
            2
            * self.D1.loc[i, j]
            * self.TC_ETICS
            * sum(vars[f"x({p},{i},{j})"] for p in self.P)
            for i in self.S
            for j in self.OCF
        )
        # transportation costs between OCF and MPF
        self.transportation_cost_2 = sum(
            2
            * self.D2.loc[j, k]
            * self.TC_comp_ETICS
            * sum(vars[f"x({p},{j},{k})"] for p in self.P)
            for j in self.OCF
            for k in self.MPF
        )
        # transportation cost between MPF and CPF
        self.transportation_cost_3 = sum(
            2
            * self.D3.loc[k, l]
            * self.TC_pre_concentrate
            * sum(vars[f"x({p},{k},{l})"] for p in self.P)
            for k in self.MPF
            for l in self.CPF
        )
        # transportation cost between CPF and DPF
        self.transportation_cost_4 = sum(
            2
            * self.D4.loc[l, m]
            * self.TC_pyrolysis_oil
            * sum(vars[f"x({p},{l},{m})"] for p in self.P)
            for l in self.CPF
            for m in self.DPF
        )
        # transportation cost between DPF and C
        self.transportation_cost_5 = sum(
            2
            * self.D5.loc[m, n]
            * self.TC_styrene
            * sum(vars[f"x({p},{m},{n})"] for p in self.P)
            for m in self.DPF
            for n in self.C
        )
        # CAPEX of facilities in the value chain
        self.capex_OCF = sum(self.ACI_OCF * vars[f"b({j})"] for j in self.OCF)
        self.capex_MPF = sum(self.ACI_MPF * vars[f"b({k})"] for k in self.MPF)
        self.capex_CPF = sum(self.ACI_CPF * vars[f"b({l})"] for l in self.CPF)
        self.capex_DPF = sum(self.ACI_DPF * vars[f"b({m})"] for m in self.DPF)
        # OPEX of facilities int he value chain
        self.opex_OCF = sum(
            (
                self.fOPEX_OCF * vars[f"b({j})"]
                + self.vOPEX_OCF
                * sum(vars[f"x({p},{i},{j})"] for i in self.S for p in self.P)
            )
            for j in self.OCF
        )
        self.opex_MPF = sum(
            (
                self.fOPEX_MPF * vars[f"b({k})"]
                + self.vOPEX_MPF
                * sum(vars[f"x({p},{j},{k})"] for j in self.OCF for p in self.P)
            )
            for k in self.MPF
        )
        self.opex_CPF = sum(
            (
                self.fOPEX_CPF * vars[f"b({l})"]
                + self.vOPEX_CPF
                * sum(vars[f"x({p},{k},{l})"] for k in self.MPF for p in self.P)
            )
            for l in self.CPF
        )
        self.opex_DPF = sum(
            (
                self.fOPEX_DPF * vars[f"b({m})"]
                + self.vOPEX_DPF
                * sum(vars[f"x({p},{l},{m})"] for l in self.CPF for p in self.P)
            )
            for m in self.DPF
        )

        # compute individual elements of the environmental objective function
        # transportation impact between S and OCF
        self.transportation_impact_1 = sum(
            2
            * self.D1.loc[i, j]
            * self.TI_ETICS
            * sum(vars[f"x({p},{i},{j})"] for p in self.P)
            for i in self.S
            for j in self.OCF
        )
        # transportation impact between OCF and MPF
        self.transportation_impact_2 = sum(
            2
            * self.D2.loc[j, k]
            * self.TI_comp_ETICS
            * sum(vars[f"x({p},{j},{k})"] for p in self.P)
            for j in self.OCF
            for k in self.MPF
        )
        # transportation impact between MPF and CPF
        self.transportation_impact_3 = sum(
            2
            * self.D3.loc[k, l]
            * self.TI_pre_concentrate
            * sum(vars[f"x({p},{k},{l})"] for p in self.P)
            for k in self.MPF
            for l in self.CPF
        )
        # transportation impact between CPF and DPF
        self.transportation_impact_4 = sum(
            2
            * self.D4.loc[l, m]
            * self.TI_pyrolysis_oil
            * sum(vars[f"x({p},{l},{m})"] for p in self.P)
            for l in self.CPF
            for m in self.DPF
        )
        # transportation impact between DPF and C
        self.transportation_impact_5 = sum(
            2
            * self.D5.loc[m, n]
            * self.TI_styrene
            * sum(vars[f"x({p},{m},{n})"] for p in self.P)
            for m in self.DPF
            for n in self.C
        )
        # construction impact
        self.construction_impact_OCF = sum(
            self.CI_OCF * sum(vars[f"x({p},{i},{j})"] for i in self.S for p in self.P)
            for j in self.OCF
        )
        self.construction_impact_MPF = sum(
            self.CI_MPF * sum(vars[f"x({p},{j},{k})"] for j in self.OCF for p in self.P)
            for k in self.MPF
        )
        self.construction_impact_CPF = sum(
            self.CI_CPF * sum(vars[f"x({p},{k},{l})"] for k in self.MPF for p in self.P)
            for l in self.CPF
        )
        self.construction_impact_DPF = sum(
            self.CI_DPF * sum(vars[f"x({p},{l},{m})"] for l in self.CPF for p in self.P)
            for m in self.DPF
        )
        # operational impact
        self.operational_impact_OCF = sum(
            self.OI_OCF * sum(vars[f"x({p},{i},{j})"] for i in self.S for p in self.P)
            for j in self.OCF
        )
        self.operational_impact_MPF = sum(
            self.OI_MPF * sum(vars[f"x({p},{j},{k})"] for j in self.OCF for p in self.P)
            for k in self.MPF
        )
        self.operational_impact_CPF = sum(
            self.OI_CPF * sum(vars[f"x({p},{k},{l})"] for k in self.MPF for p in self.P)
            for l in self.CPF
        )
        self.operational_impact_DPF = sum(
            self.OI_DPF * sum(vars[f"x({p},{l},{m})"] for l in self.CPF for p in self.P)
            for m in self.DPF
        )

        # compute break-even price of styrene and the LCA's functional unit
        self.styrene_amount = sum(
            sum(vars[f"x({p},{m},{n})"] for m in self.DPF for p in self.P)
            for n in self.C
        )
        self.break_even_price = (
            abs(self.obj_economic - self.Revenue) / self.styrene_amount
        )
        self.functional_unit = (
            self.obj_environmental / self.env_cost / self.styrene_amount
        )
        print(f"Amount of styrene produced: {self.styrene_amount:.2f} [ton]")
        print(
            f"Break-even price of styrene: {self.break_even_price:.2f} [euro/ton of styrene]"
        )
        print(
            f"LCA functional unit: {self.functional_unit:.2f} [ton CO2eq/ton of styrene]"
        )

    def plot_infrastructure(self, country=None, img_path=None):
        """
        Create plot where the nodes are plotted as a scatter plot with the size
        of the node corresponding to the amount of waste sourced from it. The
        installed facilities (and the presence or lack of customers) is then
        indicated by icons which are used to annotate nodes on the plot.

        Parameters
        ----------
        country (str): optional string containing name of the considered country
        in english, used to obtain the country's centre and extremes (northmost
        southmost, eastmost, westmost) for setting plot limits

        img_path (str): optional string containing the location of the image
        file which (if specified) will be used as a background for the generated
        plot, REQUIRES ``country`` to be specified as well

        Notes
        -----
        This is a good way for visually inspecting and checking smaller
        networks.
        """

        # convert source_cap DataFrame to list containing row sums
        source_cap_row_sums = self.source_cap.sum(axis=1).to_list()
        # extract source coordinates to list
        sources = pd.read_csv("results/coordinates_sources.csv")
        x_coords = sources["xcord"].to_list()
        y_coords = sources["ycord"].to_list()
        # extract customer coordinates list
        customers = pd.read_csv("results/coordinates_customers.csv")
        x_coords_c = customers["xcord"].to_list()
        y_coords_c = customers["ycord"].to_list()

        # obtain plot limits using geocoder if ``country`` is specified
        if country != None:
            # obtain corresponding lats and lngs using Nominatim geocoder
            centre = gc.get_country_coords(country)
            bounding_box = gc.get_country_coords(country, output_as="boundingbox")
            # calculate x and y distances to the bottom left and top right corners
            bottom_left = cc.coords_to_distances(
                (bounding_box[0], bounding_box[2]), (centre[0], centre[1])
            )
            top_right = cc.coords_to_distances(
                (bounding_box[1], bounding_box[3]), (centre[0], centre[1])
            )
            # extract the required extents for the plot
            x_min = bottom_left[0]
            x_max = top_right[0]
            y_min = bottom_left[1]
            y_max = top_right[1]
        # otherwise, obtain plot limits based on plotted data
        else:
            length_x = max(x_coords) - min(x_coords)
            length_y = max(y_coords) - min(y_coords)

        # convert name_list of installed facilities into an int_list
        int_list_OCF = name_list_to_int_list(self.name_list_OCF)
        int_list_MPF = name_list_to_int_list(self.name_list_MPF)
        int_list_CPF = name_list_to_int_list(self.name_list_CPF)
        int_list_DPF = name_list_to_int_list(self.name_list_DPF)

        # use source row sums to create scatter plot with scaled source size
        fig, ax = plt.subplots(figsize=(8, 8))
        factor = int(300 / max(source_cap_row_sums))
        size = [factor * val for val in source_cap_row_sums]
        ax.scatter(x_coords, y_coords, c="k", s=size, zorder=1)
        ax.set_xlabel("Horizontal distance [km]")
        ax.set_ylabel("Vertical distance [km]")
        # set corresponding limits to the plot
        if country != None:
            ax.set_xlim(x_min * 1.2, x_max * 1.2)
            ax.set_ylim(y_min * 1.2, y_max * 1.2)
        else:
            ax.set_xlim(min(x_coords) - length_x * 0.2, max(x_coords) + length_x * 0.2)
            ax.set_ylim(min(y_coords) - length_y * 0.2, max(y_coords) + length_y * 0.2)
        # set background image if ``img_path`` has been provided
        if img_path != None and country != None:
            background_img = plt.imread(img_path)
            ax.imshow(background_img, zorder=0, extent=[x_min, x_max, y_min, y_max])
        # chosen offset for annotation from the node
        offset = max(20, self.source_cap.values.max() / 60)
        # annotate nodes where OCFs have been installed
        imagebox = osb.OffsetImage(plt.imread("icons/OCF.png"), zoom=0.03)
        for value in int_list_OCF:
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coords[value], y_coords[value]),
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
                xy=(x_coords[value], y_coords[value]),
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
                xy=(x_coords[value], y_coords[value]),
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
                xy=(x_coords[value], y_coords[value]),
                xybox=(offset * math.sqrt(2) / 2, offset * math.sqrt(2) / 2),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # annotate nodes where customers are located
        imagebox = osb.OffsetImage(plt.imread("icons/C.png"), zoom=0.03)
        for idx in range(0, len(x_coords_c)):
            ab = osb.AnnotationBbox(
                imagebox,
                xy=(x_coords_c[idx], y_coords_c[idx]),
                xybox=(offset, 0),
                frameon=False,
                boxcoords="offset points",
            )
            plt.gca().add_artist(ab)
        # set plot title
        plt.title(
            f"Net profit: {self.obj_economic:_.2f} euro/day\nEnv. impact: {self.obj_environmental/self.env_cost:_.2f} tons CO2-eq/day".replace(
                "_", "'"
            )
        )
        # plot and save the figure
        fig.savefig("results/infrastructure.pdf", dpi=1200)

    def plot_product_flow(self, country=None, img_path=None, layered=False):
        """
        Create a plot where product flows are plotted between nodes represented
        by a scatter plot. The nodes of the scatter plot are scaled according to
        the amount of waste available at them. Product interchange between nodes
        is represented by a line connecting them. The lines and nodes are colour
        and shape coded according to the facilities installed at the node and
        the type of material being transported.

        Parameters
        ----------
        country (str): optional string containing name of the considered country
        in english, used to obtain the country's centre and extremes (northmost
        southmost, eastmost, westmost) for setting plot limits

        img_path (str): optional string containing the location of the image
        file which (if specified) will be used as a background for the generated
        plot, REQUIRES ``country`` to be specified as well

        layered (bool): optional boolean which is False by default, if set to
        True then the default figure will be generated along with a series of
        figures where the product flow is separeted into layers by plotting each
        product flow type in a separate figure (the layered visualisation is
        useful for analysing complex or very large networks)

        Notes
        -----
        This is a good way for visually inspecting and checking larger networks.
        """

        # define colour scheme used throughout using hex notation
        # colours correspond to: yellow, orange, red, purple, indigo
        colours = ["#ffa600", "#ff6361", "#bc5090", "#58508d", "#003f5c"]

        # convert source_cap DataFrame to list containing row sums
        source_cap_row_sums = self.source_cap.sum(axis=1).to_list()
        # extract source coordinates to list
        sources = pd.read_csv("results/coordinates_sources.csv")
        x_coords = sources["xcord"].to_list()
        y_coords = sources["ycord"].to_list()
        # convert name_list of installed facilities into an int_list
        int_list_OCF = name_list_to_int_list(self.name_list_OCF)
        int_list_MPF = name_list_to_int_list(self.name_list_MPF)
        int_list_CPF = name_list_to_int_list(self.name_list_CPF)
        int_list_DPF = name_list_to_int_list(self.name_list_DPF)
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
            label="up to OCF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[1],
            markerfacecolor=colours[1],
            linestyle="",
        )
        point_MPF = mlines.Line2D(
            [0],
            [0],
            label="up to MPF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[2],
            markerfacecolor=colours[2],
            linestyle="",
        )
        point_CPF = mlines.Line2D(
            [0],
            [0],
            label="up to CPF",
            marker="o",
            markersize=10,
            markeredgecolor=colours[3],
            markerfacecolor=colours[3],
            linestyle="",
        )
        point_DPF = mlines.Line2D(
            [0],
            [0],
            label="up to DPF",
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
            label="styrene flow",
            c=colours[4],
        )
        # obtain plot limits using geocoder if ``country`` is specified
        if country != None:
            # obtain corresponding lats and lngs using Nominatim geocoder
            centre = gc.get_country_coords(country)
            bounding_box = gc.get_country_coords(country, output_as="boundingbox")
            # calculate x and y distances to the bottom left and top right corners
            bottom_left = cc.coords_to_distances(
                (bounding_box[0], bounding_box[2]), (centre[0], centre[1])
            )
            top_right = cc.coords_to_distances(
                (bounding_box[1], bounding_box[3]), (centre[0], centre[1])
            )
            # extract the required extents for the plot
            x_min = bottom_left[0]
            x_max = top_right[0]
            y_min = bottom_left[1]
            y_max = top_right[1]
        # otherwise, obtain plot limits based on plotted data
        else:
            length_x = max(x_coords) - min(x_coords)
            length_y = max(y_coords) - min(y_coords)

        # create the default unlayered figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("Horizontal distance [km]")
        ax.set_ylabel("Vertical distance [km]")
        # set corresponding limits to the plot
        if country != None:
            ax.set_xlim(x_min * 1.2, x_max * 1.2)
            ax.set_ylim(y_min * 1.2, y_max * 1.5)
        else:
            ax.set_xlim(min(x_coords) - length_x * 0.2, max(x_coords) + length_x * 0.2)
            ax.set_ylim(min(y_coords) - length_y * 0.2, max(y_coords) + length_y * 0.5)
        # set background image if ``img_path`` has been provided
        if img_path != None and country != None:
            background_img = plt.imread(img_path)
            ax.imshow(background_img, zorder=0, extent=[x_min, x_max, y_min, y_max])
        # loop over the source coordinates (that is, number of nodes)
        factor = int(300 / max(source_cap_row_sums))
        for idx in range(0, len(x_coords)):
            # plot node with select node colour
            if idx in int_list_DPF:
                colour = colours[4]
            elif idx in int_list_CPF:
                colour = colours[3]
            elif idx in int_list_MPF:
                colour = colours[2]
            elif idx in int_list_OCF:
                colour = colours[1]
            else:
                colour = colours[0]
            # plot the node
            plt.scatter(
                x_coords[idx],
                y_coords[idx],
                c=colour,
                s=source_cap_row_sums[idx] * factor,
                zorder=2.5,
            )
        # draw lines representing exchanged products
        product_flows = pd.read_csv("results/product_flow.csv")
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
                x = (x_coords[origin_int], x_coords[destination_int])
                y = (y_coords[origin_int], y_coords[destination_int])
                plt.plot(x, y, lw=2, c=colour)
                # annotate the line with an arrow showing flow direction
                x_diff = x_coords[destination_int] - x_coords[origin_int]
                y_diff = y_coords[destination_int] - y_coords[origin_int]
                plt.annotate(
                    "",
                    xy=(
                        x_coords[origin_int] + 0.8 * x_diff,
                        y_coords[origin_int] + 0.8 * y_diff,
                    ),
                    xytext=(
                        x_coords[origin_int] + 0.6 * x_diff,
                        y_coords[origin_int] + 0.6 * y_diff,
                    ),
                    arrowprops=dict(
                        arrowstyle="->", lw=2, color=colour, mutation_scale=25
                    ),
                    zorder=1,
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
        plt.title(
            f"Net profit: {self.obj_economic:_.2f} euro/day\nEnv. impact: {self.obj_environmental/self.env_cost:_.2f} tons CO2-eq/day".replace(
                "_", "'"
            )
        )
        fig.savefig("results/product_flow.pdf", dpi=1200)

        # create layered figure if specified
        if layered:
            # loop over the number of layers required & clear existing figure
            plt.clf()
            for layer in range(0, len(colours)):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlabel("Horizontal distance [km]")
                ax.set_ylabel("Vertical distance [km]")
                # set corresponding limits to the plot
                if country != None:
                    ax.set_xlim(x_min * 1.2, x_max * 1.2)
                    ax.set_ylim(y_min * 1.2, y_max * 1.5)
                else:
                    ax.set_xlim(
                        min(x_coords) - length_x * 0.2, max(x_coords) + length_x * 0.2
                    )
                    ax.set_ylim(
                        min(y_coords) - length_y * 0.2, max(y_coords) + length_y * 0.5
                    )
                # set background image if ``img_path`` has been provided
                if img_path != None and country != None:
                    background_img = plt.imread(img_path)
                    ax.imshow(
                        background_img, zorder=0, extent=[x_min, x_max, y_min, y_max]
                    )
                # loop over the source coordinates (that is, number of nodes)
                # create a scatter plont of all nodes
                factor = int(300 / max(source_cap_row_sums))
                for idx in range(0, len(x_coords)):
                    # plot node with select node colour
                    if idx in int_list_DPF:
                        colour = colours[4]
                    elif idx in int_list_CPF:
                        colour = colours[3]
                    elif idx in int_list_MPF:
                        colour = colours[2]
                    elif idx in int_list_OCF:
                        colour = colours[1]
                    else:
                        colour = colours[0]
                    # plot the node
                    plt.scatter(
                        x_coords[idx],
                        y_coords[idx],
                        c=colour,
                        s=source_cap_row_sums[idx] * factor,
                        zorder=2.5,
                    )
                # draw lines representing product flow ONLY on current layer
                product_flows = pd.read_csv("results/product_flow.csv")
                title = f"Layer: {self.P[layer]} flow".replace("_", " ")
                for _, row in product_flows.iterrows():
                    origin_int = name_to_int(row["Origin"])
                    destination_int = name_to_int(row["Destination"])
                    # draw line only if origin & destination have diff node num.
                    if origin_int != destination_int:
                        # plot lines only if they should appear on current layer
                        if self.P.index(row["Product"]) == layer:
                            colour = colours[layer]
                            # draw the corresponding product flow line
                            x = (x_coords[origin_int], x_coords[destination_int])
                            y = (y_coords[origin_int], y_coords[destination_int])
                            plt.plot(x, y, lw=2, c=colour)
                            # annotate w arrow showing flow direction
                            x_diff = x_coords[destination_int] - x_coords[origin_int]
                            y_diff = y_coords[destination_int] - y_coords[origin_int]
                            plt.annotate(
                                "",
                                xy=(
                                    x_coords[origin_int] + 0.8 * x_diff,
                                    y_coords[origin_int] + 0.8 * y_diff,
                                ),
                                xytext=(
                                    x_coords[origin_int] + 0.6 * x_diff,
                                    y_coords[origin_int] + 0.6 * y_diff,
                                ),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    lw=2,
                                    color=colour,
                                    mutation_scale=25,
                                ),
                                zorder=1,
                            )
                        else:
                            continue
                # finish annotating figure and save it
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
                plt.title(title)
                file_name = f"results/product_flow_layer{layer}.pdf"
                fig.savefig(file_name, dpi=1200)

    def plot_objective_function_breakdown(self):
        """
        Function for plotting a bar graph where the individual components of
        both the economic and environmental objective functions are broken down.
        The plot generates an individual subplot for each of the objective
        functions.

        Notes
        -----
        This is an excellent method for comparing where the costs and
        environmental impacts occur in different networks.
        """

        # define colour scheme used throughout using hex notation
        # colours correspond to: yellow, orange, red, purple, indigo
        colours = ["#ffa600", "#ff6361", "#bc5090", "#58508d", "#003f5c"]

        # define number of categories
        N = 3

        # clear the figure
        plt.clf()

        # define economic subcategories and their values
        ETICS = (self.transportation_cost_1, 0, 0)
        compressed_ETICS = (
            self.transportation_cost_2,
            self.capex_OCF,
            self.opex_OCF,
        )
        pre_concentrate = (
            self.transportation_cost_3,
            self.capex_MPF,
            self.opex_MPF,
        )
        pyrolysis_oil = (
            self.transportation_cost_4,
            self.capex_CPF,
            self.opex_CPF,
        )
        styrene = (
            self.transportation_cost_5,
            self.capex_DPF,
            self.opex_DPF,
        )

        # define the bottoms using numpy to add the tuples
        bottom3 = np.add(np.array(ETICS), np.array(compressed_ETICS))
        bottom4 = np.add(np.array(bottom3), np.array(pre_concentrate))
        bottom5 = np.add(np.array(bottom4), np.array(pyrolysis_oil))

        # plot the bar graphs for the economic values and save the figure
        ind = np.arange(N)
        width = 0.65
        fig, ax = plt.subplots(figsize=(8, 8))
        b1 = ax.bar(ind, ETICS, width, color=colours[0])
        b2 = ax.bar(ind, compressed_ETICS, width, bottom=ETICS, color=colours[1])
        b3 = ax.bar(
            ind,
            pre_concentrate,
            width,
            bottom=bottom3,
            color=colours[2],
        )
        b4 = ax.bar(
            ind,
            pyrolysis_oil,
            width,
            bottom=bottom4,
            color=colours[3],
        )
        b5 = ax.bar(
            ind,
            styrene,
            width,
            bottom=bottom5,
            color=colours[4],
        )
        ax.set_ylabel("Economic cost [euro/day]")
        ax.set_xticks(ind, ("TC", "CAPEX", "OPEX"))
        ax.legend(
            (b1[0], b2[0], b3[0], b4[0], b5[0]),
            (
                "ETICS",
                "compressed ETICS or OCF",
                "pre-concentrate or MPF",
                "pyrolysis oil or CPF",
                "styrene or DPF",
            ),
            loc="upper left",
        )
        ax.set_title(
            f"Total economic cost: {abs(self.obj_economic - self.Revenue):_.2f} euro/day".replace(
                "_", "'"
            )
        )
        fig.savefig("results/economic_objective_breakdown.pdf", dpi=1200)

        # clear the figure
        plt.clf()

        # define environmental subcategories and their values
        ETICS = (float(self.transportation_impact_1), float(0), float(0))
        compressed_ETICS = (
            self.transportation_impact_2,
            self.construction_impact_OCF,
            self.operational_impact_OCF,
        )
        pre_concentrate = (
            self.transportation_impact_3,
            self.construction_impact_MPF,
            self.operational_impact_MPF,
        )
        pyrolysis_oil = (
            self.transportation_impact_4,
            self.construction_impact_CPF,
            self.operational_impact_CPF,
        )
        styrene = (
            self.transportation_impact_5,
            self.construction_impact_DPF,
            self.operational_impact_DPF,
        )

        # define the bottoms using numpy to add the tuples
        bottom3 = np.add(np.array(ETICS), np.array(compressed_ETICS))
        bottom4 = np.add(np.array(bottom3), np.array(pre_concentrate))
        bottom5 = np.add(np.array(bottom4), np.array(pyrolysis_oil))

        # plot the bar graphs for the environmental values and save the figure
        ind = np.arange(N)
        width = 0.65
        fig, ax = plt.subplots(figsize=(8, 8))
        b1 = ax.bar(ind, ETICS, width, color=colours[0])
        b2 = ax.bar(ind, compressed_ETICS, width, bottom=ETICS, color=colours[1])
        b3 = ax.bar(
            ind,
            pre_concentrate,
            width,
            bottom=bottom3,
            color=colours[2],
        )
        b4 = ax.bar(
            ind,
            pyrolysis_oil,
            width,
            bottom=bottom4,
            color=colours[3],
        )
        b5 = ax.bar(
            ind,
            styrene,
            width,
            bottom=bottom5,
            color=colours[4],
        )
        ax.set_ylabel("Environmental impact [tons CO2-eq/day]")
        ax.set_xticks(ind, ("TI", "CI", "OI"))
        ax.legend(
            (b1[0], b2[0], b3[0], b4[0], b5[0]),
            (
                "ETICS",
                "compressed ETICS or OCF",
                "pre-concentrate or MPF",
                "pyrolysis oil or CPF",
                "styrene or DPF",
            ),
            loc="upper left",
        )
        ax.set_title(
            f"Total environmental impact: {self.obj_environmental/self.env_cost:_.2f} tons CO2-eq/day".replace(
                "_", "'"
            )
        )
        fig.savefig("results/environmental_objective_breakdown.pdf", dpi=1200)

    def tabulate_product_flows(self):
        """
        Generate a table with the facility types as columns and the node numbers
        as rows. The cells are populated with the amount of product flowing
        through a specific type of facility in a specific node [tons/day], along
        with the number of facilities installed and their capacity in
        parenthesis. Two tables are generated from the results: one through
        creating a tex file and callind pdflatex, and one using Seaborn which
        displays only teh values (not the number of installed facilities).
        """

        # extract resulting variable values and store them in a dictionary
        vars = {}
        for var in self.model.getVars():
            vars[f"{var.varName}"] = var.x

        # read the product_flow csv using pandas
        product_flows = pd.read_csv("results/product_flow.csv")

        # generate a pandas dataframe for the tabulation
        columns = ["OCF [ton/day]", "MPF [ton/day]", "CPF [ton/day]", "DPF [ton/day]"]
        tabulated_product_flow = pd.DataFrame(columns=columns)
        tabulated_product_flow.rename_axis("node", axis=1)
        # generate another dataframe which will store only the values
        tabulated_product_flow_val = pd.DataFrame(columns=columns)
        tabulated_product_flow_val.rename_axis("node", axis=1)

        # loop over the nodes in the network
        sources = pd.read_csv("results/coordinates_sources.csv")
        x_coords = sources["xcord"].to_list()
        for node in range(0, len(x_coords)):
            # filter the df using OCF & current node, then extract info
            name = f"OCF_{node}"
            OCF_amount = 0
            filtered = product_flows[product_flows["Destination"] == name]
            for _, row in filtered.iterrows():
                OCF_amount += row["Amount"]
            OCF_installed = abs(vars[f"b({name})"])
            OCF_entry = f"{OCF_amount:.1f} ({OCF_installed:.0f} * {self.facility_cap['OCF']:.1f})"
            OCF_entry_val = OCF_amount

            # filter the df using MPF & current node, then extract info
            name = f"MPF_{node}"
            MPF_amount = 0
            filtered = product_flows[product_flows["Destination"] == name]
            for _, row in filtered.iterrows():
                MPF_amount += row["Amount"]
            MPF_installed = abs(vars[f"b({name})"])
            MPF_entry = f"{MPF_amount:.1f} ({MPF_installed:.0f} * {self.facility_cap['MPF']:.1f})"
            MPF_entry_val = MPF_amount

            # filter the df using CPF & current node, then extract info
            name = f"CPF_{node}"
            CPF_amount = 0
            filtered = product_flows[product_flows["Destination"] == name]
            for _, row in filtered.iterrows():
                CPF_amount += row["Amount"]
            CPF_installed = abs(vars[f"b({name})"])
            CPF_entry = f"{CPF_amount:.1f} ({CPF_installed:.0f} * {self.facility_cap['CPF']:.1f})"
            CPF_entry_val = CPF_amount

            # filter the df using DPF & current node, then extract info
            name = f"DPF_{node}"
            DPF_amount = 0
            filtered = product_flows[product_flows["Destination"] == name]
            for _, row in filtered.iterrows():
                DPF_amount += row["Amount"]
            DPF_installed = abs(vars[f"b({name})"])
            DPF_entry = f"{DPF_amount:.1f} ({DPF_installed:.0f} * {self.facility_cap['DPF']:.1f})"
            DPF_entry_val = DPF_amount

            # construct new data for this node and append it to the dataframe
            new_data = {
                "OCF [ton/day]": OCF_entry,
                "MPF [ton/day]": MPF_entry,
                "CPF [ton/day]": CPF_entry,
                "DPF [ton/day]": DPF_entry,
            }
            tabulated_product_flow = tabulated_product_flow._append(
                new_data, ignore_index=True
            )
            # do the same for the dataframe containing only the values
            new_data_val = {
                "OCF [ton/day]": OCF_entry_val,
                "MPF [ton/day]": MPF_entry_val,
                "CPF [ton/day]": CPF_entry_val,
                "DPF [ton/day]": DPF_entry_val,
            }
            tabulated_product_flow_val = tabulated_product_flow_val._append(
                new_data_val, ignore_index=True
            )

        # name index columns of both dataframes
        tabulated_product_flow.index.name = "node"
        tabulated_product_flow_val.index.name = "node"

        # save the dataframe as a table in pdf format using latex
        filename = "results/tabulated_product_flow.tex"
        directory = "results/"
        template = r"""\documentclass[preview]{{standalone}}
        \usepackage{{booktabs}}
        \begin{{document}}
        {}
        \end{{document}}
        """
        with open(filename, "w") as f:
            f.write(template.format(tabulated_product_flow.to_latex()))
        # subprocess.call(["pdflatex", filename])
        subprocess.run(
            ["pdflatex", "-output-directory=" + directory, filename],
            stdout=subprocess.PIPE,
        )

        # generate Seaborn heatmap
        plt.clf()
        sns.heatmap(
            tabulated_product_flow_val,
            annot=True,
            cmap="Reds",
            yticklabels=tabulated_product_flow_val.index[::-1],
        )
        plt.savefig("results/tabulated_product_flow_heatmap.pdf", dpi=1200)


@staticmethod
def six_tenths_rule(reference_capacity, target_capacity, reference_cost):
    """
    Apply the six-tenths rule to approximate the cost ``target_cost`` for a
    ``target_capacity`` using known ``reference_cost`` for a
    ``reference_capacity``. The function is only reliable if the
    ``target_capacity`` and ``reference_capacity`` differ by a factor which is
    smaller than 10.

    Parameters
    ----------
    reference_capacity (float): capacity of the facility being used as a
    reference [ton/day]

    target_capacity (float): capacity of the upscaledd or downscaled facility
    for which we want to determine the CAPEX [ton/day]

    reference_cost (float): cost of the reference facility [Meuro]

    Returns
    -------
    target_cost (float): cost of the upscaled or downscaled facility [Meuro]

    NOTE: The function will exit the code instance and print a warning to the
    user in the case that it is being used within a range where the method is no
    longer considered to provide a reliable approximation.
    """

    ratio = reference_capacity / target_capacity

    if ratio > 10 or ratio < 0.1:
        print(
            f"WARNING: You are using the six-tenths rule to scale-up or scale-down cost for capacities that differ by a factor of {ratio:.2f}. This method cannot be used beyond a factor of 10."
        )
        sys.exit()

    target_cost = reference_cost * (target_capacity / reference_capacity) ** 0.6

    return target_cost


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
