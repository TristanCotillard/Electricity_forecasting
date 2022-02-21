from pyomo.environ import *
import pandas as pd

def GetElectricSystemModel_Param_Interco_Storage_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters,
                                                                   StorageParameters,
                                                                   obj_param_df,
                                                                   availabilityFactor_import,
                                                                   TechParameters_import,
                                                                   import_df,
                                                                   availabilityFactor_export,
                                                                   TechParameters_export,
                                                                   export_df,
                                                                   ExchangeParameters=pd.DataFrame({'empty': []}),
                                                                   isAbstract=False, LineEfficiency=1):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """

    # isAbstract=False
    availabilityFactor.isna().sum()

    ### Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    INTERCOS = set(TechParameters_import.index.get_level_values('INTERCOS').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()
    AREAS = set(areaConsumption.index.get_level_values('AREAS').unique())

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = AbstractModel()
    else:
        model = ConcreteModel()

    ###############
    # Sets       ##
    ###############

    # Simple
    model.AREAS = Set(initialize=AREAS, doc="Area", ordered=False)
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.INTERCOS = Set(initialize=INTERCOS, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    # Products
    model.TIMESTAMP_TECHNOLOGIES = model.TIMESTAMP * model.TECHNOLOGIES
    model.AREAS_AREAS = model.AREAS * model.AREAS
    model.AREAS_TECHNOLOGIES = model.AREAS * model.TECHNOLOGIES
    model.AREAS_STOCKTECHNO = model.AREAS * model.STOCK_TECHNO
    model.AREAS_INTERCOS = model.AREAS * model.INTERCOS
    model.AREAS_TIMESTAMP = model.AREAS * model.TIMESTAMP
    model.AREAS_TIMESTAMP_TECHNOLOGIES = model.AREAS * model.TIMESTAMP * model.TECHNOLOGIES
    model.AREAS_TIMESTAMP_INTERCOS = model.AREAS * model.TIMESTAMP * model.INTERCOS

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.AREAS_TIMESTAMP,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Any,
                                  mutable=True)
    model.availabilityFactor = Param(model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.squeeze().to_dict())
    model.availabilityFactor_import = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=PercentFraction, default=1,
                                            initialize=availabilityFactor_import.squeeze().to_dict())
    model.availabilityFactor_export = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=PercentFraction, default=1,
                                            initialize=availabilityFactor_export.squeeze().to_dict())

    if not ExchangeParameters.empty:
        model.maxExchangeCapacity = Param(model.AREAS_AREAS, initialize=ExchangeParameters.squeeze().to_dict(),
                                          domain=NonNegativeReals, default=0)
    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        exec("model." + COLNAME + " =          Param(model.AREAS_TECHNOLOGIES, default=0," +
             "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")
    for COLNAME in StorageParameters:
        exec("model." + COLNAME + " =Param(model.AREAS_STOCKTECHNO,domain=NonNegativeReals,default=0," +
             "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict())")
    for COLNAME in TechParameters_import:
        exec("model." + COLNAME + "_import =          Param(model.AREAS_INTERCOS, default=0," +
             "initialize=TechParameters_import." + COLNAME + ".squeeze().to_dict())")
    for COLNAME in TechParameters_export:
        exec("model." + COLNAME + "_export =          Param(model.AREAS_INTERCOS, default=0," +
             "initialize=TechParameters_export." + COLNAME + ".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())

    ################
    # Variables    #
    ################

    model.energy = Var(model.AREAS, model.TIMESTAMP, model.TECHNOLOGIES,
                       domain=NonNegativeReals)  ### Energy produced by a production mean at time t
    model.energy_import = Var(model.AREAS, model.TIMESTAMP, model.INTERCOS,
                              domain=NonNegativeReals)  ### Energy produced by a production mean at time t
    model.energy_export = Var(model.AREAS, model.TIMESTAMP, model.INTERCOS,
                              domain=NonNegativeReals)  ### Energy produced by a production mean at time t
    model.exchange = Var(model.AREAS_AREAS, model.TIMESTAMP)
    # model.energyCosts=Var(model.AREAS,model.TECHNOLOGIES)   ### Cost of energy by a production mean for area at time t (explicitely defined by constraint energyCostsCtr)

    ###Storage means :
    model.storageIn = Var(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO,
                          domain=NonNegativeReals)  ### Energy stored by a storage mean for areas at time t
    model.storageOut = Var(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO,
                           domain=NonNegativeReals)  ### Energy taken out of a storage mean for areas at time t
    model.stockLevel = Var(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO,
                           domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.storageCosts = Var(model.AREAS,
                             model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.Cmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum capacity of a storage mean
    model.Pmax = Var(model.AREAS, model.STOCK_TECHNO)  # Maximum flow of energy in/out of a storage mean
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    # model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    #   Param prices       #
    ########################

    model.p0 = Param(model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=Any, default=1,
                     initialize=obj_param_df[['p0']].squeeze().to_dict())
    model.p1 = Param(model.AREAS_TIMESTAMP_TECHNOLOGIES, domain=Any, default=1,
                     initialize=obj_param_df[['p1']].squeeze().to_dict())

    model.p0_import = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=Any, default=1,
                            initialize=import_df[['p0']].squeeze().to_dict())
    model.p1_import = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=Any, default=1,
                            initialize=import_df[['p1']].squeeze().to_dict())

    model.p0_export = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=Any, default=1,
                            initialize=export_df[['p0']].squeeze().to_dict())
    model.p1_export = Param(model.AREAS_TIMESTAMP_INTERCOS, domain=Any, default=1,
                            initialize=export_df[['p1']].squeeze().to_dict())

    obj_param_df.isnull().values.any()

    ########################
    # Objective Function   #
    ########################

    # price_param_df[['techno_scarc_param']]
    # model.energyCost[area, tech]

    def ObjectiveFunction_rule(model):  # OBJ
        #     my_sum = 0
        #     for area in model.AREAS:
        #         for t in model.TIMESTAMP:
        #             for tech in model.TECHNOLOGIES:
        #                 if model.capacity[area,tech] !=0:
        #                     my_sum = (my_sum + (
        #                             price_param_df.loc[(area, tech), 'intercept_param']
        #                             + margin_price_df.loc[(area, t, tech), 'margin_price']
        #                             + price_param_df.loc[(area, tech),'techno_scarc_param'] * model.energy[area, t, tech]/model.capacity[area,tech])
        #                             * model.energy[area, t, tech])
        #
        #     return my_sum
        return (
                quicksum(
                    ((model.p0[area, t, tech] + model.p1[area, t, tech] * model.energy[area, t, tech]) * model.energy[
                        area, t, tech])
                    for area in model.AREAS for t in model.TIMESTAMP for tech in model.TECHNOLOGIES) +
                quicksum(
                    model.storageCosts[area, s_tech] for s_tech in model.STOCK_TECHNO for area in model.AREAS) +
                quicksum(
                    (model.p0_import[area, t, interco] * model.energy_import[area, t, interco])
                    for area in model.AREAS for t in model.TIMESTAMP for interco in model.INTERCOS) -
                quicksum(
                    (model.p0_export[area, t, interco] * model.energy_export[area, t, interco])
                    for area in model.AREAS for t in model.TIMESTAMP for interco in model.INTERCOS)
        )

        model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

        # return quicksum(
        #     (price_param_df.loc[(area, tech), 'intercept_param']
        #         + margin_price_df.loc[(area, t, tech), 'margin_price']
        #         #+ price_param_df.loc[(area, tech),'techno_scarc_param'] * vm_non_zero_division(model.energy[area, t, tech],model.capacity[area,tech]))
        #         + price_param_df.loc[(area, tech),'techno_scarc_param_on_capa'] * model.energy[area, t, tech])
        #     * model.energy[area, t, tech]
        #     for area in model.AREAS for t in model.TIMESTAMP for tech in model.TECHNOLOGIES)

    # OK	return sum(sum((model.energyCost[area,tech]+margin_price_df.loc[(area,t,tech),'margin_price'])*model.energy[area,t,tech] for t in model.TIMESTAMP) for tech in model.TECHNOLOGIES for area in model.AREAS);
    # OK    return sum(sum(model.energyCost[area, tech] * model.energy[area, t, tech] for t in model.TIMESTAMP) for tech in model.TECHNOLOGIES for area in model.AREAS);
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    #### 1 - Basics
    ########

    # # energyCost/totalCosts definition Constraints
    # # AREAS x TECHNOLOGIES
    # def energyCostsDef_rule(model,area,tech): #EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech] / 1E6;
    #     temp=model.energyCost[area,tech]#/10**6 ;
    #     return sum(temp*model.energy[area,t,tech] for t in model.TIMESTAMP) == model.energyCosts[area,tech];
    # model.energyCostsDef = Constraint(model.AREAS,model.TECHNOLOGIES, rule=energyCostsDef_rule)

    # Capacity constraint
    # AREAS x TIMESTAMP x TECHNOLOGIES
    def CapacityCtr_rule(model, area, t, tech):  # INEQ forall t, tech
        return model.capacity[area, tech] * model.availabilityFactor[area, t, tech] >= model.energy[area, t, tech]

    model.CapacityCtr = Constraint(model.AREAS, model.TIMESTAMP, model.TECHNOLOGIES, rule=CapacityCtr_rule)

    # capacityCosts definition Constraints
    # AREAS x STOCK_TECHNO
    def storageCostsDef_rule(model, area,
                             s_tech):  # EQ forall s_tech in STOCK_TECHNO storageCosts = storageCost[area, s_tech]*c_max[area, s_tech] / 1E6;
        return model.storageCost[area, s_tech] * model.Cmax[area, s_tech] == model.storageCosts[
            area, s_tech]  # /10**6 ;;

    model.storageCostsDef = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCostsDef_rule)

    def CapacityCtr_import_rule(model, area, t, interco):  # INEQ forall t, tech
        return model.capacity_import[area, interco] * model.availabilityFactor_import[area, t, interco] >= \
               model.energy_import[area, t, interco]

    model.CapacityCtr_import = Constraint(model.AREAS, model.TIMESTAMP, model.INTERCOS, rule=CapacityCtr_import_rule)

    def CapacityCtr_export_rule(model, area, t, interco):  # INEQ forall t, tech
        return model.capacity_export[area, interco] * model.availabilityFactor_export[area, t, interco] >= \
               model.energy_export[area, t, interco]

    model.CapacityCtr_export = Constraint(model.AREAS, model.TIMESTAMP, model.INTERCOS, rule=CapacityCtr_export_rule)

    # Storage max capacity constraint
    # AREAS x STOCK_TECHNO
    def storageCapacity_rule(model, area, s_tech):  # INEQ forall s_tech
        return model.Cmax[area, s_tech] <= model.c_max[area, s_tech]

    model.storageCapacityCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storageCapacity_rule)

    # Storage max power constraint
    # AREAS x STOCK_TECHNO
    def storagePower_rule(model, area, s_tech):  # INEQ forall s_tech
        return model.Pmax[area, s_tech] <= model.p_max[area, s_tech]

    model.storagePowerCtr = Constraint(model.AREAS, model.STOCK_TECHNO, rule=storagePower_rule)

    # contraintes de stock puissance
    # AREAS x TIMESTAMP x STOCK_TECHNO
    def StoragePowerUB_rule(model, area, t, s_tech):  # INEQ forall t
        return model.storageIn[area, t, s_tech] - model.Pmax[area, s_tech] <= 0

    model.StoragePowerUBCtr = Constraint(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, area, t, s_tech, ):  # INEQ forall t
        return model.storageOut[area, t, s_tech] - model.Pmax[area, s_tech] <= 0

    model.StoragePowerLBCtr = Constraint(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO, rule=StoragePowerLB_rule)

    # contraintes de stock capacité
    # AREAS x TIMESTAMP x STOCK_TECHNO
    def StockLevel_rule(model, area, t, s_tech):  # EQ forall t
        if t > 1:
            return model.stockLevel[area, t, s_tech] == model.stockLevel[area, t - 1, s_tech] * (
                    1 - model.dissipation[area, s_tech]) + model.storageIn[area, t, s_tech] * model.efficiency_in[
                       area, s_tech] - model.storageOut[area, t, s_tech] / model.efficiency_out[area, s_tech]
        else:
            return model.stockLevel[area, t, s_tech] == 0

    model.StockLevelCtr = Constraint(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockCapacity_rule(model, area, t, s_tech, ):  # INEQ forall t
        return model.stockLevel[area, t, s_tech] <= model.Cmax[area, s_tech]

    model.StockCapacityCtr = Constraint(model.AREAS, model.TIMESTAMP, model.STOCK_TECHNO, rule=StockCapacity_rule)

    # contrainte d'equilibre offre demande
    # AREAS x TIMESTAMP
    def energyCtr_rule(model, area, t):  # INEQ forall t
        return sum(model.energy[area, t, tech] for tech in model.TECHNOLOGIES) + sum(
            model.storageOut[area, t, s_tech] - model.storageIn[area, t, s_tech] for s_tech in model.STOCK_TECHNO) + sum(
            model.energy_import[area, t, interco] for interco in model.INTERCOS) - sum(
            model.energy_export[area, t, interco] for interco in model.INTERCOS) + sum(
            model.exchange[b, area, t] * LineEfficiency for b in model.AREAS) == model.areaConsumption[area, t]

    model.energyCtr = Constraint(model.AREAS, model.TIMESTAMP, rule=energyCtr_rule)

    # #import or export constraint
    # def intercoCtr_rule(model,area,t,interco): #INEQ forall t, tech
    #     return model.energy_import[area,t,interco] * model.energy_export[area,t,interco] == 0
    # model.intercoCtr = Constraint(model.AREAS, model.TIMESTAMP, model.INTERCOS, rule=intercoCtr_rule)

    # Exchange capacity constraint (duplicate of variable definition)
    # if not ExchangeParameters.empty:
    # AREAS x AREAS x TIMESTAMP
    def exchangeCtrPlus_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a != b:
            return model.exchange[a, b, t] <= model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0

    model.exchangeCtrPlus = Constraint(model.AREAS, model.AREAS, model.TIMESTAMP, rule=exchangeCtrPlus_rule)

    def exchangeCtrMoins_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        if a != b:
            return model.exchange[a, b, t] >= -model.maxExchangeCapacity[a, b];
        else:
            return model.exchange[a, a, t] == 0

    model.exchangeCtrMoins = Constraint(model.AREAS, model.AREAS, model.TIMESTAMP, rule=exchangeCtrMoins_rule)

    def exchangeCtr2_rule(model, a, b, t):  # INEQ forall area.axarea.b in AREASxAREAS  t in TIMESTAMP
        return model.exchange[a, b, t] == -model.exchange[b, a, t];

    model.exchangeCtr2 = Constraint(model.AREAS, model.AREAS, model.TIMESTAMP, rule=exchangeCtr2_rule)

    # def energyCtr_rule(model,t): #INEQ forall t
    # 	return sum(model.energy[t,tech] for tech in model.TECHNOLOGIES ) >= model.areaConsumption[t]
    # model.energyCtr = Constraint(model.TIMESTAMP,rule=energyCtr_rule)

    #### 2 - Optional
    ########

    # contrainte de stock annuel
    # AREAS x TECHNOLOGIES
    if "EnergyNbhourCap" in TechParameters:
        def storageCtr_rule(model, area, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[(area, tech)] > 0:
                # return model.EnergyNbhourCap[area,tech]*model.capacity[area,tech] >= sum(model.energy[area,t,tech] for t in model.TIMESTAMP)
                return model.EnergyNbhourCap[area, tech] >= sum(model.energy[area, t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.AREAS, model.TECHNOLOGIES, rule=storageCtr_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[(area, tech)] < 1.:
                return model.energy[area, t + 1, tech] - model.energy[area, t, tech] <= model.capacity[area, tech] * \
                       model.RampConstraintPlus[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.AREAS, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[area, tech] < 1.:
                return model.energy[area, t + 1, tech] - model.energy[area, t, tech] >= - model.capacity[area, tech] * \
                       model.RampConstraintMoins[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.AREAS, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES,
                                        rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[(area, tech)] < 1.:
                var = (model.energy[area, t + 2, tech] + model.energy[area, t + 3, tech]) / 2 - (
                            model.energy[area, t + 1, tech] + model.energy[area, t, tech]) / 2;
                return var <= model.capacity[area, tech] * model.RampConstraintPlus[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.AREAS, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES,
                                        rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, area, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[(area, tech)] < 1.:
                var = (model.energy[area, t + 2, tech] + model.energy[area, t + 3, tech]) / 2 - (
                            model.energy[area, t + 1, tech] + model.energy[area, t, tech]) / 2;
                return var >= - model.capacity[area, tech] * model.RampConstraintMoins2[area, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.AREAS, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES,
                                         rule=rampCtrMoins2_rule)

    ### Contraintes de rampe
    # C1
    #     subject to rampCtrPlus{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintPlus[a,t]>0 } :
    #         energy[a,h+1,t] - energy[a,h,t] <= capacity[a,t]*RampConstraintPlus[a,t] ;

    # subject to rampCtrMoins{a in AREAS, h in TIMESTAMPMOINS1, t in TECHNOLOGIES : RampConstraintMoins[a,t]>0 } :
    #  energy[a,h+1,t] - energy[a,h,t] >= - capacity[a,t]*RampConstraintMoins[a,t] ;

    #  /*contrainte de rampe2 */
    # subject to rampCtrPlus2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintPlus2[a,t]>0 } :
    #  (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 <= capacity[a,t]*RampConstraintPlus2[a,t] ;

    # subject to rampCtrMoins2{a in AREAS, h in TIMESTAMPMOINS4, t in TECHNOLOGIES : RampConstraintMoins2[a,t]>0 } :
    #   (energy[a,h+2,t]+energy[a,h+3,t])/2 -  (energy[a,h+1,t]+energy[a,h,t])/2 >= - capacity[a,t]*RampConstraintMoins2[a,t] ;

    return model;