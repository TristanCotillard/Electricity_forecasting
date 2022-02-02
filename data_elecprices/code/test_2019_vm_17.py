# As close as possible to R (1100)
# All interco
# Fuel and CO2 params
# Rampes
# Avec storage => Fonctionne mais valeurs à éditer (i.e. PHES, batteries)
# for phes: approximation c_max = 7 * p_max from fig 6 https://ec.europa.eu/jrc/sites/default/files/jrc_20130503_assessment_european_phs_potential.pdf
# Idee: Forcer tous les params hydros à 0 ???

#region importation of modules
import os
import importlib
import pickle

import numpy as np
import pandas as pd
import csv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import datetime
import calendar
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys

import itertools
import time
from sklearn.metrics import mean_squared_error
from scipy.optimize import lsq_linear

from functions.f_operationModels_03 import *
from functions.f_optimization import *
from functions.f_graphicalTools import *
from functions.f_vm import *
# importlib.reload(sys.modules['functions.f_vm'])
#endregion''

#region Import pickle data
da_prices_df = pd.read_pickle('Data/for_pyomo/r_da_prices_df_15_18.pkl')
#gen_per_type_selected_df = pd.read_pickle('Data/for_pyomo/gen_per_type_phes_and_res_df_15_19.pkl')
fuel_prices_df = pd.read_pickle('Data/for_pyomo/r_fuel_prices_df_15_18.pkl')
#exch_physical_selected_df = pd.read_pickle('Data/for_pyomo/exch_physical_selected_df_15_19.pkl')

fuel_prices_df.ffill(axis = 0, inplace=True)

#endregion

#region Solver and data location definition

InputFolder='Data/input/'
for_pyomo_folder='Data/for_pyomo'

if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following to loanch the license server
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64' ### change this to the folder with knitro ampl ...
## in order to obtain more solver see see https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
sys.path.append(BaseSolverPath)
solvers= ['gurobi','knitro','cbc'] # try 'glpk', 'cplex'
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

#region II - Ramp Ctrs Single area : loading parameters loading parameterscase with ramp constraints
Zones=sorted(['FR']); tech_case='r_article_ramp' # 'Simple','RAMP1', 'MultiNode', 'RAMP2', 'r_article'
year_train= 2015; year_test=2016
is_with_bias_corr = True
if year_train == 2015:
    best_iter = 3
elif year_train == 2016:
    best_iter = 4
elif year_train == 2017:
    best_iter = 2
elif year_train == 2018:
    best_iter = 4
elif year_train == 2019:
    best_iter = 4
else:
    sys.exit("case not found!")


excluded_productiontype = ['Hydro Run-of-river and poundage','Other','Wind Onshore','Hydro Pumped Storage','Biomass',
                           'Solar','Wind Offshore','Waste','Other renewable','Geothermal','Marine']

if year_train == year_test:
    is_train = True
else:
    is_train = False

#### reading areaConsumption availabilityFactor and TechParameters CSV files
small_dfs = []
for my_zone in Zones:
    tmp_df = pd.read_csv(InputFolder + 'r_areaConsumption_no_phes' + str(year_test) + '_' + str(my_zone) + '.csv', sep=',',
                                      decimal='.', skiprows=0)
    tmp_df['AREAS'] = my_zone
    small_dfs.append(tmp_df)
areaConsumption = pd.concat(small_dfs, ignore_index=True)

small_dfs = []
for my_zone in Zones:
    tmp_df = pd.read_csv(InputFolder + 'r_availabilityFactor' + str(year_test) + '_' + str(my_zone) + '.csv',
                                         sep=',', decimal='.', skiprows=0)
    tmp_df['AREAS'] = my_zone
    small_dfs.append(tmp_df)
availabilityFactor = pd.concat(small_dfs, ignore_index=True)

if (Zones.__len__() == 1):
    TechParameters = pd.read_csv(InputFolder+'Gestion_'+tech_case+'_TECHNOLOGIES'+str(year_test)+'.csv',sep=',',decimal='.',skiprows=0)
    TechParameters['AREAS'] = Zones[0]
    ExchangeParameters = pd.DataFrame({'empty': []})
    StorageParameters = pd.read_csv(InputFolder + 'Planing-RAMP1_STOCK_TECHNO_'+str(year_test)+'.csv', sep=',', decimal='.',skiprows=0)
    StorageParameters['AREAS'] = Zones[0]
else:
    zones_concat = '-'.join(Zones)
    TechParameters = pd.read_csv(InputFolder+'Gestion_'+tech_case+'_'+zones_concat+'_AREAS_TECHNOLOGIES'+str(year_test)+'.csv',sep=',',decimal='.',skiprows=0)
    ExchangeParameters = pd.read_csv(InputFolder+'Hypothese_'+zones_concat+'_AREAS_AREAS.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["AREAS","AREAS.1"])

if not is_train:
    price_param_df_list = pd.read_pickle('./Data/for_pyomo/r_price_param_df_list'+str(year_train)+'.pkl')
    bias_correction_df_list = pd.read_pickle('./Data/for_pyomo/r_bias_correction_df_list'+str(year_train)+'.pkl')
    bias_correction_lagr_df_list = pd.read_pickle('./Data/for_pyomo/r_bias_correction_lagr_df_list'+str(year_train)+'.pkl')

    price_param_df = price_param_df_list[best_iter].copy()
    bias_correction_df = bias_correction_df_list[best_iter].copy()
    bias_correction_lagr_df = bias_correction_lagr_df_list[best_iter].copy()

#Hypothese_DE-FR_AREAS_AREAS.csv

# Replace timestamp by datetime
timestamp_df = areaConsumption.query("AREAS == 'FR'")[['TIMESTAMP','DateTime']]
timestamp_df['TIMESTAMP_d'] = pd.to_datetime(timestamp_df.DateTime, format='%Y-%m-%dT%H:%M:%SZ')

# # Fix PHES or RES
# # Run vm_agg_gen_per_type.py first
# tmp_phes_and_res = gen_per_type_selected_df.merge(timestamp_df)
# areaConsumption = areaConsumption.merge(tmp_phes_and_res[['AREAS','TIMESTAMP','prod_rel']], how = 'left', right_on = ['AREAS','TIMESTAMP'], left_on = ['AREAS','TIMESTAMP'])
# areaConsumption = areaConsumption.sort_values(by=['AREAS','TIMESTAMP'])
# areaConsumption.fillna(method='ffill', inplace=True)
# areaConsumption = areaConsumption.assign(areaConsumption = lambda x: x.areaConsumption - x.prod_rel)
# del(tmp_phes_and_res)

# Use GW to avoid problems with quadratic objective function
conversion_factor = 1E3

areaConsumption.areaConsumption = areaConsumption.areaConsumption / conversion_factor
TechParameters.capacity = TechParameters.capacity / conversion_factor
TechParameters.EnergyNbhourCap = TechParameters.EnergyNbhourCap / conversion_factor
StorageParameters.p_max = StorageParameters.p_max / conversion_factor
StorageParameters.c_max = StorageParameters.c_max / conversion_factor
if (ExchangeParameters.shape[0] > 0):
    ExchangeParameters.maxExchangeCapacity = ExchangeParameters.maxExchangeCapacity / conversion_factor

# Set indexes
areaConsumption = areaConsumption.set_index(['AREAS','TIMESTAMP'])

availabilityFactor = availabilityFactor.set_index(['AREAS','TIMESTAMP','TECHNOLOGIES'])

TechParameters = TechParameters.set_index(['AREAS','TECHNOLOGIES'])

StorageParameters = StorageParameters.set_index(['AREAS','STOCK_TECHNO'])

Real_AREAS_TECHNOLOGIES = TechParameters.index.values
Real_AREAS_TECHNOLOGIES[0][0]

#### Selection of subset

Selected_AREAS = Zones
Selected_TIMESTAMP = sorted(set(areaConsumption.reset_index()['TIMESTAMP']))
Selected_TECHNOLOGIES = sorted(set(TechParameters.reset_index()['TECHNOLOGIES']) - set(excluded_productiontype))

full_TechParameters = vm_expand_grid({"AREAS": Selected_AREAS, "TECHNOLOGIES": Selected_TECHNOLOGIES})
full_TechParameters = full_TechParameters.set_index(["AREAS","TECHNOLOGIES"])
full_TechParameters = full_TechParameters.merge(TechParameters, how = 'left', left_index=True, right_index=True)
full_TechParameters = full_TechParameters.fillna(0)
TechParameters = full_TechParameters.copy()

full_availabilityFactor = vm_expand_grid({"AREAS": Selected_AREAS, "TIMESTAMP": Selected_TIMESTAMP, "TECHNOLOGIES": Selected_TECHNOLOGIES})
full_availabilityFactor = full_availabilityFactor.set_index(["AREAS","TIMESTAMP","TECHNOLOGIES"])
full_availabilityFactor = full_availabilityFactor.merge(availabilityFactor, how = 'left', left_index=True, right_index=True)
full_availabilityFactor = full_availabilityFactor.fillna(1) # fix curtailment
availabilityFactor = full_availabilityFactor.copy()

capa_avail_df = vm_make_capa_avail(installed_capa=TechParameters, availability_factor=availabilityFactor)
margin_df = vm_make_margin(capa_avail_df=capa_avail_df, conso_df=areaConsumption)
#opex_df = vm_make_opex()
# Run vm_fuel_prices.py first
fuel_price_df = timestamp_df.merge(fuel_prices_df, how = 'left')[['TIMESTAMP', 'oil_price', 'gas_price', 'coal_price']]
fuel_price_df = fuel_price_df.rename(columns = {'oil_price':'Fossil Oil','gas_price':'Fossil Gas','coal_price':'Fossil Hard coal'})
fuel_price_df = pd.melt(fuel_price_df, id_vars=['TIMESTAMP']).rename(columns={'variable' : 'TECHNOLOGIES','value' : 'fuel_price'})
fuel_price_df.set_index(['TIMESTAMP','TECHNOLOGIES'], inplace= True)

co2_price_df = timestamp_df.merge(fuel_prices_df, how = 'left')[['TIMESTAMP', 'co2_price']]
co2_price_df.set_index(['TIMESTAMP'], inplace= True)

availabilityFactor=availabilityFactor.loc[(Selected_AREAS,slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[(Selected_AREAS,Selected_TECHNOLOGIES),:]

#[i[0] for i in Real_AREAS_TECHNOLOGIES]

if (is_train):
    price_param_df = vm_expand_grid({"AREAS": Selected_AREAS, "TECHNOLOGIES": Selected_TECHNOLOGIES})
    price_param_df = price_param_df.set_index(['AREAS', 'TECHNOLOGIES'])
    #price_param_df = TechParameters[[]].copy()
    price_param_df['intercept_param'] = np.NaN
    price_param_df['energy_param'] = 0 # 1
    price_param_df['margin_param'] = 0 # -1
    price_param_df['fuel_price_param'] = 1
    price_param_df['co2_price_param'] = 1

price_param_df[['margin_param']]
margin_df[['margin']]

empty_indexed_df = vm_expand_grid({"AREAS": Selected_AREAS, "TIMESTAMP": Selected_TIMESTAMP, "TECHNOLOGIES": Selected_TECHNOLOGIES})
empty_indexed_df = empty_indexed_df.set_index(['AREAS','TIMESTAMP','TECHNOLOGIES'])

daPrices_df = da_prices_df.merge(
    timestamp_df, on='TIMESTAMP_d').set_index(
    ['AREAS','TIMESTAMP'])
daPrices_df = daPrices_df[['TIMESTAMP_d','price_obs']]

#endregion
#region INTERCO
## Prepare
interco_TechParameters_df = pd.read_csv(InputFolder + 'r_interco_TechParameters_' + str(year_test) + '.csv',
                     sep=',', decimal='.', skiprows=0)

interco_tmp_df = pd.read_csv(InputFolder + 'r_interco_values_' + str(year_test) + '.csv',
                     sep=',', decimal='.', skiprows=0)
interco_tmp_df['TIMESTAMP_d'] = pd.to_datetime(interco_tmp_df.TIMESTAMP_d, format='%Y-%m-%dT%H:%M:%SZ')
interco_tmp_df = timestamp_df[['TIMESTAMP','TIMESTAMP_d']].merge(interco_tmp_df, how = 'left')

# availabilityFactor_import ['AREAS','TIMESTAMP','INTERCOS'] 'availabilityFactor' (availabilityFactor_import)
availabilityFactor_import = interco_tmp_df.copy()
availabilityFactor_import = availabilityFactor_import.rename(columns = {'availabilityFactor_import':'availabilityFactor'})
availabilityFactor_import.set_index(['AREAS','TIMESTAMP','INTERCOS'], inplace = True)
availabilityFactor_import = availabilityFactor_import[['availabilityFactor']]

# availabiltyFactor_export
availabilityFactor_export = interco_tmp_df.copy()
availabilityFactor_export = availabilityFactor_export.rename(columns = {'availabilityFactor_export':'availabilityFactor'})
availabilityFactor_export.set_index(['AREAS','TIMESTAMP','INTERCOS'], inplace = True)
availabilityFactor_export = availabilityFactor_export[['availabilityFactor']]

# import_df ['AREAS','TIMESTAMP','INTERCOS'] 'p0', 'p1' (da_price)
import_df = interco_tmp_df.copy()
import_df = import_df.rename(columns = {'da_price':'p0'})
import_df['p1'] = 0
import_df.set_index(['AREAS','TIMESTAMP','INTERCOS'], inplace = True)
import_df = import_df[['p0','p1']]
# export_df
export_df = import_df.copy()

# TechParameters_import ['AREAS','INTERCOS'] 'capacity' (capacity_import)
TechParameters_import = interco_TechParameters_df.copy()
TechParameters_import = TechParameters_import.rename(columns = {'capacity_import':'capacity'})
TechParameters_import.set_index(['AREAS','INTERCOS'], inplace = True)
TechParameters_import['capacity'] = TechParameters_import['capacity'] / conversion_factor
TechParameters_import = TechParameters_import[['capacity']]
# TechParameters_export
TechParameters_export = interco_TechParameters_df.copy()
TechParameters_export = TechParameters_export.rename(columns = {'capacity_export':'capacity'})
TechParameters_export.set_index(['AREAS','INTERCOS'], inplace = True)
TechParameters_export['capacity'] = TechParameters_export['capacity'] / conversion_factor
TechParameters_export = TechParameters_export[['capacity']]
#endregion

#region II - Ramp Ctrs Single area : solving and loading results
if (is_train):
    # my_range = range(5)
    my_range = range(5)
else:
    my_range = range(1)

iter = 0
for iter in my_range:
    #if price_param_df['intercept_param'].isnull().all():
    if (iter==0):
        # (iter == 0)
        # Init iterator and lists
        model_list = list()
        results_list = list()
        final_merit_order_df_list = list()
        production_df_list = list()
        areaConsumption_list = list()
        duals_df_list = list()

        # fuel_price_df
        full_fuel_price_df = empty_indexed_df.merge(fuel_price_df,how='left', left_index=True, right_index=True)
        full_fuel_price_df = full_fuel_price_df.fillna(0)

        # co2_price_df
        full_co2_price_df = empty_indexed_df.merge(co2_price_df,how='left', left_index=True, right_index=True)
        full_co2_price_df = full_co2_price_df.fillna(0)

        # Init margin_factor
        margin_price_df = vm_expand_grid(
            {"AREAS": Selected_AREAS, "TIMESTAMP": Selected_TIMESTAMP, "TECHNOLOGIES": Selected_TECHNOLOGIES})
        margin_price_df = margin_price_df.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
        margin_price_df = margin_price_df.merge(
            price_param_df[['margin_param']], left_index=True, right_index=True).merge(
            margin_df[['margin']], left_index=True, right_index=True)  # .assign(
        # margin_factor=lambda x: x.margin_param * x.margin)

        if (is_train):
            price_param_df_list = list()
            bias_correction_df_list = list()
            bias_correction_lagr_df_list = list()

            # Init intercept_param
            cols = price_param_df.columns.tolist()
            price_param_df = (price_param_df.merge(
                TechParameters[['energyCost']], left_index=True, right_index=True)).drop(
                columns = 'intercept_param').rename(
                columns={'energyCost':'intercept_param'}
            )
            price_param_df = price_param_df[cols]
            # # if opex price is used
            # price_param_df.loc[('FR','Fossil Gas'),'intercept_param'] = 0
            # price_param_df.loc[('FR', 'Fossil Hard coal'), 'intercept_param'] = 0
            # price_param_df.loc[('FR', 'Fossil Oil'), 'intercept_param'] = 0

            # # price_param_df
            price_param_df = price_param_df.merge(TechParameters[['capacity']], left_index=True, right_index=True)
            price_param_df['energy_param'] = np.where(price_param_df['capacity'] > 0,
                                                                    1 / price_param_df['capacity'],
                                                                    0)
            price_param_df = price_param_df.drop(['capacity'], axis=1)
            # # if opex price is used
            # price_param_df.energy_param = 0

            # fuel_price_param_df
            price_param_df['fuel_price_param'] = 0
            tmp_technologies = ['Fossil Gas','Fossil Hard coal','Fossil Oil']
            price_param_df.loc[(slice(None),tmp_technologies), 'fuel_price_param'] = 1


            # co2_price_param_df
            price_param_df['co2_price_param'] = 0
            price_param_df.loc[(slice(None),'Fossil Hard coal'), 'co2_price_param'] = 0.986 #* conversion_factor
            price_param_df.loc[(slice(None),'Fossil Oil'), 'co2_price_param'] = 0.777 #* conversion_factor
            price_param_df.loc[(slice(None),'Fossil Gas'), 'co2_price_param'] = 0.429 #* conversion_factor
            # price_param_df.loc[(slice(None),'Biomass'), 'co2_price_param'] = 0.494 * conversion_factor


    obj_param_df = empty_indexed_df.merge(
        price_param_df['intercept_param'], how='left', left_index=True, right_index=True).merge(
        price_param_df['margin_param'], how='left', left_index=True, right_index=True).merge(
        margin_price_df['margin'],how='left', left_index=True, right_index=True).merge(
        price_param_df['fuel_price_param'], how='left', left_index=True, right_index=True).merge(
        full_fuel_price_df['fuel_price'], how='left', left_index=True, right_index=True).merge(
        price_param_df['co2_price_param'], how='left', left_index=True, right_index=True).merge(
        full_co2_price_df['co2_price'], how='left', left_index=True, right_index=True).merge(
        price_param_df['energy_param'], how='left', left_index=True, right_index=True).assign(
        p0=lambda x: x.intercept_param + x.margin_param * x.margin + x.fuel_price_param * x.fuel_price + x.co2_price_param * x.co2_price).assign(
        p1=lambda x: x.energy_param)
    obj_param_df = obj_param_df.reorder_levels(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])

    isAbstract = False
    LineEfficiency = 1

    availabilityFactor.loc[('FR', slice(None), 'Fossil Gas')].describe()
    availabilityFactor.loc[('FR', slice(None), 'Fossil Hard coal')].describe()
    availabilityFactor.loc[('FR', slice(None), 'Fossil Oil')].describe()
    availabilityFactor.loc[('FR', slice(None), 'Hydro Water Reservoir')].describe()
    availabilityFactor.loc[('FR', slice(None), 'Nuclear')].describe()
    areaConsumption.describe()
    TechParameters
    obj_param_df.groupby('TECHNOLOGIES').describe()['intercept_param']
    obj_param_df.groupby('TECHNOLOGIES').describe()['margin_param']
    obj_param_df.groupby('TECHNOLOGIES').describe()['margin']
    obj_param_df.groupby('TECHNOLOGIES').describe()['fuel_price_param']
    obj_param_df.groupby('TECHNOLOGIES').describe()['fuel_price']
    obj_param_df.groupby('TECHNOLOGIES').describe()['co2_price_param']
    obj_param_df.groupby('TECHNOLOGIES').describe()['co2_price']
    obj_param_df.groupby('TECHNOLOGIES').describe()['energy_param']
    obj_param_df.groupby('TECHNOLOGIES').describe()['p0']
    obj_param_df.groupby('TECHNOLOGIES').describe()['p1']
    # availabilityFactor_interco.groupby('INTERCOS').describe()
    # TechParameters_interco
    # interco_df.groupby('INTERCOS').describe()

    t = time.time()
    model = GetElectricSystemModel_Param_Interco_Storage_GestionSingleNode(areaConsumption,availabilityFactor,TechParameters,StorageParameters,
                                                                   obj_param_df=obj_param_df,
                                                                   # price_param_df=price_param_df, margin_df=margin_df,
                                                                   #empty_indexed_df=empty_indexed_df,
                                                                   availabilityFactor_import=availabilityFactor_import,
                                                                   TechParameters_import=TechParameters_import,
                                                                   import_df=import_df,
                                                                   availabilityFactor_export=availabilityFactor_export,
                                                                   TechParameters_export=TechParameters_export,
                                                                   export_df=export_df,
                                                                   ExchangeParameters=ExchangeParameters,
                                                                   LineEfficiency=LineEfficiency)

    opt = SolverFactory(solver)
    results=opt.solve(model)
    results
    elapsed = time.time() - t
    print(elapsed)

    model_list.append(model)
    results_list.append(results)

    if (is_train):
        price_param_df_list.append(price_param_df)

    Variables=getVariables_panda_indexed(model)
    production_df=EnergyAndExchange2Prod(Variables)
    #production_df=production_df.reset_index().set_index(['AREAS', 'TIMESTAMP'])
    value(model.OBJ)/1000000000 # Cout total en milliards d'euros

    # # Get ICO
    # production_interco_df = Variables['energy_interco'].pivot(index=["AREAS", "TIMESTAMP"], columns='INTERCOS', values='energy_interco')
    # production_df = production_df.merge(production_interco_df, how='left', left_index=True, right_index=True)

    # https://stackoverflow.com/questions/60620691/how-can-i-export-results-from-a-pyomo-variable-to-a-pandas-dataframe-and-excel
    t = time.time()

    # all_areas = ['BE','DE','FR']
    all_areas = ['FR']

    production_df_melted = production_df.drop(all_areas, axis=1).melt(
        ignore_index=False, var_name = "TECHNOLOGIES", value_name="energy")
    production_df_melted = production_df_melted.reset_index()
    production_df_melted.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'], inplace=True)

    production_all_df_melted = production_df.melt(
        ignore_index=False, var_name = "TECHNOLOGIES", value_name="energy")
    production_all_df_melted = production_all_df_melted.reset_index()
    production_all_df_melted.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'], inplace=True)

    # qwe_df == obj_param_df with order price == all_orders_df

    # # !!! ONLY IF INTERNATIONAL PRICES KNOWN !!! #
    # orders_ico_df = Variables['energy_interco'].set_index(['AREAS', 'TIMESTAMP', 'INTERCOS']).merge(
    #     interco_df, how='left', left_index=True, right_index=True).rename_axis(
    #     index=['AREAS', 'TIMESTAMP', 'TECHNOLOGIES']).rename(
    #     columns={"energy_interco": "energy"})
    #
    # all_orders_df = obj_param_df.merge(
    #     production_df_melted[['energy']], how='left', left_index=True, right_index=True).append(
    #     orders_ico_df).assign(
    #     order_price=lambda x: x.p0 + x.p1 * x.energy).reorder_levels(
    #     ['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])

    all_orders_df = obj_param_df.merge(
        production_df_melted[['energy']], how='left', left_index=True, right_index=True).assign(
        order_price=lambda x: x.p0 + x.p1 * x.energy).reorder_levels(
        ['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])

    # qwe2_df => merit_order_df
    # https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
    merit_order_df = all_orders_df.query('energy > 1E-3')

    # # Handle curtailment
    Selected_real_TECHNOLOGIES = Selected_TECHNOLOGIES.copy()
    Selected_real_TECHNOLOGIES.remove('curtailment')
    merit_order_df = merit_order_df.loc[(Selected_AREAS, slice(None), Selected_real_TECHNOLOGIES), :]

    merit_order_df = merit_order_df.sort_values('order_price', ascending=False).groupby(['AREAS','TIMESTAMP']).head(1)
    merit_order_df = merit_order_df.reorder_levels(['AREAS','TIMESTAMP','TECHNOLOGIES']).sort_index()
    merit_order_df = merit_order_df.rename(columns={'order_price':'price_sim'})

    # curtailment_marginal_df = merit_order_df.loc[(Selected_AREAS, slice(None), 'curtailment'), :].reset_index()[
    #     ["AREAS", "TIMESTAMP"]]
    # max_besides_curtailment = max(
    #     final_merit_order_df.reset_index()[final_merit_order_df.reset_index().TECHNOLOGIES != "curtailment"].price_sim)

    # qwe3_df =  final_merit_order_df
    if (Zones.__len__() > 1):
        # Marginal type for zone A can come from zone B
        import_ener_from_df = production_df[Selected_AREAS] > 0
        import_ener_from_df = pd.melt(import_ener_from_df, ignore_index=False, var_name='eligible_prod_AREAS', value_name='import_b')
        import_ener_from_df = import_ener_from_df.query('import_b').drop('import_b', axis = 1)

        valid_areas_df = vm_expand_grid({"AREAS": Selected_AREAS, "TIMESTAMP": Selected_TIMESTAMP})
        valid_areas_df['eligible_prod_AREAS'] = valid_areas_df.AREAS
        valid_areas_df = valid_areas_df.set_index(['AREAS', 'TIMESTAMP'])
        valid_areas_df = pd.concat([valid_areas_df, import_ener_from_df])


        l_df = valid_areas_df.reset_index()
        r_df = merit_order_df.reset_index().rename(columns={'AREAS':'eligible_prod_AREAS'})

        final_merit_order_df = pd.merge(l_df, r_df, how='left',
                           left_on= ['eligible_prod_AREAS', 'TIMESTAMP'],
                           right_on=['eligible_prod_AREAS', 'TIMESTAMP'])

        final_merit_order_df = final_merit_order_df.set_index(['AREAS','TIMESTAMP'])

        final_merit_order_df = final_merit_order_df.dropna(subset = ['price_sim']).sort_values('price_sim', ascending=False).groupby(['AREAS','TIMESTAMP']).head(1)
        final_merit_order_df = final_merit_order_df.reorder_levels(['AREAS','TIMESTAMP']).sort_index()
        final_merit_order_df = final_merit_order_df.rename(columns={'eligible_prod_AREAS':'marginal_AREAS'})
    else:
        final_merit_order_df = merit_order_df

    # Add bias correction
    if (is_train):
        tmp_1 = final_merit_order_df.merge(
            daPrices_df['price_obs'], how = 'left', left_index = True, right_index = True)
        tmp_1['delta_price'] = tmp_1['price_obs'] - tmp_1['price_sim']

        timestamp_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
        timestamp_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday

        tmp_2 = tmp_1[['price_sim','price_obs','delta_price']].merge(
            timestamp_df.set_index(['TIMESTAMP']), how='left', left_index=True, right_index=True)

        bias_correction_df = tmp_2.reset_index().groupby(['AREAS','hour','weekday'])[['delta_price']].mean()

        bias_correction_ts_df = bias_correction_df.merge(
            timestamp_df.set_index(['hour','weekday'])['TIMESTAMP'], how='left', left_index=True, right_index=True).reset_index(
            ).set_index(['AREAS','TIMESTAMP'])[['delta_price']].sort_index()

    final_merit_order_df = final_merit_order_df.reset_index()
    final_merit_order_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
    final_merit_order_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday

    if is_with_bias_corr:
        final_merit_order_df = final_merit_order_df.merge(
            bias_correction_df.reset_index()[['AREAS','hour','weekday','delta_price']], how = 'left', on = ['hour','weekday','AREAS']).assign(
            price_sim =lambda x: x.price_sim + x.delta_price).drop(
            columns = ['hour','weekday','delta_price']).set_index(
            ['AREAS','TIMESTAMP','TECHNOLOGIES'])
    else:
        final_merit_order_df = final_merit_order_df.set_index(
            ['AREAS','TIMESTAMP','TECHNOLOGIES'])

    # final_merit_order_df = final_merit_order_df.merge(
    #         bias_correction_ts_df['delta_price'], how = 'left', left_index = True, right_index = True).assign(
    #         price_sim =lambda x: x.price_sim + x.delta_price).drop(columns = ['delta_price'])

    final_merit_order_df_list.append(final_merit_order_df)
    production_df_list.append(production_df)
    areaConsumption_list.append(areaConsumption)

    if (is_train):
        bias_correction_df_list.append(bias_correction_df)

    #qw3_df.query('AREAS == "FR" and TIMESTAMP == 152')
    #
    # elapsed = time.time() - t
    # print(elapsed)

    for_estim_param_df = final_merit_order_df.merge(daPrices_df['price_obs'], how = 'left', left_index = True, right_index = True)
    for_estim_param_df['intercept'] = 1
    # for_estim_param_df = for_estim_param_df.merge(margin_df['margin'], how = 'left', left_index = True, right_index = True)
    # for_estim_param_df = for_estim_param_df.merge(TechParameters['capacity'], how = 'left', left_index = True, right_index = True)

    # for_estim_param_df['techno_scarc'] = np.where(for_estim_param_df['capacity'] > 0,
    #                                              for_estim_param_df['energy'] / for_estim_param_df['capacity'],
    #                                              0)
    #for_estim_param_df = for_estim_param_df.assign(techno_scarc = lambda x: vm_non_zero_division(x.energy,x.capacity))

    if (Zones.__len__() > 1):
        for_estim_param_df = for_estim_param_df[['marginal_AREAS','TECHNOLOGIES','price_obs','intercept','margin','techno_scarc']]

        #for_estim_param_df = for_estim_param_df.reorder_levels(['AREAS','TIMESTAMP','TECHNOLOGIES']).sort_index()
        for_estim_param_df = for_estim_param_df.reset_index().drop('AREAS', axis = 1).rename(columns={'marginal_AREAS':'AREAS'})

    print(for_estim_param_df.reset_index()['TECHNOLOGIES'].value_counts())
    print(final_merit_order_df['price_sim'].mean())

    extended_for_estim_param_df = all_orders_df.merge(
        daPrices_df['price_obs'], how = 'left', left_index = True, right_index = True).merge(
        availabilityFactor, how = 'left', left_index = True, right_index = True
    )

    extended_for_estim_param_df['price_factor'] = np.where(extended_for_estim_param_df['energy'] > 1E-3,
                                              0.6,
                                              1.0)

    extended_for_estim_param_df['price_reg'] = extended_for_estim_param_df['price_factor'] * extended_for_estim_param_df['price_obs']

    if (is_train):
        if (is_train):
            my_Selected_TECHNOLOGIES = Selected_TECHNOLOGIES.copy()
            my_Selected_TECHNOLOGIES.remove('curtailment')
            param_dict = {}
            my_variables = ['intercept', 'energy', 'margin', 'fuel_price', 'co2_price']
            param_names = [f'{i}_param' for i in my_variables]
            nb_min_rows = 5
            for my_area in Selected_AREAS:
                for my_tech in my_Selected_TECHNOLOGIES:
                    lb = [0,      0,     -np.inf, 0,      0]
                    ub = [np.inf, np.inf,      0,       np.inf, 0.00001] # 0.00001
                    # special values for co2_prices
                    if (my_tech == 'Fossil Hard coal'):
                        lb[4] = 0.986  # tCO2/MWh
                        ub[4] = lb[4] + 0.00001
                    if (my_tech == 'Fossil Oil'):
                        lb[4] = 0.777
                        ub[4] = lb[4] + 0.00001
                    if (my_tech == 'Fossil Gas'):
                        lb[4] = 0.429
                        ub[4] = lb[4] + 0.00001
                    if (my_tech == 'Biomass'):
                        lb[4] = 0.494
                        ub[4] = lb[4] + 0.001
                    # if (my_tech == 'Hydro Water Reservoir'):
                    #     lb[4] = 0.
                    #     ub[4] = lb[4] + 0.001
                    # if (my_tech == 'Nuclear'):
                    #     lb[4] = 0.
                    #     ub[4] = lb[4] + 0.001
                    # if (my_tech == 'curtailment'):
                    #     lb[4] = 0.
                    #     ub[4] = lb[4] + 0.001
                    tmp_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
                    if (tmp_df.shape[0] >= nb_min_rows):
                        A = tmp_df[my_variables].values
                        b = tmp_df['price_obs'].values
                        calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                        param_dict[(my_area, my_tech)] = calib.x
                    # elif (tmp_df.shape[0] > 0 and tmp_df.shape[0] < nb_min_rows):
                    #     tmp_mean_price = np.mean(tmp_df['price_obs'].values)
                    #     param_dict[(my_area, my_tech)] = np.array([tmp_mean_price, 0, 0, 0, lb[4]])
                    # else:
                    #     tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
                    #     tmp_mean_price = np.mean(tmp2_df['price_reg'].values)
                    #     param_dict[(my_area, my_tech)] = np.array([tmp_mean_price, 0, 0, 0, lb[4]])
                    else:
                        tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech').copy()
                        tmp2_df['intercept'] = 1
                        A = tmp2_df[my_variables].values
                        b = tmp2_df['price_reg'].values
                        calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                        param_dict[(my_area, my_tech)] = calib.x
                    # else:
                    #     # Keep previous value
                    #     param_dict[(my_area, my_tech)] = np.array(price_param_df.loc[(my_area, my_tech), param_names])


        asd = pd.DataFrame.from_dict(param_dict, orient='index')
        asd.index = pd.MultiIndex.from_tuples(asd.index.values, names=['AREAS', 'TECHNOLOGIES'])
        asd.columns = param_names

        cols = asd.columns.tolist()
        zxc = asd.combine_first(price_param_df_list[iter])[cols]

        price_param_df = zxc
        #end iter

    ### Get Lagrangian ###
    Constraints = getConstraintsDual_panda(model)
    Constraints.keys()
    duals_df = Constraints['energyCtr'].set_index(['AREAS', 'TIMESTAMP'])

    # Handles curtailment
    curtailment_marginal_df = all_orders_df.query('energy > 1E-3')
    curtailment_marginal_df = curtailment_marginal_df.loc[(Selected_AREAS, slice(None), Selected_TECHNOLOGIES), :]
    curtailment_marginal_df = curtailment_marginal_df.sort_values('order_price', ascending=False).groupby(['AREAS','TIMESTAMP']).head(1)
    if sum(curtailment_marginal_df.reset_index().TECHNOLOGIES == 'curtailment') > 0:
        curtailment_marginal_df = curtailment_marginal_df.loc[(Selected_AREAS, slice(None), 'curtailment'), :].reset_index()[["AREAS", "TIMESTAMP"]]
        curtailment_marginal_df = curtailment_marginal_df.set_index(["AREAS", "TIMESTAMP"])
        # merge merit_order_df
        curtailment_marginal_df = curtailment_marginal_df.merge(merit_order_df, how='left', left_index=True, right_index=True)[['price_sim']]
        curtailment_marginal_df = curtailment_marginal_df.reset_index()[["AREAS", "TIMESTAMP", "price_sim"]].set_index(["AREAS", "TIMESTAMP"])
        #curtailment_marginal_df = curtailment_marginal_df.rename(columns = {'energyCtr': 'price_sim'})
        duals_df = duals_df.merge(curtailment_marginal_df, how='left', left_index=True, right_index=True)
        duals_df['energyCtr'] = np.where(duals_df['price_sim'].isnull(),
                                           duals_df['energyCtr'],
                                           duals_df['price_sim'])
        duals_df = duals_df[['energyCtr']]

    # mean_energyCtr = duals_df.groupby(['AREAS'])[['energyCtr']].mean()
    #
    # # rampCtrPlus = -rampCtrMoins
    # # On compte la somme des abs contraintes de rampes sur les technos pour chaque instant t
    # # On ajoute cette cette contrainte au prix si energyCtr>mean(energyCtr), sinon on soustrait
    # tmp = Constraints['rampCtrMoins']
    # tmp0 = tmp.query('TIMESTAMP_MinusOne == 1').copy()
    # tmp0['TIMESTAMP_MinusOne'] = 0
    # tmp = pd.concat([tmp0,tmp])
    # tmp['TIMESTAMP_MinusOne'] = tmp['TIMESTAMP_MinusOne'] + 1
    # tmp = tmp.rename(columns = {'TIMESTAMP_MinusOne': 'TIMESTAMP', 'rampCtrMoins': 'rampCtr'})
    # tmp['rampCtr'] = abs(tmp['rampCtr'])
    # tmp = tmp.groupby(['AREAS','TIMESTAMP'])[['rampCtr']].sum()
    #
    # duals_df = duals_df.merge(
    #         tmp, how = 'left', left_index = True, right_index = True)
    #
    # tmp_mean_energyCtr = duals_df[[]].merge(
    #         mean_energyCtr, how = 'left', left_index = True, right_index = True)
    #
    # duals_df['rampCtr_rel'] = np.where(duals_df['energyCtr'] > tmp_mean_energyCtr['energyCtr'],
    #                                               duals_df['rampCtr'],
    #                                               -duals_df['rampCtr'])
    #
    # #'storageCtr' # à répartir au prorata de la prod hydro
    # prod_hydro_sum = production_df.groupby(['AREAS'])[['Hydro Water Reservoir']].sum().rename(
    #     columns = {'Hydro Water Reservoir': 'prod_hydro_tot'}
    # )
    # prod_hydro_rel_df = production_df[['Hydro Water Reservoir']].merge(
    #     prod_hydro_sum, how = 'left', left_index = True, right_index = True).rename(
    #     columns = {'Hydro Water Reservoir': 'prod_hydro_t'}).assign(
    #     prod_hydro_rel = lambda x: x.prod_hydro_t / x.prod_hydro_tot)
    #
    # tmp_storage_ctr = Constraints['storageCtr'].set_index('AREAS')[['storageCtr']]
    # prod_hydro_rel_df = prod_hydro_rel_df[['prod_hydro_rel']].merge(
    #     tmp_storage_ctr, how = 'left', left_index = True, right_index = True).assign(
    #     storageCtr = lambda x: x.storageCtr * x.prod_hydro_rel)
    #
    # duals_df = duals_df.merge(
    #         prod_hydro_rel_df[['storageCtr']], how = 'left', left_index = True, right_index = True)
    #
    # duals_df = duals_df.assign(
    #     price_sim_lagr = lambda x: x.energyCtr + x.rampCtr_rel + x.storageCtr)

    duals_df = duals_df.assign(
        price_sim_lagr = lambda x: x.energyCtr)

    # Add bias correction duals
    if (is_train):
        tmp_1 = duals_df.merge(
            daPrices_df['price_obs'], how = 'left', left_index = True, right_index = True)
        tmp_1['delta_price'] = tmp_1['price_obs'] - tmp_1['price_sim_lagr']

        timestamp_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
        timestamp_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday

        tmp_2 = tmp_1[['price_sim_lagr','price_obs','delta_price']].merge(
            timestamp_df.set_index(['TIMESTAMP']), how='left', left_index=True, right_index=True)

        bias_correction_lagr_df = tmp_2.reset_index().groupby(['AREAS','hour','weekday'])[['delta_price']].mean()

        bias_correction_lagr_ts_df = bias_correction_lagr_df.merge(
            timestamp_df.set_index(['hour','weekday'])['TIMESTAMP'], how='left', left_index=True, right_index=True).reset_index(
            ).set_index(['AREAS','TIMESTAMP'])[['delta_price']].sort_index()

    duals_df = duals_df.reset_index()
    duals_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
    duals_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday

    if is_with_bias_corr:
        duals_df = duals_df.merge(
            bias_correction_lagr_df.reset_index()[['AREAS','hour','weekday','delta_price']], how = 'left', on = ['hour','weekday','AREAS']).assign(
            price_sim_lagr = lambda x: x.price_sim_lagr + x.delta_price).drop(
            columns = ['hour','weekday','delta_price']).set_index(
            ['AREAS','TIMESTAMP'])

    duals_df_list.append(duals_df)

    if (is_train):
        bias_correction_lagr_df_list.append(bias_correction_lagr_df)
# endregion

# iter = 1
# model = model_list[iter]
#
# Constraints = getConstraintsDual_panda(model)
# Constraints.keys()
# energyCtrDual = c
# round(energyCtrDual.energyCtr,2).unique()
#
# storageCtrDual=Constraints['storageCtr']
# round(storageCtrDual.storageCtr,2).unique()
#
# rampCtrPlusDual=Constraints['rampCtrPlus']
# round(rampCtrPlusDual.rampCtrPlus,2).unique()
#
# rampCtrMoinsDual=Constraints['rampCtrMoins']
# round(rampCtrMoinsDual.rampCtrMoins,2).unique()
#
# # DF duals
# Constraints['energyCtr'].set_index(['AREAS', 'TIMESTAMP'])
# Constraints['rampCtrMoins'].set_index(['AREAS', 'TIMESTAMP'])
# Constraints['rampCtrPlus'].set_index(['AREAS', 'TIMESTAMP'])

# Save
with open('./Data/for_pyomo/r_results_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
    pickle.dump(results_list, f)

with open('./Data/for_pyomo/r_final_merit_order_df_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
    pickle.dump(final_merit_order_df_list, f)

with open('./Data/for_pyomo/r_production_df_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
    pickle.dump(production_df_list, f)

with open('./Data/for_pyomo/r_areaConsumption_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
    pickle.dump(areaConsumption_list, f)

with open('./Data/for_pyomo/r_duals_df_list' + str(year_train) + str(year_test) + '.pkl', 'wb') as f:
    pickle.dump(duals_df_list, f)

with open('./Data/for_pyomo/r_daPrices_df_' + str(year_test) + '.pkl', 'wb') as f:
    pickle.dump(daPrices_df, f)

if is_train:
    with open('./Data/for_pyomo/r_price_param_df_list'+str(year_train)+'.pkl', 'wb') as f:
        pickle.dump(price_param_df_list, f)

    with open('./Data/for_pyomo/r_bias_correction_df_list'+str(year_train)+'.pkl', 'wb') as f:
        pickle.dump(bias_correction_df_list, f)

    with open('./Data/for_pyomo/r_bias_correction_lagr_df_list'+str(year_train)+'.pkl', 'wb') as f:
        pickle.dump(bias_correction_lagr_df_list, f)

    with open('./Data/for_pyomo/r_merit_order_df_'+str(year_train)+'.pkl', 'wb') as f:
        pickle.dump(merit_order_df, f)
