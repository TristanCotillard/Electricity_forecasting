import pickle
import numpy as np
import pandas as pd
import time
from pyomo.environ import SolverFactory, value
from functions.f_vm import vm_expand_grid, vm_make_capa_avail, vm_make_margin
from functions.f_graphicalTools import EnergyAndExchange2Prod
from functions.f_optimization import getVariables_panda_indexed, getConstraintsDual_panda
from pyomo_preprocessed import GetElectricSystemModel_Param_Interco_Storage_GestionSingleNode
from update_parameters import linear_create_params, linear_update, linear_create_data, local_linear_create_params, local_linear_create_data, time_local_linear_update, preprocessed_linear_update, preprocessed_local_linear_update


def import_data(year_train, year_test, is_train, solve_model, preprocessing, window_size, step_local_linear, best_iter, is_with_bias_corr, FOLDER_OUTPUT, margin_func, fuel_func, C02_func):
    # Define constants
    solver= 'mosek' ## no need for solverpath with mosek.
    area = 'FR'
    InputFolder='data_elecprices/data/'
    PreprocessedFolder = 'data_elecprices/data_preprocessed/'

    # Not considered means of production
    excluded_productiontype = ['Hydro Run-of-river and poundage','Other','Wind Onshore','Hydro Pumped Storage','Biomass',
                           'Solar','Wind Offshore','Waste','Other renewable','Geothermal','Marine']

    # Import means and sd
    mean_sd = pd.read_csv(PreprocessedFolder + 'mean_sd.csv')

    # Import price and drivers
    da_prices_df = pd.read_csv(PreprocessedFolder + 'obs_prices.csv')

    fuel_prices_df = pd.read_csv(PreprocessedFolder + 'fuel_prices.csv')
    fuel_prices_df.drop('Unnamed: 0', axis='columns', inplace=True)
    fuel_prices_df.ffill(axis = 0, inplace=True)
    
    # Read area consumption and availability factors
    area_consumption_df = pd.read_csv(InputFolder + 'r_areaConsumption_no_phes' + str(year_test) + '_' + str(area) + '.csv')
    area_consumption_df['AREAS'] = area
    avail_factor_df = pd.read_csv(InputFolder + 'r_availabilityFactor' + str(year_test) + '_' + str(area) + '.csv')
    avail_factor_df['AREAS'] = area
    # Read technologies and storage constants
    tech_case='r_article_ramp'
    tech_parameters_df = pd.read_csv(InputFolder+'Gestion_'+tech_case+'_TECHNOLOGIES'+str(year_test)+'.csv')
    tech_parameters_df['AREAS'] = area
    tech_parameters_df.fillna(0, inplace=True)
    storage_parameters_df = pd.read_csv(InputFolder + 'Planing-RAMP1_STOCK_TECHNO_'+str(year_test)+'.csv')
    storage_parameters_df['AREAS'] = area

    # print(da_prices_df, fuel_prices_df, area_consumption_df, avail_factor_df, tech_parameters_df, storage_parameters_df)

    timestamp_df = area_consumption_df.query("AREAS == 'FR'")[['TIMESTAMP','DateTime']]
    timestamp_df['TIMESTAMP_d'] = pd.to_datetime(timestamp_df.DateTime, format='%Y-%m-%dT%H:%M:%SZ')

    if not is_train:
        # read train values when testing and extract timestamp for testing
        area_consumption_df_train = pd.read_csv(InputFolder + 'r_areaConsumption_no_phes' + str(year_train) + '_' + str(area) + '.csv')
        area_consumption_df_train['AREAS'] = area
        timestamp_train_df = area_consumption_df_train.query("AREAS == 'FR'")[['TIMESTAMP','DateTime']]
        timestamp_train_df['TIMESTAMP_d'] = pd.to_datetime(timestamp_train_df.DateTime, format='%Y-%m-%dT%H:%M:%SZ')

    # print(timestamp_df)

    # Use GW to avoid problems with quadratic objective function
    conversion_factor = 1E3
    
    area_consumption_df.areaConsumption = area_consumption_df.areaConsumption / conversion_factor
    tech_parameters_df.capacity = tech_parameters_df.capacity / conversion_factor
    tech_parameters_df.EnergyNbhourCap = tech_parameters_df.EnergyNbhourCap / conversion_factor
    storage_parameters_df.p_max = storage_parameters_df.p_max / conversion_factor
    storage_parameters_df.c_max = storage_parameters_df.c_max / conversion_factor
    storage_parameters_df = storage_parameters_df.set_index(['AREAS','STOCK_TECHNO'])

    selected_techno = sorted(set(tech_parameters_df['TECHNOLOGIES']) - set(excluded_productiontype))
    selected_area =  [area]
    selected_timestamp = sorted(set(area_consumption_df['TIMESTAMP']))

    # Gather
    avail_factor_df.set_index(['AREAS','TIMESTAMP','TECHNOLOGIES'], inplace=True)
    availabilityFactor = vm_expand_grid({"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    availabilityFactor = availabilityFactor.set_index(["AREAS","TIMESTAMP","TECHNOLOGIES"])
    availabilityFactor = availabilityFactor.merge(avail_factor_df, how = 'left', left_index=True, right_index=True)
    availabilityFactor = availabilityFactor.fillna(1) # fix curtailment
    # print(availabilityFactor)

    # Set index for merging
    tech_parameters_df.set_index(['AREAS','TECHNOLOGIES'], inplace=True)
    area_consumption_df.set_index(['AREAS','TIMESTAMP'], inplace =True)
    
    # Read margin preprocessed
    capacity_available_df = vm_make_capa_avail(installed_capa=tech_parameters_df, availability_factor=availabilityFactor)
    margin_df = vm_make_margin(capa_avail_df=capacity_available_df, conso_df=area_consumption_df)
    # print('margin', margin_df)
    all_margin_df = pd.read_csv(PreprocessedFolder + 'margin.csv')
    # print(timestamp_df)
    # print(all_margin_df)
    all_margin_df['TIMESTAMP_d'] = all_margin_df.TIMESTAMP_d.astype('datetime64[ns]')
    margin_df = all_margin_df.merge(timestamp_df, on='TIMESTAMP_d')
    margin_df = margin_df[['margin']]
    # print(margin_df)
    margin_index_df = vm_expand_grid(
        {"AREAS": selected_area, "TIMESTAMP": selected_timestamp})
    margin_index_df = margin_index_df.merge(margin_df, left_index=True, right_index=True)
    margin_index_df.set_index(['AREAS', 'TIMESTAMP'], inplace=True)
    # print(margin_index_df)
    # print('margin_preprocessed', margin_df)

    availabilityFactor = availabilityFactor.loc[(selected_area,slice(None),selected_techno),:]
    TechParameters = tech_parameters_df.loc[(selected_area,selected_techno),:]

    # Fuel price and CO_2 price
    fuel_prices_df['TIMESTAMP_d'] = fuel_prices_df.TIMESTAMP_d.astype('datetime64[ns]')
    fuel_price_df = timestamp_df.merge(fuel_prices_df, how = 'left')[['TIMESTAMP', 'oil_price', 'gas_price', 'coal_price']]
    fuel_price_df = fuel_price_df.rename(columns = {'oil_price':'Fossil Oil','gas_price':'Fossil Gas','coal_price':'Fossil Hard coal'})
    fuel_price_df = pd.melt(fuel_price_df, id_vars=['TIMESTAMP']).rename(columns={'variable' : 'TECHNOLOGIES','value' : 'fuel_price'})
    fuel_price_df.set_index(['TIMESTAMP','TECHNOLOGIES'], inplace= True)
    co2_price_df = timestamp_df.merge(fuel_prices_df, how = 'left')[['TIMESTAMP', 'co2_price']]
    co2_price_df.set_index(['TIMESTAMP'], inplace= True)

    # print('fuel', fuel_price_df)

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
    #print(availabilityFactor_import)

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

    empty_indexed_df = vm_expand_grid({"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    empty_indexed_df = empty_indexed_df.set_index(['AREAS','TIMESTAMP','TECHNOLOGIES'])

    da_prices_df['TIMESTAMP_d'] = da_prices_df.TIMESTAMP_d.astype('datetime64[ns]')
    daPrices_df = da_prices_df.merge(timestamp_df, on='TIMESTAMP_d').set_index(['AREAS','TIMESTAMP'])
    daPrices_df = daPrices_df[['TIMESTAMP_d','price_obs', 'price_trend', 'price_seasonal', 'price_resid']]
    # print('Prix avant boucle', daPrices_df)

    if not is_train:
        daPrices_train_df = da_prices_df.merge(timestamp_train_df, on='TIMESTAMP_d').set_index(['AREAS','TIMESTAMP'])
        daPrices_df['price_trend'] = daPrices_train_df['price_trend']
        daPrices_df['price_seasonal'] = daPrices_train_df['price_seasonal']
    # if test : price_obs: year_test, price_trend: year_train, price_seasonal: year_train, price_resid: year_test
    # if train : price_obs: year_train, price_trend: year_train, price_seasonal: year_train, price_resid: year_train   

    # fuel_price_df
    full_fuel_price_df = empty_indexed_df.merge(fuel_price_df,how='left', left_index=True, right_index=True)
    full_fuel_price_df = full_fuel_price_df.fillna(0)

    # co2_price_df
    full_co2_price_df = empty_indexed_df.merge(co2_price_df,how='left', left_index=True, right_index=True)
    full_co2_price_df = full_co2_price_df.fillna(0)

    # Extract files generated after train
    if not is_train:
        price_param_df_list = pd.read_pickle(FOLDER_OUTPUT + 'r_price_param_df_list'+str(year_train)+'.pkl')
        bias_correction_df_list = pd.read_pickle(FOLDER_OUTPUT + 'r_bias_correction_df_list'+str(year_train)+'.pkl')
        bias_correction_lagr_df_list = pd.read_pickle(FOLDER_OUTPUT + 'r_bias_correction_lagr_df_list'+str(year_train)+'.pkl')

        price_param_df = price_param_df_list[best_iter].copy()
        bias_correction_df = bias_correction_df_list[best_iter].copy()
        bias_correction_lagr_df = bias_correction_lagr_df_list[best_iter].copy()

    # Create model parameters
    if is_train:
        price_param_df_list = list()
        bias_correction_df_list = list()
        bias_correction_lagr_df_list = list()
    if is_train and solve_model == 'linear':
        price_param_df = linear_create_params(selected_techno, selected_area, TechParameters)
    elif is_train and solve_model == 'local_linear':
        price_param_df = local_linear_create_params(selected_techno, selected_area, selected_timestamp, TechParameters)

    # Init margin_factor
    margin_price_df = vm_expand_grid(
        {"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    margin_price_df = margin_price_df.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    margin_price_df = margin_price_df.merge(
        price_param_df[['margin_param']], left_index=True, right_index=True).merge(
        margin_index_df[['margin']], left_index=True, right_index=True) 
    # print('margin price', margin_price_df)

    # Initialize outputs
    model_list = list()
    results_list = list()
    final_merit_order_df_list = list()
    production_df_list = list()
    areaConsumption_list = list()
    duals_df_list = list()

    # Choose number of iteration
    nb_iteration = 1
    if is_train:
        nb_iteration = 4

    # Apply non linearities
    margin_price_df['margin'] = margin_func(margin_price_df['margin'])
    full_fuel_price_df['fuel_price'] = fuel_func(full_fuel_price_df['fuel_price'])
    full_co2_price_df['co2_price'] = C02_func(full_co2_price_df['co2_price'])
    
    for iteration in range(nb_iteration):
        if solve_model == 'linear':
            obj_param_df = linear_create_data(empty_indexed_df, price_param_df, margin_price_df, full_fuel_price_df, full_co2_price_df)
        elif solve_model == 'local_linear':
            obj_param_df = local_linear_create_data(empty_indexed_df, price_param_df, margin_price_df, full_fuel_price_df, full_co2_price_df)
        
        # print(obj_param_df['fuel_price_param'], full_fuel_price_df['fuel_price'])
        # print(obj_param_df['p2'])
        # print(price_param_df)
        
        # Fill na values
        obj_param_df.fillna(0, inplace=True)

        isAbstract = False
        LineEfficiency = 1

        # Résolution du problème d'optimisation
        t = time.time()
        model = GetElectricSystemModel_Param_Interco_Storage_GestionSingleNode(daPrices_df , area_consumption_df,availabilityFactor,TechParameters,storage_parameters_df,
                                                                    obj_param_df=obj_param_df, 
                                                                    mean_sd = mean_sd,
                                                                    # price_param_df=price_param_df, margin_df=margin_df,
                                                                    #empty_indexed_df=empty_indexed_df,
                                                                    availabilityFactor_import=availabilityFactor_import,
                                                                    TechParameters_import=TechParameters_import,
                                                                    import_df=import_df,
                                                                    availabilityFactor_export=availabilityFactor_export,
                                                                    TechParameters_export=TechParameters_export,
                                                                    export_df=export_df,
                                                                    LineEfficiency=LineEfficiency)

        t2 = time.time()
        print('iteration : ', iteration)
        print('model',t2 - t)

        # print(obj_param_df.iloc[:12, :])
        # print(TechParameters_import)

        opt = SolverFactory(solver)
        results = opt.solve(model)
        #results
        elapsed = time.time() - t
        print('solve',elapsed)

        model_list.append(model)
        results_list.append(results)

        if (is_train):
            price_param_df_list.append(price_param_df)

        Variables = getVariables_panda_indexed(model)
        production_df = EnergyAndExchange2Prod(Variables)
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
            order_price_residue=lambda x: x.p0 + x.p1 * x.energy).reorder_levels(
            ['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
        
        # Add price trend and price seasonal to residue 
        all_orders_df['order_price'] = (all_orders_df['order_price_residue'] + daPrices_df['price_trend'] + daPrices_df['price_seasonal']) * mean_sd['price_sd'].values + mean_sd['price_mean'].values
        # qwe2_df => merit_order_df
        # https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
        merit_order_df = all_orders_df.query('energy > 1E-3')
        # print("merit_order before", merit_order_df)
        # # Handle curtailment
        Selected_real_TECHNOLOGIES = selected_techno.copy()
        Selected_real_TECHNOLOGIES.remove('curtailment')
        merit_order_df = merit_order_df.loc[(selected_area, slice(None), Selected_real_TECHNOLOGIES), :]

        merit_order_df = merit_order_df.sort_values('order_price', ascending=False).groupby(['AREAS','TIMESTAMP']).head(1)
        merit_order_df = merit_order_df.reorder_levels(['AREAS','TIMESTAMP','TECHNOLOGIES']).sort_index()
        merit_order_df = merit_order_df.rename(columns={'order_price':'price_sim'})
        # print("merit_order after", merit_order_df)
        # curtailment_marginal_df = merit_order_df.loc[(Selected_AREAS, slice(None), 'curtailment'), :].reset_index()[
        #     ["AREAS", "TIMESTAMP"]]
        # max_besides_curtailment = max(
        #     final_merit_order_df.reset_index()[final_merit_order_df.reset_index().TECHNOLOGIES != "curtailment"].price_sim)

        # qwe3_df =  final_merit_order_df
        final_merit_order_df = merit_order_df
        print(final_merit_order_df)
        # Add bias correction
        if (is_train):
            tmp_1 = final_merit_order_df.merge(
                daPrices_df['price_obs']* mean_sd['price_sd'].values + mean_sd['price_mean'].values, how = 'left', left_index = True, right_index = True)
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

        
        final_merit_order_df_list.append(final_merit_order_df)
        production_df_list.append(production_df)
        areaConsumption_list.append(area_consumption_df)

        if (is_train):
            bias_correction_df_list.append(bias_correction_df)


        for_estim_param_df = final_merit_order_df.merge(daPrices_df['price_resid'], how = 'left', left_index = True, right_index = True)
        for_estim_param_df['intercept'] = 1
        
        print(for_estim_param_df.reset_index()['TECHNOLOGIES'].value_counts())
        print("prix moyen simule : ", final_merit_order_df['price_sim'].mean())
        print(final_merit_order_df)

        extended_for_estim_param_df = all_orders_df.merge(
            daPrices_df['price_resid'], how = 'left', left_index = True, right_index = True).merge(
            availabilityFactor, how = 'left', left_index = True, right_index = True
        )

        extended_for_estim_param_df['price_factor'] = np.where(extended_for_estim_param_df['energy'] > 1E-3,
                                                0.6,
                                                1.0)

        extended_for_estim_param_df['price_reg'] = extended_for_estim_param_df['price_factor'] * extended_for_estim_param_df['price_resid']

        # print(final_merit_order_df, extended_for_estim_param_df)

        # Train model and update parameters
        if is_train and solve_model == 'linear' and not preprocessing:
            print("linear model")
            price_param_df = linear_update(selected_techno, selected_area, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list)
        if is_train and solve_model == 'local_linear' and not preprocessing:
            print('local linear model')
            price_param_df = local_linear_update(selected_techno, selected_area, selected_timestamp, step_local_linear, window_size, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list)
        if is_train and solve_model == 'linear' and preprocessing:
            print('preprocessed linear')
            price_param_df = preprocessed_linear_update(selected_techno, selected_area, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list)
        if is_train and solve_model == 'local_linear' and preprocessing:
            print('preprocessed local linear')
            price_param_df = preprocessed_local_linear_update(selected_techno, selected_area, selected_timestamp, step_local_linear, window_size, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list)

        ### Get Lagrangian ###
        Constraints = getConstraintsDual_panda(model)
        Constraints.keys()
        duals_df = Constraints['energyCtr'].set_index(['AREAS', 'TIMESTAMP'])

        # Handles curtailment
        curtailment_marginal_df = all_orders_df.query('energy > 1E-3')
        curtailment_marginal_df = curtailment_marginal_df.loc[(selected_area, slice(None), selected_techno), :]
        curtailment_marginal_df = curtailment_marginal_df.sort_values('order_price', ascending=False).groupby(['AREAS','TIMESTAMP']).head(1)
        if sum(curtailment_marginal_df.reset_index().TECHNOLOGIES == 'curtailment') > 0:
            curtailment_marginal_df = curtailment_marginal_df.loc[(selected_area, slice(None), 'curtailment'), :].reset_index()[["AREAS", "TIMESTAMP"]]
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
        
    # Save
    with open(FOLDER_OUTPUT + 'r_results_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
        pickle.dump(results_list, f)

    with open(FOLDER_OUTPUT + 'r_final_merit_order_df_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
        pickle.dump(final_merit_order_df_list, f)

    with open(FOLDER_OUTPUT + 'r_production_df_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
        pickle.dump(production_df_list, f)

    with open(FOLDER_OUTPUT + 'r_areaConsumption_list'+str(year_train)+str(year_test)+'.pkl', 'wb') as f:
        pickle.dump(areaConsumption_list, f)

    with open(FOLDER_OUTPUT + 'r_duals_df_list' + str(year_train) + str(year_test) + '.pkl', 'wb') as f:
        pickle.dump(duals_df_list, f)

    with open(FOLDER_OUTPUT + 'r_daPrices_df_' + str(year_test) + '.pkl', 'wb') as f:
        pickle.dump(daPrices_df, f)

    if is_train:
        with open(FOLDER_OUTPUT + 'r_price_param_df_list'+str(year_train)+'.pkl', 'wb') as f:
            pickle.dump(price_param_df_list, f)

        with open(FOLDER_OUTPUT + 'r_bias_correction_df_list'+str(year_train)+'.pkl', 'wb') as f:
            pickle.dump(bias_correction_df_list, f)

        with open(FOLDER_OUTPUT + 'r_bias_correction_lagr_df_list'+str(year_train)+'.pkl', 'wb') as f:
            pickle.dump(bias_correction_lagr_df_list, f)

        with open(FOLDER_OUTPUT + 'r_merit_order_df_'+str(year_train)+'.pkl', 'wb') as f:
            pickle.dump(merit_order_df, f)


def return_best_iter(year_train):
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
    return best_iter

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def id_func(data):
    return data

def squared(data):
    return data**2

def inv(data):
    data[data != 0] = 1/data[data != 0]
    return data

def truncated_log(data):
        data[data > 0] = np.log(data[data > 0])
        data[data <= 0] = 0
        return data

def step_func(threshold):
    def step_func_threshold(data):
        data[data < threshold] = 0
        return data
    return step_func_threshold

start = time.time()
step = 50

# Test with linear
# FOLDER_OUTPUT = 'data_elecprices/output/'
# model = 'linear'
# print(model)
# import_data(2015, 2016, False, model, 0, 0, return_best_iter(2015), True, FOLDER_OUTPUT, id_func, id_func, id_func)

# Train Test with local linear
FOLDER_OUTPUT = 'data_elecprices/output_modified/'
model = 'linear'
step_local_linear = 24 # a week
window_size = step_local_linear//2
preprocessing = True

import_data(2015, 2015, True, model, preprocessing, window_size, step_local_linear, 4, True, FOLDER_OUTPUT, id_func, id_func, id_func)

import_data(2015, 2016, False, model, preprocessing, window_size, step_local_linear, 3, True, FOLDER_OUTPUT, id_func, id_func, id_func)

print(time.time() - start)