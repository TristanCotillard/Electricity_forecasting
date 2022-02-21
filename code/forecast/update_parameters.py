import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from functions.f_vm import vm_expand_grid
import time

from drivers_nn import find_driver_nn


def linear_create_params(selected_techno, selected_area, TechParameters):
    """
    initialize linear regression parameters (one value per year)
    """
    
    price_param_df = vm_expand_grid({"AREAS": selected_area, "TECHNOLOGIES": selected_techno})
    price_param_df = price_param_df.set_index(['AREAS', 'TECHNOLOGIES'])
    # price_param_df = TechParameters[[]].copy()
    price_param_df['intercept_param'] = np.NaN
    price_param_df['energy_param'] = 0 # 1
    price_param_df['margin_param'] = 0 # -1
    price_param_df['fuel_price_param'] = 1
    price_param_df['co2_price_param'] = 1

    # Init intercept_param
    cols = price_param_df.columns.tolist()
    price_param_df = (price_param_df.merge(
        TechParameters[['energyCost']], left_index=True, right_index=True)).drop(
        columns = 'intercept_param').rename(
        columns={'energyCost':'intercept_param'}
    )
    price_param_df['intercept_param'] = 0
    price_param_df = price_param_df[cols]

    # price_param_df
    price_param_df = price_param_df.merge(TechParameters[['capacity']], left_index=True, right_index=True)
    price_param_df['energy_param'] = np.where(price_param_df['capacity'] > 0, 1 / price_param_df['capacity'], 0)
    price_param_df = price_param_df.drop(['capacity'], axis=1)

    # fuel_price_param_df
    price_param_df['fuel_price_param'] = 0
    tmp_technologies = ['Fossil Gas','Fossil Hard coal','Fossil Oil']
    price_param_df.loc[(slice(None),tmp_technologies), 'fuel_price_param'] = 1

    # co2_price_param_df
    price_param_df['co2_price_param'] = 0
    price_param_df.loc[(slice(None),'Fossil Hard coal'), 'co2_price_param'] = 0.986 #* conversion_factor
    price_param_df.loc[(slice(None),'Fossil Oil'), 'co2_price_param'] = 0.777 #* conversion_factor
    price_param_df.loc[(slice(None),'Fossil Gas'), 'co2_price_param'] = 0.429 #* conversion_factor

    return price_param_df

def linear_create_data(empty_indexed_df, price_param_df, margin_price_df, full_fuel_price_df, full_co2_price_df):
    """
    return parameters and drivers values in case of linear regression
    """
    
    obj_param_df = empty_indexed_df.merge(
            price_param_df['intercept_param'], how='left', left_index=True, right_index=True).merge(
            price_param_df['margin_param'], how='left', left_index=True, right_index=True).merge(
            margin_price_df['margin'],how='left', left_index=True, right_index=True).merge(
            price_param_df['fuel_price_param'], how='left', left_index=True, right_index=True).merge(
            full_fuel_price_df['fuel_price'], how='left', left_index=True, right_index=True).merge(
            price_param_df['co2_price_param'], how='left', left_index=True, right_index=True).merge(
            full_co2_price_df['co2_price'], how='left', left_index=True, right_index=True).merge(
            # price_param_df['unavail_power_param'], how='left', left_index=True, right_index=True).merge(
            # unplanned['unavail_power'], how='left', left_index=True, right_index=True).merge(
            price_param_df['energy_param'], how='left', left_index=True, right_index=True).assign(
            p0=lambda x: x.intercept_param + x.margin_param * x.margin + x.fuel_price_param * x.fuel_price + x.co2_price_param * x.co2_price).assign(
            p1=lambda x: x.energy_param)
    obj_param_df = obj_param_df.reorder_levels(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    
    return obj_param_df


def linear_update(selected_techno, selected_area, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list):
    """
    Update parameters values with linear regressions on observed prices
    """
    
    my_Selected_TECHNOLOGIES = selected_techno.copy()
    my_Selected_TECHNOLOGIES.remove('curtailment')
    param_dict = {}
    my_variables = ['intercept', 'energy', 'margin', 'fuel_price', 'co2_price']
    param_names = [f'{i}_param' for i in my_variables]
    nb_min_rows = 5
    for my_area in selected_area:
        for my_tech in my_Selected_TECHNOLOGIES:
            # lower and upper bounds of parameters
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
            
            tmp_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
            
            if (tmp_df.shape[0] >= nb_min_rows):
                A = tmp_df[my_variables].values
                b = tmp_df['price_obs'].values
                calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                param_dict[(my_area, my_tech)] = calib.x
            else:
                tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech').copy()
                tmp2_df['intercept'] = 1
                A = tmp2_df[my_variables].values
                b = tmp2_df['price_reg'].values
                calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                param_dict[(my_area, my_tech)] = calib.x

    asd = pd.DataFrame.from_dict(param_dict, orient='index')
    asd.index = pd.MultiIndex.from_tuples(asd.index.values, names=['AREAS', 'TECHNOLOGIES'])
    asd.columns = param_names

    cols = asd.columns.tolist()
    zxc = asd.combine_first(price_param_df_list[iteration])[cols]

    price_param_df = zxc

    return price_param_df

def preprocessed_linear_update(selected_techno, selected_area, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list):
    """
    Update parameters values with linear regressions on observed residual prices
    """
    
    my_Selected_TECHNOLOGIES = selected_techno.copy()
    my_Selected_TECHNOLOGIES.remove('curtailment')
    param_dict = {}
    my_variables = ['intercept', 'energy', 'margin', 'fuel_price', 'co2_price']
    param_names = [f'{i}_param' for i in my_variables]
    nb_min_rows = 5
    for my_area in selected_area:
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
            
            tmp_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
            
            if (tmp_df.shape[0] >= nb_min_rows):
                A = tmp_df[my_variables].values
                b = tmp_df['price_trend'].values
                calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                param_dict[(my_area, my_tech)] = calib.x
            else:
                tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech').copy()
                tmp2_df['intercept'] = 1
                A = tmp2_df[my_variables].values
                b = tmp2_df['price_reg'].values
                calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                param_dict[(my_area, my_tech)] = calib.x

    asd = pd.DataFrame.from_dict(param_dict, orient='index')
    asd.index = pd.MultiIndex.from_tuples(asd.index.values, names=['AREAS', 'TECHNOLOGIES'])
    asd.columns = param_names

    cols = asd.columns.tolist()
    zxc = asd.combine_first(price_param_df_list[iteration])[cols]

    price_param_df = zxc

    return price_param_df


def local_linear_create_params(selected_techno, selected_area, selected_timestamp, TechParameters):
    """
    initialize parameters (a value per timestamp)
    """
    
    price_param_df = vm_expand_grid({"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    price_param_df = price_param_df.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])

    price_param_df['energy_param'] = 0
    price_param_df['margin_param'] = 0
    price_param_df['fuel_price_param'] = 1
    price_param_df['co2_price_param'] = 1
    
    # Init intercept_param
    # print(TechParameters['energyCost'])
    price_param_df = price_param_df.merge( 
        TechParameters[['energyCost']], left_index=True, right_index=True).rename(
            columns={'energyCost':'intercept_param'})

    # price_param_df
    price_param_df = price_param_df.merge(TechParameters[['capacity']], left_index=True, right_index=True)
    price_param_df['energy_param'] = np.where(price_param_df['capacity'] > 0, 1 / price_param_df['capacity'], 0)
    price_param_df = price_param_df.drop(['capacity'], axis=1)

    # fuel_price_param_df
    price_param_df['fuel_price_param'] = 0
    tmp_technologies = ['Fossil Gas','Fossil Hard coal','Fossil Oil']
    price_param_df.loc[(slice(None),tmp_technologies), 'fuel_price_param'] = 1

    # co2_price_param_df
    price_param_df['co2_price_param'] = 0
    price_param_df.loc[(slice(None),'Fossil Hard coal'), 'co2_price_param'] = 0.986 #* conversion_factor
    price_param_df.loc[(slice(None),'Fossil Oil'), 'co2_price_param'] = 0.777 #* conversion_factor
    price_param_df.loc[(slice(None),'Fossil Gas'), 'co2_price_param'] = 0.429 #* conversion_factor

    return price_param_df

def local_linear_create_data(empty_indexed_df, price_param_df, margin_price_df, full_fuel_price_df, full_co2_price_df):
    """
    return parameters and drivers values adapted with local linear model
    """

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
    return obj_param_df


def time_local_linear_update(selected_techno, selected_area, selected_timestamp, step, window_size, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list, drivers):
    """
    update parameters with time local linear model (linear regression with a fixed window)
    """
    
    my_Selected_TECHNOLOGIES = selected_techno.copy()
    my_Selected_TECHNOLOGIES.remove('curtailment')
    selected_techno = my_Selected_TECHNOLOGIES
    
    param_dict = {}
    my_variables = ['intercept', 'energy', 'margin', 'fuel_price', 'co2_price']
    param_names = [f'{i}_param' for i in my_variables]
    nb_min_rows = 5
    t = time.time()
    for my_area in selected_area:
        for rg_time in range(0, len(selected_timestamp), step):
            my_time = selected_timestamp[rg_time]
            for my_tech in selected_techno:
                lb = [0, 0, -np.inf, 0, 0]
                ub = [np.inf, np.inf, 0, np.inf, 0.00001]
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
                
                # for time nearest neighbours
                tmp_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech and @my_time - @window_size <= TIMESTAMP and TIMESTAMP <= @my_time + @window_size')
                # tmp_df = tmp_df[ my_time - window_size <= for_estim_param_df['TIMESTAMP'] <= my_time + window_size ]
                
                if (tmp_df.shape[0] >= nb_min_rows):
                    A = tmp_df[my_variables].values
                    b = tmp_df['price_obs'].values
                    weigths = window_kernel(tmp_df.shape[0])
                    A = np.einsum('ij,i->ij', A, weigths)
                    b = np.multiply(b, weigths)

                    calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                    for rg_all_time in range(step):
                        param_dict[(my_area, my_time + rg_all_time, my_tech)] = calib.x
                else:
                    tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech and @my_time - @window_size <= TIMESTAMP and TIMESTAMP <= @my_time + @window_size').copy()
                    tmp2_df['intercept'] = 1
                    A = tmp2_df[my_variables].values
                    b = tmp2_df['price_reg'].values
                    calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                    for rg_all_time in range(step):
                        param_dict[(my_area, my_time + rg_all_time, my_tech)] = calib.x

                # if my_time % 500 == 0:
                #     print(my_tech, my_time)
                #     print(tmp_df) 
    asd = pd.DataFrame.from_dict(param_dict, orient='index')
    asd.index = pd.MultiIndex.from_tuples(asd.index.values, names=['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    asd.columns = param_names

    cols = asd.columns.tolist()
    zxc = asd.combine_first(price_param_df_list[iteration])[cols]

    price_param_df = zxc
    # print(price_param_df)
    print('linear approx time : ', time.time() - t)
    return price_param_df

def driver_local_linear_update(selected_techno, selected_area, selected_timestamp, step, window_size, for_estim_param_df, extended_for_estim_param_df, iteration, price_param_df_list, drivers):
    """
    update parameters with k-nn local linear model (linear regression with k nearest neighbors)
    """
    
    my_Selected_TECHNOLOGIES = selected_techno.copy()
    my_Selected_TECHNOLOGIES.remove('curtailment')
    selected_techno = my_Selected_TECHNOLOGIES

    param_dict = {}
    my_variables = ['intercept', 'energy', 'margin', 'fuel_price', 'co2_price']
    param_names = [f'{i}_param' for i in my_variables]
    nb_min_rows = 5
    t = time.time()
    for my_area in selected_area:
        for my_tech in selected_techno:
            # passed_timestamp = np.zeros(len(selected_timestamp))
            for rg_time in range(0, len(selected_timestamp), step):
                # if not passed_timestamp[rg_time]:
                my_time = selected_timestamp[rg_time]

                lb = [0, 0, -np.inf, 0, 0]
                ub = [np.inf, np.inf, 0, np.inf, 0.00001]
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
                
                # for driver nearest neighbours
                tmp_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
                tmp_df, index_nn = find_driver_nn(my_time, tmp_df, drivers)

                if (tmp_df.shape[0] >= nb_min_rows):
                    # passed_timestamp[ index_nn - 1]  = 1 # timestamp from 1 to end

                    A = tmp_df[my_variables].values
                    b = tmp_df['price_obs'].values
                    weigths = window_kernel(tmp_df.shape[0])
                    A = np.einsum('ij,i->ij', A, weigths)
                    b = np.multiply(b, weigths)

                    calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                    # for rg_all_time in list(index_nn.reshape(-1)):
                    param_dict[(my_area, my_time, my_tech)] = calib.x
                else:
                    tmp2_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech and @my_time - @window_size <= TIMESTAMP and TIMESTAMP <= @my_time + @window_size').copy()
                    tmp2_df['intercept'] = 1
                    A = tmp2_df[my_variables].values
                    b = tmp2_df['price_reg'].values
                    calib = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
                    param_dict[(my_area, my_time, my_tech)] = calib.x

    asd = pd.DataFrame.from_dict(param_dict, orient='index')
    asd.index = pd.MultiIndex.from_tuples(asd.index.values, names=['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    asd.columns = param_names

    cols = asd.columns.tolist()
    zxc = asd.combine_first(price_param_df_list[iteration])[cols]

    price_param_df = zxc
    # print(price_param_df)
    print('linear approx time : ', time.time() - t)
    return price_param_df

def window_kernel(size):
    return np.ones(size)

def gaussian_kernel(size):
    sigma = size/8 # approximation for having most part of gaussian

    x = np.arange(-size//2, size//2+1)[:size] # force output size
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-x**2/(2*sigma**2))


# A = np.array([[1,2, 3, 4, 5],[1,2, 3, 4, 5], [1,2, 3, 4, 5], [1,2, 3, 4, 5], [1,2, 3, 4, 5], [1,2, 3, 4, 5], [1,2, 3, 4, 5],[1,2, 3, 4, 5],[1,2, 3, 4, 5], [1,2, 3, 4, 5],[1,2, 3, 4, 5]])
# b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# weigths = gaussian_kernel(A.shape[0])
# # weigths = np.random.randint(0, 5, A.shape[0])/5
# print(weigths)

# A = np.einsum('ij,i->ij', A, weigths)
# b = np.multiply(b, weigths)
# print(A)
# print(b)
# calib = lsq_linear(A, b, lsmr_tol='auto')
# print(calib)
# plt.plot(window_kernel(24))
# plt.show()

# plt.plot(gaussian_kernel(50))
# plt.show()