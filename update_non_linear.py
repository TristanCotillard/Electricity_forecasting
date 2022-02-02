import numpy as np
import pandas as pd
from functions.f_vm import vm_expand_grid
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle

def create_prices_sim(selected_area, selected_timestamp, selected_techno):
    price_sim = vm_expand_grid({"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    price_sim = price_sim.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    price_sim['price_res_sim'] = 0

    return price_sim

def test_prices_sim(selected_area, selected_timestamp, selected_techno, drivers_df, FOLDER_OUTPUT):
    price_sim = create_prices_sim(selected_area, selected_timestamp, selected_techno)

    area = selected_area[0]
    for tech in selected_techno:
        model = pickle.load(open(FOLDER_OUTPUT + tech + '.pkl', 'rb'))
        test_df = drivers_df.query('AREAS == @area and TECHNOLOGIES == @tech')
        result = model.predict(test_df)
        # print(result.shape)
        # print(price_sim.loc[(area, slice(None), tech), 'price_res_sim'].shape)
        price_sim.loc[(area, slice(None), tech), 'price_res_sim'] = result
    return price_sim

def test_apply_model(all_orders_df, drivers, selected_area, selected_techno, FOLDER_OUTPUT):

    area = selected_area[0]
    for tech in selected_techno:
        model = pickle.load(open(FOLDER_OUTPUT + tech + '.pkl', 'rb'))
        test_df = all_orders_df.query('AREAS == @area and TECHNOLOGIES == @tech')
        result = model.predict(test_df[drivers])
        all_orders_df.loc[(area, slice(None), tech), 'order_price_residue'] = result
    return all_orders_df
        
def create_drivers(empty_indexed_df, margin_price_df, full_fuel_price_df, full_co2_price_df, prices_sim):
    obj_param_df = empty_indexed_df.merge(
            margin_price_df['margin'],how='left', left_index=True, right_index=True).merge(
            full_fuel_price_df['fuel_price'], how='left', left_index=True, right_index=True).merge(
            full_co2_price_df['co2_price'], how='left', left_index=True, right_index=True).merge(
            prices_sim['price_res_sim'], how='left', left_index=True, right_index=True)
    obj_param_df = obj_param_df.reorder_levels(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    
    return obj_param_df

def test_create_drivers(empty_indexed_df, margin_price_df, full_fuel_price_df, full_co2_price_df):
    obj_param_df = empty_indexed_df.merge(
            margin_price_df['margin'],how='left', left_index=True, right_index=True).merge(
            full_fuel_price_df['fuel_price'], how='left', left_index=True, right_index=True).merge(
            full_co2_price_df['co2_price'], how='left', left_index=True, right_index=True)
    obj_param_df['energy'] = 0
    obj_param_df = obj_param_df.reorder_levels(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    obj_param_df = obj_param_df[['energy', 'margin', 'fuel_price', 'co2_price']]

    return obj_param_df

def update_price(selected_techno, selected_area, selected_timestamp, for_estim_param_df, extended_for_estim_param_df, save, OUTPUT_FOLDER):
    my_Selected_TECHNOLOGIES = selected_techno.copy()
    my_Selected_TECHNOLOGIES.remove('curtailment')
    my_variables = ['energy', 'margin', 'fuel_price', 'co2_price']
    price_sim = vm_expand_grid({"AREAS": selected_area, "TIMESTAMP": selected_timestamp, "TECHNOLOGIES": selected_techno})
    price_sim = price_sim.set_index(['AREAS', 'TIMESTAMP', 'TECHNOLOGIES'])
    price_sim['price_res_sim'] = 0

    t = time.time()

    nb_min_rows = 5
    for my_area in selected_area:
        for my_tech in selected_techno:
            train_df = for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
            model = AdaBoostRegressor()
            
            if (train_df.shape[0] >= nb_min_rows):
                model.fit(train_df[my_variables], train_df['price_resid'])
                result = model.predict(extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')[my_variables])
            else:
                train_df = extended_for_estim_param_df.query('AREAS == @my_area and TECHNOLOGIES == @my_tech')
                model.fit(train_df[my_variables], train_df['price_resid'])
                result = model.predict(train_df[my_variables])

            price_sim.loc[(my_area, slice(None), my_tech), 'price_res_sim'] = result

            if save:
                with open(OUTPUT_FOLDER + my_tech + '.pkl', 'wb') as f:
                    pickle.dump(model, f)

    # print(for_estim_param_df)
    print('linear approx time : ', time.time() - t)

    return price_sim