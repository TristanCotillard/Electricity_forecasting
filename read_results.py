from typing import final
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

FOLDER_result = "data_elecprices/output_modified/linear/linear_2017_20_iter/"
# window_hebdo_2015_2016
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

year_train = '2017'
year_test = '2017'
# Train
# areaConsump = pd.read_pickle(FOLDER_result + 'r_areaConsumption_list' + year_train + year_train+ '.pkl')
#print(areaConsump)
# biasCorr = pd.read_pickle(FOLDER_result + 'r_bias_correction_df_list' + year_train +'.pkl')
#print(biasCorr)
# meritOrder = pd.read_pickle(FOLDER_result + 'r_merit_order_df_' + year_train + '.pkl')
#print(meritOrder['price_sim'])

# Test sur 2015-2016
# duals = pd.read_pickle(FOLDER_result + 'r_duals_df_list' + year_train + year_test + '.pkl')[0]
# print(duals)
# prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_' + year_test + '.pkl')
# old_prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_' + year_train + '.pkl')

# results = pd.read_pickle(FOLDER_result + 'r_results_list' + year_train + year_test + '.pkl')
final_merit_order = pd.read_pickle(FOLDER_result + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')
# biais = pd.read_pickle(FOLDER_result + 'r_bias_correction_df_list'+ year_train +'.pkl')
# print(biais[0])
# biais_lagr = pd.read_pickle(FOLDER_result + 'r_bias_correction_lagr_df_list'+ year_train +'.pkl')
# print(biais_lagr[0])

# price_params = pd.read_pickle(FOLDER_result + 'r_price_param_df_list'+ year_train +'.pkl')

# Look at the data !
# prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_2018.pkl')
# final_merit_order = pd.read_pickle(FOLDER_result + 'r_final_merit_order_df_list20172018.pkl')[0]

for merit in final_merit_order:
    merit.reset_index(inplace=True)
    final_tech = merit.groupby(by='TECHNOLOGIES')['AREAS'].count()
    print(final_tech)
# prices.reset_index(inplace=True)
# duals.reset_index(inplace=True)
# old_prices.reset_index(inplace=True)

# print(prices.columns)
# plt.plot(prices['TIMESTAMP_d'], prices['price_obs'])
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.grid(True)
# plt.show()

def display_params(price_params, techno):
    params = price_params[ price_params['TECHNOLOGIES'] == techno ]
    inter_param = params['intercept_param']
    energy_param = params['energy_param']
    margin_param = params['energy_param']
    fuel_param = params['fuel_price_param']
    co2_param = params['co2_price_param']

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(inter_param, label='inter')
    axs[0, 1].plot(energy_param, label='energy')
    axs[0, 2].plot(margin_param, label='margin')
    axs[1, 0].plot(fuel_param, label='fuel')
    axs[1, 1].plot(co2_param, label='co2')
    for i in range(2):
        for j in range(3):
            axs[i,j].legend()
    fig.tight_layout()
    # plt.show()
    print(margin_param)

# for techno in ['Fossil Gas', 'Fossil Hard coal', 'Fossil oil', 'Hydro Water Reservoir', 'Nuclear']:
#     display_params(price_params[-1].reset_index(), techno)

def print_samples(final_merit_order, prices):
    for _ in range(10):
        nb = rd.randint(0, 8660)

        deb = nb
        fin = nb+100

        fig, axs = plt.subplots(2, 1)

        axs[0].plot(final_merit_order['price_sim'][deb:fin], color='orange')
        axs[0].plot(prices['price_obs'][deb:fin])
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('price_sim')
        axs[0].grid(True)

        axs[1].plot(prices['price_obs'][deb:fin])
        axs[1].set_ylabel('price_obs')
        axs[1].grid(True)

        fig.tight_layout()
        plt.show()

# print_samples(final_merit_order, prices)