import pandas as pd
import matplotlib.pyplot as plt
import random as rd

# Read observed and simulated prices and plot results 

FOLDER_result = "data_elecprices/output_modified/linear/linear_2017_20_iter/"
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

year_train = '2017'
year_test = '2017'

# Train
# prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_2018.pkl')
# price_params = pd.read_pickle(FOLDER_result + 'r_price_param_df_list'+ year_train +'.pkl')
# areaConsump = pd.read_pickle(FOLDER_result + 'r_areaConsumption_list' + year_train + year_train+ '.pkl')
# biasCorr = pd.read_pickle(FOLDER_result + 'r_bias_correction_df_list' + year_train +'.pkl')
# meritOrder = pd.read_pickle(FOLDER_result + 'r_merit_order_df_' + year_train + '.pkl')

# Test
# duals = pd.read_pickle(FOLDER_result + 'r_duals_df_list' + year_train + year_test + '.pkl')[0]
# prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_' + year_test + '.pkl')
# old_prices = pd.read_pickle(FOLDER_result + 'r_daPrices_df_' + year_train + '.pkl')
# results = pd.read_pickle(FOLDER_result + 'r_results_list' + year_train + year_test + '.pkl')
# final_merit_order = pd.read_pickle(FOLDER_result + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')
# biais = pd.read_pickle(FOLDER_result + 'r_bias_correction_df_list'+ year_train +'.pkl')
# biais_lagr = pd.read_pickle(FOLDER_result + 'r_bias_correction_lagr_df_list'+ year_train +'.pkl')
# final_merit_order = pd.read_pickle(FOLDER_result + 'r_final_merit_order_df_list' + year_train + year_test '.pkl')[0]

# prices.reset_index(inplace=True)
# duals.reset_index(inplace=True)
# old_prices.reset_index(inplace=True)




def display_params(price_params, techno):
    """
    plot linear regression parameters values of all drivers for a given technology of production
    """
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
    plt.show()

# for techno in ['Fossil Gas', 'Fossil Hard coal', 'Fossil oil', 'Hydro Water Reservoir', 'Nuclear']:
#     display_params(price_params[-1].reset_index(), techno)

def print_samples(final_merit_order, prices, size):
    """
    plot 10 random windows of observed and simulated prices
    used to compare observation and simulation
    """
    for _ in range(10):
        nb = rd.randint(0, 8660)

        deb = nb
        fin = nb+size

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