import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# When testing the model on detrended data
# Add up simulation on trend and residual and compare it with observed trend and residual

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_trend():
    """
    plot observed and simulated trend (train and test)
    """
    INPUT_FOLDER_trend = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/output_modified/linear_preprocessed/trend/'

    year_train = '2015' 
    year_test = '2016'
    deb = 0
    fin = 8760

    prices_train_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_daPrices_df_' + year_train + '.pkl')
    final_merit_order_train_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_train + '.pkl')[-1]
    prices_train_trend.reset_index(inplace=True)
    final_merit_order_train_trend.reset_index(inplace=True)
    plt.plot(prices_train_trend['price_obs'], color='black', label='Observation')
    plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin], color='r', label='Simulation')
    # plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin] + final_merit_order_train_res['order_price_residue'][deb:fin] + prices_train_res['price_seasonal'][deb:fin], color='r', label='Simulation')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

    prices_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_daPrices_df_' + year_test + '.pkl')
    final_merit_order_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')[-1]
    prices_test_trend.reset_index(inplace=True)
    final_merit_order_test_trend.reset_index(inplace=True)
    plt.plot(prices_test_trend['price_obs'], color='black', label='Observation')
    plt.plot(final_merit_order_test_trend['order_price_trend'][deb:fin], color='r', label='Simulation')
    # plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin] + final_merit_order_train_res['order_price_residue'][deb:fin] + prices_train_res['price_seasonal'][deb:fin], color='r', label='Simulation')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

plot_trend()

def sum_trend_res(year_train, year_test, INPUT_FOLDER_trend, INPUT_FOLDER_res, deb, fin):
    """
    Sum simulated trend and residual
    """
    prices_train_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_daPrices_df_' + year_train + '.pkl')
    final_merit_order_train_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_train + '.pkl')[-1]

    prices_train_trend.reset_index(inplace=True)
    prices_train_res = pd.read_pickle(INPUT_FOLDER_res + 'r_daPrices_df_' + year_train + '.pkl')
    final_merit_order_train_res = pd.read_pickle(INPUT_FOLDER_res + 'r_final_merit_order_df_list' + year_train + year_train + '.pkl')[-1]
    prices_train_res.reset_index(inplace=True)
    final_merit_order_train_res.reset_index(inplace=True)
    final_merit_order_train_trend.reset_index(inplace=True)

    plt.plot(prices_train_res['price_obs'], color='black', label='Observation')
    plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin])
    # plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin] + final_merit_order_train_res['order_price_residue'][deb:fin] + prices_train_res['price_seasonal'][deb:fin], color='r', label='Simulation')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    rmse_train = np.sqrt(np.mean((prices_train_res['price_obs'] - final_merit_order_train_res['order_price_residue']- final_merit_order_train_trend['order_price_trend'] - prices_train_res['price_seasonal'][deb:fin])**2))
    delta_sd_train = np.std(prices_train_res['price_obs']) - np.std(final_merit_order_train_res['order_price_residue'] + final_merit_order_train_trend['order_price_trend'] + prices_train_res['price_seasonal'])
    print('rmse_train, delta_sd_train : ', rmse_train, delta_sd_train)

    prices_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_daPrices_df_' + year_test + '.pkl')
    final_merit_order_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')[-1]
    final_merit_order_test_trend.reset_index(inplace=True)

    prices_test_trend.reset_index(inplace=True)
    final_merit_order_test_trend.reset_index(inplace=True)
    prices_test_res = pd.read_pickle(INPUT_FOLDER_res + 'r_daPrices_df_' + year_test + '.pkl')
    final_merit_order_test_res = pd.read_pickle(INPUT_FOLDER_res + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')[-1]
    prices_test_res.reset_index(inplace=True)
    final_merit_order_test_res.reset_index(inplace=True)

    plt.plot(prices_test_res['price_obs'][deb:fin], color='black', label='Observation')
    plt.plot(final_merit_order_test_trend['order_price_trend'][deb:fin])
    # plt.plot(final_merit_order_test_trend['order_price_trend'][deb:fin] + final_merit_order_test_res['order_price_residue'][deb:fin] + prices_test_res['price_seasonal'][deb:fin], color='r', label='Simulation')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    rmse_test = np.sqrt(np.mean((prices_test_res['price_obs'] - final_merit_order_test_res['order_price_residue']- final_merit_order_test_trend['order_price_trend'] - prices_test_res['price_seasonal'][deb:fin])**2))
    delta_sd_test = np.std(prices_test_res['price_obs']) - np.std(final_merit_order_test_res['order_price_residue'] + final_merit_order_test_trend['order_price_trend'] + prices_train_res['price_seasonal'])
    print('rmse_test, delta_sd_test : ', rmse_test, delta_sd_test)


def marginal_tech(year_train, year_test, INPUT_FOLDER_trend, deb, fin):
    """
    plot marginal technology (a value per technology) and simulated trend 
    """
    final_merit_order_train_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_train + '.pkl')[-1]
    final_merit_order_train_trend.reset_index(inplace=True)

    final_merit_order_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_final_merit_order_df_list' + year_train + year_test + '.pkl')[-1]
    final_merit_order_test_trend.reset_index(inplace=True)

    marginal_tech = pd.DataFrame()
    marginal_tech['TECH'] = final_merit_order_train_trend['TECHNOLOGIES']
    values = np.unique(marginal_tech['TECH'].values)

    for i, value in enumerate(values):
        marginal_tech[ marginal_tech['TECH'] == value ] = i
        print(i, ' : ', value)
    plt.plot(marginal_tech[deb:fin])
    plt.plot(final_merit_order_train_trend['order_price_trend'][deb:fin])
    plt.show()

    marginal_tech = pd.DataFrame()
    marginal_tech['TECH'] = final_merit_order_test_trend['TECHNOLOGIES']
    values = np.unique(marginal_tech['TECH'].values)

    for i, value in enumerate(values):
        marginal_tech[ marginal_tech['TECH'] == value ] = i
        print(i, ' : ', value)
    plt.plot(marginal_tech[deb:fin])
    plt.plot(final_merit_order_test_trend['order_price_trend'][deb:fin])
    plt.show()

    for merit in final_merit_order_test_trend:
        merit.reset_index(inplace=True)
        # plt.plot(prices_test_trend['price_obs'][deb:fin], color='r')
        plt.plot(merit['order_price_trend'][deb:fin])
    plt.show()

    print(final_merit_order_test_trend)
    bias_test_trend = pd.read_pickle(INPUT_FOLDER_trend + 'r_bias_correction_df_list' + year_train + '.pkl')[-1]
    bias_test_trend.reset_index(inplace=True)
    print(bias_test_trend)
    plt.show()


INPUT_FOLDER_trend = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/output_modified/linear_preprocessed/trend/'
INPUT_FOLDER_res = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/output_modified/linear_preprocessed/std_detrend/'

year_train = '2018' 
year_test = '2017'
deb = 0
fin = 8760