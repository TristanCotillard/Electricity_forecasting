from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd


def display_price_drivers():
    INPUT_FOLDER = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/'

    prices = pd.read_pickle(INPUT_FOLDER + 'r_da_prices_df_15_18.pkl')
    drivers = pd.read_pickle(INPUT_FOLDER + 'r_fuel_prices_df_15_18.pkl')

    prices = prices[:8760]
    drivers = drivers[:8760]
    time = prices['TIMESTAMP_d'][:8760]
    col_drivers = list(drivers.columns.values)
    col_drivers.remove('TIMESTAMP_d')

    fig, axs = plt.subplots(4, 2)

    for rg, driver in enumerate(col_drivers):
        axs[rg, 0].plot(time, drivers[driver])
        axs[rg, 0].set_xlabel('time')
        axs[rg, 0].set_ylabel(driver)
        axs[rg, 0].grid(True)

        axs[rg, 1].plot(time, prices['price_obs'])
        axs[rg, 1].set_ylabel('price')

    fig.tight_layout()
    plt.show()

def compute_margin(year_test, area):
    INPUT_FOLDER = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/'

    avail_factor_df = pd.read_csv(INPUT_FOLDER + 'r_availabilityFactor' + str(year_test) + '_' + str(area) + '.csv')
    area_consumption_df = pd.read_csv(INPUT_FOLDER + 'r_areaConsumption_no_phes' + str(year_test) + '_' + str(area) + '.csv')
    tech_case='r_article_ramp'
    tech_parameters_df = pd.read_csv(INPUT_FOLDER+'Gestion_'+tech_case+'_TECHNOLOGIES'+str(year_test)+'.csv')
    tech_parameters_df.fillna(0, inplace=True)

    avail_factor_df = avail_factor_df.join(tech_parameters_df.set_index('TECHNOLOGIES')['capacity'], on='TECHNOLOGIES')
    avail_factor_df = avail_factor_df.assign(real_capacity=lambda x:x.capacity*x.availabilityFactor)
    real_capacities = avail_factor_df[['TIMESTAMP', 'real_capacity']].groupby('TIMESTAMP').sum()
    real_capacities.reset_index(inplace=True)
    real_capacities['margin'] = real_capacities['real_capacity'] - area_consumption_df['areaConsumption']

    return real_capacities['margin']


def display_margin_price(year_test, file_prediction):
    INPUT_FOLDER = 'C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/'

    margin = compute_margin(year_test, 'FR')

    prices = pd.read_pickle(INPUT_FOLDER + 'r_da_prices_df_15_18.pkl')
    final_merit_order = pd.read_pickle(INPUT_FOLDER + file_prediction)[-1]
    final_merit_order.reset_index(inplace=True)

    deb = (year_test - 2015)*8760
    fin = (year_test - 2014)*8760
    prices = prices[deb:fin]

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(margin[4100:4900])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('margin')
    axs[0].grid(True)

    axs[1].plot(prices['price_obs'][4100:4900])
    axs[1].set_ylabel('price')

    fig.tight_layout()
    plt.show()

'r_final_merit_order_df_list20172018.pkl'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Compute supply and demand curve
# prod = [0, 15, 15, 25, 25, 35, 35, 37, 37, 40, 40, 42, 42, 46, 46]
# price = [5, 5, 10, 10, 12, 12, 14, 14, 15, 15, 18, 18, 22, 22, 26 ]
# plt.xlim([0, 46])
# plt.ylim([0, 26])
# plt.xlabel('Production')
# plt.ylabel('Price')
# plt.vlines(36, 0, 26, color='r', linestyles='dashed', label=' Inelastic Demand')
# plt.plot(prod, price, label='Supply')
# plt.legend(loc='upper left')
# plt.show()