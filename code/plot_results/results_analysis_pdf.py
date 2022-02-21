import sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Print pdf observed and simulated prices

year_train = [2017]
year_test = [2017, 2018]
with_preprocessed = False # if seasonal and trend to be added

my_top_folder = 'data_elecprices/output_modified/'
my_folder = my_top_folder

for year_train in year_train:
    for year_test in year_test:
        if year_train  != year_test:
            is_with_bias_corr = True

        if year_train == year_test:
            is_train = True; my_range = 4
        else:
            is_train = False; my_range = 1

        # if is_with_bias_corr:
        #     str_biais = 'with bias corr'
        # else:
        #     str_biais = 'without bias corr'

        # if with_ramp:
        #     str_ramp = ', ramp, '
        # else:
        #     str_ramp = ', no_ramp, '
        str_ramp = ', ramp, '

        # Load data
        results_list = pd.read_pickle(my_folder + '/r_results_list'+str(year_train)+str(year_test)+'.pkl')
        final_merit_order_df_list = pd.read_pickle(my_folder + '/r_final_merit_order_df_list'+str(year_train)+str(year_test)+'.pkl')
        production_df_list = pd.read_pickle(my_folder + '/r_production_df_list'+str(year_train)+str(year_test)+'.pkl')
        areaConsumption_list = pd.read_pickle(my_folder + '/r_areaConsumption_list'+str(year_train)+str(year_test)+'.pkl')
        duals_df_list = pd.read_pickle(my_folder + '/r_duals_df_list' + str(year_train) + str(year_test) + '.pkl')
        # price_param_df_list = pd.read_pickle(my_folder + '/r_price_param_df_list'+str(year_train)+'.pkl')
        bias_correction_df_list = pd.read_pickle(my_folder + '/r_bias_correction_df_list'+str(year_train)+'.pkl')
        bias_correction_lagr_df_list = pd.read_pickle(my_folder + '/r_bias_correction_lagr_df_list'+str(year_train)+'.pkl')
        merit_order_df = pd.read_pickle(my_folder + '/r_merit_order_df_'+str(year_train)+'.pkl')
        daPrices_df= pd.read_pickle(my_folder + '/r_daPrices_df_' + str(year_test) + '.pkl')

        if with_preprocessed:
            PreprocessedFolder = 'data_elecprices/data_preprocessed/'
            mean_sd = pd.read_csv(PreprocessedFolder + 'mean_sd.csv')
            daPrices_df['price_obs'] = daPrices_df['price_obs'] * mean_sd['price_sd'].values + mean_sd['price_mean'].values

        plot_list = list()
        i_plot = 0
        for iter in range(my_range):
            results = results_list[iter]
            final_merit_order_df = final_merit_order_df_list[iter]
            production_df = production_df_list[iter]
            areaConsumption = areaConsumption_list[iter]
            bias_correction_df = bias_correction_df_list[iter]
            duals_df = duals_df_list[iter]
            bias_correction_lagr_df = bias_correction_lagr_df_list[iter]

            timestamp_df = areaConsumption.query("AREAS == 'FR'").copy().reset_index()[['TIMESTAMP', 'DateTime']]
            timestamp_df['TIMESTAMP_d'] = pd.to_datetime(timestamp_df.DateTime, format='%Y-%m-%dT%H:%M:%SZ')
            timestamp_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
            timestamp_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday
            tmp = timestamp_df.copy().set_index(['hour','weekday'])[['TIMESTAMP']]
            bias_correction_df_ts = tmp.merge(bias_correction_df, how='left', left_index=True, right_index=True).reset_index(
                 ).set_index(['AREAS','TIMESTAMP'])

            tmp = timestamp_df.copy().set_index(['hour','weekday'])[['TIMESTAMP']]
            bias_correction_lagr_df_ts = tmp.merge(bias_correction_lagr_df, how='left', left_index=True, right_index=True).reset_index(
                 ).set_index(['AREAS','TIMESTAMP'])

            # if is_train:
            #     price_param_df = price_param_df_list[iter]

            for i in ['param_with_bias','param_without_bias','duals_with_bias','duals_without_bias']:
                if i == 'param_with_bias':
                    for_res_analysis_df = final_merit_order_df.merge(daPrices_df['price_obs'], how='left', left_index=True, right_index=True)
                elif i == 'param_without_bias':
                    for_res_analysis_df = final_merit_order_df.merge(daPrices_df['price_obs'], how='left', left_index=True, right_index=True)
                    for_res_analysis_df = for_res_analysis_df.merge(bias_correction_df_ts['delta_price'], how='left', left_index=True, right_index=True)
                    for_res_analysis_df = for_res_analysis_df.assign(price_sim = lambda x: x.price_sim - x.delta_price)
                elif i == 'duals_with_bias':
                    for_res_analysis_df = duals_df.merge(daPrices_df['price_obs'], how='left', left_index=True, right_index=True)
                    for_res_analysis_df = for_res_analysis_df.rename(columns = {'price_sim_lagr': 'price_sim'})
                elif i == 'duals_without_bias':
                    for_res_analysis_df = duals_df.merge(daPrices_df['price_obs'], how='left', left_index=True, right_index=True)
                    for_res_analysis_df = for_res_analysis_df.rename(columns = {'price_sim_lagr': 'price_sim'})
                    for_res_analysis_df = for_res_analysis_df.merge(bias_correction_lagr_df_ts['delta_price'], how='left', left_index=True, right_index=True)
                    for_res_analysis_df = for_res_analysis_df.assign(price_sim = lambda x: x.price_sim - x.delta_price)
                else:
                    sys.exit("case not found!")

            # for_res_analysis_df = energyCtrDual[['energyCtr']].merge(daPrices_df['price_obs'], how='left', left_index=True, right_index=True)
            # for_res_analysis_df = for_res_analysis_df.rename(columns = {'energyCtr':'price_obs'})

                # RMSE
                rmse = np.sqrt(np.mean((for_res_analysis_df.price_obs-for_res_analysis_df.price_sim)**2))

                # Delta SD
                delta_sd = np.std(for_res_analysis_df.price_obs) - np.std(for_res_analysis_df.price_sim)
                delta_sd

                # ax = for_res_analysis_df[['price_obs','price_sim']].plot.hist(bins=1000, alpha=0.5)
                # plt.show()

                # Plot time series
                i_plot = i_plot + 1
                my_area = 'FR'
                plot_df = for_res_analysis_df.merge(timestamp_df.set_index(['TIMESTAMP']), how='left', left_index=True, right_index=True)
                #plot_title = 'train:'+str(year_train)+', test:'+str(year_test)+', iter: '+str(iter)+', rmse: '+str(round(rmse,1))+', delta_sd: '+str(round(delta_sd,1))+'\n'+my_area+', ramps, '+str_biais
                plot_title = 'train:' + str(year_train) + ', test:' + str(year_test) + ', iter: ' + str(iter) + ', rmse: ' + str(round(rmse, 1)) + ', delta_sd: ' + str(round(delta_sd, 1)) + '\n' + my_area + str_ramp + i
                plt.close("all")
                plt.ioff()
                fig = plt.figure()
                plt.plot('TIMESTAMP_d', 'price_obs', data=plot_df.loc[(my_area,slice(None)),:], color='black', linewidth=1, label = 'Observation', figure = fig)
                plt.plot('TIMESTAMP_d', 'price_sim', data=plot_df.loc[(my_area,slice(None)),:], color='red', linewidth=1, label = 'Simulation', figure = fig)
                plt.legend(title = 'Data type')
                plt.xlabel('Date', figure = fig)
                plt.grid(color='lightgrey', linestyle='-', linewidth=1, figure = fig)
                plt.ylabel('Price (EUR/MWh)', figure = fig)
                plt.title(plot_title, figure = fig)
                #plt.show()
                #fig.show()

                plot_list.append(fig)

        pp = PdfPages(my_folder + '/timeseries'+str(year_train)+str(year_test)+'.pdf')
        for i in range(i_plot):
            pp.savefig(plot_list[i])
        pp.close()