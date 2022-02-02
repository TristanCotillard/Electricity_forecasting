import sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with_ramp = True

my_top_folder = 'data_elecprices/output_modified/'
my_folder = my_top_folder
for year_train in [2018]:
    for year_test in [2017, 2018]:
        if year_train  != year_test:
            is_with_bias_corr = True
            with_ramp = False

        if year_train == year_test:
            is_train = True; my_range = 5
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

        # load
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

        plot_list = list()
        i_plot = 0
        for iter in range(my_range):
            #model = model_list[iter]
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
            #for i in ['duals_without_bias']:
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
                #rmse = mean_squared_error(for_res_analysis_df.price_obs, for_res_analysis_df.price_sim, squared = False)
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

    # # Distribution of simulated prices
    # for_res_analysis_df['price_sim'].plot.hist(bins=40, alpha=0.5)
    # plt.show()
    #
    # # Number of marginals hours
    # for_res_analysis_df.reset_index()['TECHNOLOGIES'].value_counts().plot(kind='bar')
    # plt.show()
    #
    # print(for_estim_param_df.reset_index()['TECHNOLOGIES'].value_counts())
    # print(final_merit_order_df['price_sim'].mean())
    #
    # # Equilibrium satisfied?
    # Delta = production_df.sum(axis=1) - areaConsumption.areaConsumption
    # # Delta = production_df.sum(axis=1) - areaConsumption.reset_index().set_index(['AREAS','TIMESTAMP'])['areaConsumption']
    # abs(Delta).sum()
    #
    # # # Production time series
    # # # production_df = production_df.set_index(['AREAS','TIMESTAMP'])
    # # # areaConsumption = areaConsumption.set_index(['AREAS','TIMESTAMP'])
    # production_df_ = vm_ChangeTIMESTAMP2Dates(production_df, timestamp_df)
    # areaConsumption_ = vm_ChangeTIMESTAMP2Dates(areaConsumption, timestamp_df)
    # fig = MyAreaStackedPlot(production_df_, Conso=areaConsumption_)
    # fig = fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
    # plotly.offline.plot(fig, filename='file.html')  ## offline
    #
    # # Check hour +weekday
    # timestamp_df['hour'] = timestamp_df.TIMESTAMP_d.dt.hour
    # timestamp_df['weekday'] = timestamp_df.TIMESTAMP_d.dt.weekday
    #
    # for_hour_weekday = for_res_analysis_df[['price_sim','price_obs']].merge(
    #     timestamp_df.set_index(['TIMESTAMP']), how='left', left_index=True, right_index=True)
    #
    #
    # # https: // www.shanelynn.ie / summarising - aggregation - and -grouping - data - in -python - pandas /
    # to_plot_hour_df = for_hour_weekday.groupby('hour')[['price_sim','price_obs']].mean()
    # to_plot_weekday_df = for_hour_weekday.groupby('weekday')[['price_sim','price_obs']].mean()
    #
    # my_area = 'FR'
    # plot_title = 'train:'+str(year_train)+', test:'+str(year_test)+', iter: '+str(iter)+', rmse: '+str(round(rmse,1))+', delta_sd: '+str(round(delta_sd,1))+'\n'+my_area+', ramps'
    # plt.close("all")
    # plt.ioff()
    # plt.plot('hour', 'price_obs', data=to_plot_hour_df.reset_index(), color='blue', linewidth=2)
    # plt.plot('hour', 'price_sim', data=to_plot_hour_df.reset_index(), color='red', linewidth=2)
    # plt.legend()
    # plt.title(plot_title)
    # plt.show()
    #
    # my_area = 'FR'
    #
    #
    # plot_title = 'train:'+str(year_train)+', test:'+str(year_test)+', iter: '+str(iter)+', rmse: '+str(round(rmse,1))+', delta_sd: '+str(round(delta_sd,1))+'\n'+my_area+', ramps'
    # plt.close("all")
    # plt.ioff()
    # positions = (0, 1, 2, 3, 4, 5, 6)
    # labels = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
    # plt.xticks(positions, labels)
    # plt.plot('weekday', 'price_obs', data=to_plot_weekday_df.reset_index(), color='blue', linewidth=2)
    # plt.plot('weekday', 'price_sim', data=to_plot_weekday_df.reset_index(), color='red', linewidth=2)
    # plt.legend()
    # plt.title(plot_title)
    # plt.show()


# def plotGraph(X, Y):
#     fig = plt.figure()
#     ### Plotting arrangements ###
#     return fig
#
# from matplotlib.backends.backend_pdf import PdfPages
#
# plot1 = plotGraph(tempDLstats, tempDLlabels)
# plot2 = plotGraph(tempDLstats_1, tempDLlabels_1)
# plot3 = plotGraph(tempDLstats_2, tempDLlabels_2)
#
# pp = PdfPages('foo.pdf')
# pp.savefig(plot1)
# pp.savefig(plot2)
# pp.savefig(plot3)
# pp.close()