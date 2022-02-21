#import plotly.graph_objects as go
#import plotly
import pandas as pd
import numpy as np
import itertools

# https://www.python.org/dev/peps/pep-0008/#function-and-variable-names
# Function names should be lowercase, with words separated by underscores as necessary to improve readability.
# Variable names follow the same convention as function names.

# https://www.datacamp.com/community/tutorials/docstrings-python
# https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
# Documentation docstrings des fonctions Sphinx Style

# """[Summary]
#
# :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
# :type [ParamName]: [ParamType](, optional)
# ...
# :raises [ErrorType]: [ErrorDescription]
# ...
# :return: [ReturnDescription]
# :rtype: [ReturnType]
# """

# def vm_define_config() :
# def vm_select_model():

def vm_ChangeTIMESTAMP2Dates(my_df,timestamp_df):

    df = my_df.copy()

    # Get index name and undo index
    index_names = df.index.names
    df.reset_index(inplace=True)

    # merge
    df = df.merge(timestamp_df[['TIMESTAMP', 'TIMESTAMP_d']], on='TIMESTAMP', how='left')

    # Drop, rename, reindex
    df = df.drop(columns='TIMESTAMP').rename(columns={'TIMESTAMP_d': 'TIMESTAMP'}).set_index(index_names)

    return(df)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html
def vm_expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

# https://stackoverflow.com/questions/27317517/make-division-by-zero-equal-to-zero
# weird_division
def vm_non_zero_division(n, d):
    return n / d if d else 0

def vm_make_capa_avail(installed_capa, availability_factor) :
    '''
    Create df with installed capa, avail factor and avail capa from the dfs TechParameters and availabilityFactor
    :param installed_capa: data frame (index = AREAS,TECHNOLOGIES; column = capacity)
    :param installd_capa: pandas.core.frame.DataFrame
    :type availability_factor: data frame (index = AREAS,TIMESTAMP,TECHNOLOGIES; column = availabilityFactor)
    :type availability_factor: pandas.core.frame.DataFrame
    :return: capa_avail_df (index = AREAS,TIMESTAMP,TECHNOLOGIES; column = capacity,availability_factor,availability)
    :rtype: pandas.core.frame.DataFrame
    '''

    capa_avail_df = availability_factor[['availabilityFactor']].merge(
        installed_capa[['capacity']], left_index=True, right_index=True, how='outer'
    ).rename(
        columns={"availabilityFactor": "availability_factor"})

    capa_avail_df = capa_avail_df.reset_index()
    capa_avail_df = capa_avail_df[capa_avail_df.TECHNOLOGIES != "curtailment"]
    capa_avail_df.set_index(['AREAS','TECHNOLOGIES','TIMESTAMP'])
    capa_avail_df = capa_avail_df.assign(availability = lambda x: x.capacity * x.availability_factor)

    return (capa_avail_df)

def vm_make_margin(capa_avail_df, conso_df) :
    '''
    Create df with installed capa, avail factor and avail capa from the dfs TechParameters and availabilityFactor
    :param capa_avail_df: data frame (index = AREAS,TIMESTAMP,TECHNOLOGIES; column = capacity,availability_factor,availability)
    :param capa_avail_df: pandas.core.frame.DataFrame
    :type conso_df: data frame (index = AREAS,TIMESTAMP; column = areaConsumption)
    :type conso_df: data frame (index = AREAS,TIMESTAMP; column = areaConsumption)
    :return: margin_df (index = AREAS,TIMESTAMP; column = consumption, capacity, margin)
    :rtype: pandas.core.frame.DataFrame
    '''

    # Total hourly avail capacity
    total_avail_df = capa_avail_df.groupby(['AREAS','TIMESTAMP']).agg({'availability': 'sum'})

    # Merge capa and conso, then compute margin
    margin_df = total_avail_df[['availability']].merge(
        conso_df[['areaConsumption']], left_index=True, right_index=True, how='outer'
    ).rename(
        columns={"areaConsumption": "consumption"}
    ).assign(margin = lambda x: x.availability - x.consumption)

    return (margin_df)

def vm_plotly_ts_capa_vs_conso(capa_avail_df, conso_df, value_selector='capacity') :
    '''
    Représenter l'empilement des disponibilités et la consommation totale au pas de temps horaire
    :param capa_avail_df: data frame (index = TIMESTAMP,TECHNOLOGIES; column = capacity,availability)
    :param conso_df: data frame (index = TIMESTAMP; column = areaConsumption)
    :param value_selector: capacity or availability
    :type capa_avail_df: pandas.core.frame.DataFrame
    :type conso_df: pandas.core.frame.DataFrame
    :type value_selector: str
    :return: fig
    '''

    y_df = pd.pivot_table(capa_avail_df[[value_selector]], values=value_selector, index='TIMESTAMP', columns='TECHNOLOGIES')

    y_names = y_df.columns.unique().tolist()
    fig = go.Figure()
    i = 0
    for col in y_df.columns:
        if i == 0:
            fig.add_trace(go.Scatter(x=y_df.index, y=y_df[col], fill='tozeroy',
                                     mode='none', name=col))  # fill down to xaxis
            colNames = [col]
        else:
            colNames.append(col)
            fig.add_trace(go.Scatter(x=y_df.index, y=y_df.loc[:, y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                     mode='none', name=Names[i]))  # fill to trace0 y
        i = i + 1

    fig.add_trace(go.Scatter(x=y_df.index, y=conso_df["areaConsumption"], name="Conso",
                            line=dict(color='red', width=0.4)))  # fill down to xaxis

    fig = fig.update_layout(title_text=value_selector+" (en KWh)", xaxis_title="heures de l'année")
    fig.update_xaxes(rangeslider_visible=True)
    plotly.offline.plot(fig, filename='file.html')
    return(fig)

def vm_make_obs_prices_df() :
    '''

    :return:
    '''

    return ()


def vm_make_sim_prices_df():
    '''

    :return:
    '''

    return ()

def vm_merge_obs_sim_prices_df(data_df, TemperatureThreshold=14, TemperatureName='Temperature',ConsumptionName='Consumption',TimeName='TIMESTAMP') :
    '''
    fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
    :param data: panda data frame with "Temperature" and "Consumption" as columns
    :param TemperatureThreshold: the threshold heating temperature
    :param TemperatureName default 'Temperature' name of column with Temperature
    :param ConsumptionName default 'Consumption' name of column with consumption
    :param TimeName default 'Date' name of column with time
    :return: a dictionary with Thermosensibilite, and a panda data frame with two new columns NTS_C and TS_C
    '''

    return (ConsoSeparee_df, Thermosensibilite)

def vm_plotly_ts_obs_sim_prices(data_df, TemperatureThreshold=14, TemperatureName='Temperature',ConsumptionName='Consumption',TimeName='TIMESTAMP') :
    '''
    fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
    :param data: panda data frame with "Temperature" and "Consumption" as columns
    :param TemperatureThreshold: the threshold heating temperature
    :param TemperatureName default 'Temperature' name of column with Temperature
    :param ConsumptionName default 'Consumption' name of column with consumption
    :param TimeName default 'Date' name of column with time
    :return: a dictionary with Thermosensibilite, and a panda data frame with two new columns NTS_C and TS_C
    '''

    return (ConsoSeparee_df, Thermosensibilite)

def vm_make_metrics_dict(data_df, TemperatureThreshold=14, TemperatureName='Temperature',ConsumptionName='Consumption',TimeName='TIMESTAMP') :
    '''
    fonction décomposant la consommation électrique d'une année en une part thermosensible et une part non thermosensible
    :param data: panda data frame with "Temperature" and "Consumption" as columns
    :param TemperatureThreshold: the threshold heating temperature
    :param TemperatureName default 'Temperature' name of column with Temperature
    :param ConsumptionName default 'Consumption' name of column with consumption
    :param TimeName default 'Date' name of column with time
    :return: a dictionary with Thermosensibilite, and a panda data frame with two new columns NTS_C and TS_C
    '''

    return (ConsoSeparee_df, Thermosensibilite)