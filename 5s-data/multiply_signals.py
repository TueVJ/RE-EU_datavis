import numpy as np
import pandas as pd

# TODO: Load up data from data dirs instead of from 

wind_layout = pd.read_csv('wind_layouts.csv').set_index('node')['uniform']
solar_layout = pd.read_csv('solar_layouts.csv').set_index('node')['uniform']

wind_signal = pd.read_csv('wind_signal_rel.csv', index_col='Time')
wind_forecast = pd.read_csv('wind_forecast_rel.csv', index_col='Time')

solar_signal = pd.read_csv('solar_signal_rel.csv', index_col='Time')
solar_forecast = pd.read_csv('solar_forecast_rel.csv', index_col='Time')

wind_signal.columns = wind_signal.columns.astype(int)
wind_forecast.columns = wind_forecast.columns.astype(int)
solar_signal.columns = solar_signal.columns.astype(int)
solar_forecast.columns = solar_forecast.columns.astype(int)

wind_signal_out = wind_signal.multiply(wind_layout)
wind_forecast_out = wind_forecast.multiply(wind_layout)
solar_signal_out = solar_signal.multiply(solar_layout)
solar_forecast_out = solar_forecast.multiply(solar_layout)

# Temp fix for bodged 'time' handling in optimization code
wind_signal_out.index = range(len(wind_signal_out))
wind_forecast_out.index = range(len(wind_forecast_out))
solar_signal_out.index = range(len(solar_signal_out))
solar_forecast_out.index = range(len(solar_forecast_out))

wind_signal_out.index.name = 'Time'
wind_forecast_out.index.name = 'Time'
solar_signal_out.index.name = 'Time'
solar_forecast_out.index.name = 'Time'

wind_signal_out.to_csv('wind_ts.csv')
wind_forecast_out.to_csv('wind_fc.csv')
solar_signal_out.to_csv('solar_ts.csv')
solar_forecast_out.to_csv('solar_fc.csv')

# Temp fix for load
load = pd.read_csv('load_orig.csv', index_col='Time')
load.index = range(len(load))
load.index.name = 'Time'
load.to_csv('load.csv')