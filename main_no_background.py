#! /usr/bin/env python
# coding=utf8

""" 
    main_no_background.py: Start visualization, no background.

    This script runs a visualization of the electricity prices over Europe
    for the period of April 2014, with controls to increase the installed wind and solar capacity,
    and change scenarios for the rest of the system.

    Wind and solar penetration are relative numbers (0-150%), with 100% corresponding to the scenario
    where the average gross production of renewables matches average demand. The installed capacities
    are 2015 numbers.

    Some commented out code refers to wind and solar backgrounds, which cannot be distributed due to
    licensing issues. Sorry.
"""

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import mpl_toolkits.basemap as bm
import defaults

from matplotlib import animation
from matplotlib.widgets import Slider, RadioButtons, Button
from helper_funcs import DiscreteSlider
from plot_classes import Production_Consumption_Plot, WindMap, Network_Plot, Pieplots, Priceplot

sns.set_style('ticks')

__author__ = "Tue V. Jensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Tue V. Jensen"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Tue V. Jensen"
__email__ = "tvjens@elektro.dtu.dk"
__status__ = "Eternal Prototype"


class expando(object):
    pass

###
# Setup
###

NUM_TIMESTEPS = defaults.NUM_TIMESTEPS
MINWIND = 0
MAXWIND = 20
WIND_LEVELS = 21
MINPRICE = -10
MAXPRICE = 10

WIND_CAP = 1200.  # GW of wind capacity at 100\% penetration
WIND_TODAY = 142. # Installed wind capacity today
SOLAR_CAP = 1048.  # GW of solar capacity at 100\% penetration
SOLAR_TODAY = 95. # Installed solar capacity today


class formatspec:
    def __init__(self, baseval, valfmt='{0:.00f} GW ({2:.00f}%)\n ({1:.00f}% vs today)', valtoday=1):
        self.baseval = baseval
        self.valfmt = valfmt
        self.valtoday = valtoday

    def format(self, x):
        return self.valfmt.format(self.baseval*x, 100.*x*self.baseval/self.valtoday, 100*x)

    def __mod__(self, x):
        # return self.format(x)
        return ''


WIND_SETTINGS = np.linspace(0, 1, 11)
SOLAR_SETTINGS = np.linspace(0, 1, 11)
wind_formatspec = formatspec(WIND_CAP, valtoday=WIND_TODAY)
solar_formatspec = formatspec(SOLAR_CAP, valtoday=SOLAR_TODAY)

###
# Plots
###


mymap = bm.Basemap(
    projection='cyl',
    llcrnrlat=defaults.LLCRNRLAT, llcrnrlon=defaults.LLCRNRLON,
    urcrnrlat=defaults.URCRNRLAT, urcrnrlon=defaults.URCRNRLON,
    resolution='l')

fig = plt.figure(figsize=(16, 9), dpi=80)
fig.patch.set_facecolor('white')
# Main map
ax1 = plt.subplot2grid((9, 6), (0, 0), colspan=4, rowspan=6)

contourholder = expando()
windticks = np.linspace(MINWIND, MAXWIND, WIND_LEVELS)
# windcontour = WindMap(ax1)
networkplot = Network_Plot(ax1)
pricecb = plt.colorbar(networkplot.nodeplot, ax=ax1, orientation='vertical', pad=0.05, aspect=30, extend='both', format='%.1f')
pricecb.set_label(U'Electricity price [€/MWh]')

coastlines = mymap.drawcoastlines(ax=ax1)
coastlines.set_alpha(0.5)
coastlines.set_zorder(10)

# Price in DK
ax2 = plt.subplot2grid((9, 6), (0, 4), rowspan=3, colspan=2)
thePriceplot = Priceplot(ax2)
# ax2.set_xlabel(u'Renewables in Europe [MW]')
ax2.set_ylabel(u'Mean European Price [€/MWh]')
ax2.set_ylim((defaults.MINPRICE, defaults.MAXPRICE*1.25))
sns.despine(ax=ax2, offset=3)

# Solar/wind use

ax3 = plt.subplot2grid((9, 6), (3, 4), rowspan=3, colspan=2)

ProdConPlot = Production_Consumption_Plot(ax3)

ax3.set_ylabel(u'Production/consumption [MWh]')
sns.despine(ax=ax3, offset=3)


# Renewable use

ax6 = plt.subplot2grid((9, 6), (6, 2), rowspan=3, colspan=4)
ax6.set_aspect(1)
ax6.axis('off')
pp = Pieplots(ax6)

plt.tight_layout()

###
# Controls
###

r = fig.canvas.get_renderer()


def wind_slider_change(*args, **kwargs):
    networkplot.update_wind(*args, **kwargs)
    ProdConPlot.update_wind(*args, **kwargs)
    thePriceplot.update_wind(*args, **kwargs)
    pp.update_wind(*args, **kwargs)
    for a in pp.get_artists():
        ax6.draw_artist(a)
    fig.canvas.blit(ax6.bbox)
    fig.canvas.blit(wind_slider_ax.bbox)
    wind_slider_text.set_text(wind_formatspec.format(wind_slider.discrete_val))
    wind_slider_text_ax.draw_artist(wind_slider_text)
    fig.canvas.blit(wind_slider_text_ax.bbox)


def solar_slider_change(*args, **kwargs):
    networkplot.update_solar(*args, **kwargs)
    ProdConPlot.update_solar(*args, **kwargs)
    thePriceplot.update_solar(*args, **kwargs)
    pp.update_solar(*args, **kwargs)
    for a in pp.get_artists():
        ax6.draw_artist(a)
    fig.canvas.blit(ax6.bbox)
    fig.canvas.blit(solar_slider_ax.bbox)
    solar_slider_text.set_text(solar_formatspec.format(solar_slider.discrete_val))
    solar_slider_text_ax.draw_artist(solar_slider_text)
    fig.canvas.blit(solar_slider_text_ax.bbox)

wind_slider_ax = plt.axes([0.08, 2.0/9, 1./3-0.16, 0.04])
wind_slider = DiscreteSlider(wind_slider_ax, 'Installed Wind', 0.0, 1.5, valinit=0.0, increment=0.1, valfmt=wind_formatspec, facecolor=sns.xkcd_rgb['sky blue'], dragging=True)
wind_slider.on_changed(wind_slider_change)
# wind_slider.valtext.set_bbox(dict(facecolor='white'))

wind_slider_text_ax = plt.axes([1./3-0.07, 2.0/9, 0.1, 0.04])
wind_slider_text_ax.axis('off')
wind_slider_text = wind_slider_text_ax.text(
    0.01, 0.02, wind_formatspec.format(wind_slider.discrete_val),
    verticalalignment='bottom', horizontalalignment='left',
    transform=wind_slider_text_ax.transAxes,
    color='black', fontsize=12, bbox=dict(facecolor='white'))

solar_slider_ax = plt.axes([0.08, 1.4/9, 1./3-0.16, 0.04])
solar_slider = DiscreteSlider(solar_slider_ax, 'Installed Solar', 0.0, 0.5, valinit=0.0, increment=0.05, valfmt=solar_formatspec, facecolor=sns.xkcd_rgb['pale yellow'], dragging=True)
solar_slider.on_changed(solar_slider_change)
# solar_slider.valtext.set_bbox(dict(facecolor='white'))

solar_slider_text_ax = plt.axes([1./3-0.07, 1.4/9, 0.1, 0.04])
solar_slider_text_ax.axis('off')
solar_slider_text = solar_slider_text_ax.text(
    0.01, 0.02, solar_formatspec.format(solar_slider.discrete_val),
    verticalalignment='bottom', horizontalalignment='left',
    transform=solar_slider_text_ax.transAxes,
    color='black', fontsize=12, bbox=dict(facecolor='white'))

scenario_dict = {
    'Today\'s system': 'base',
    'Nuclear is shut down': 'nuclear',
    'Demand increases by 15\%': 'demandincrease'
}
scenario_list = [
    'Today\'s system',
    'Nuclear is shut down',
    # u'CO2 price at 100 €/Ton',
    # 'Gas and Oil at 3x today\'s price',
    'Demand increases by 15\%'
]


scenario_select_ax = plt.axes([0.005, 0.1/9, 1./6, 1.1/9], aspect='equal', frameon=False)
scenario_select_radio = RadioButtons(scenario_select_ax, scenario_list, activecolor=sns.xkcd_rgb['dark grey'])

def scenario_change(val):
    newscen = scenario_dict[val]
    networkplot.update_scenario(newscen)
    ProdConPlot.update_scenario(newscen)
    thePriceplot.update_scenario(newscen)
    pp.update_scenario(newscen)
    for a in pp.get_artists():
        ax6.draw_artist(a)
    fig.canvas.blit(ax6.bbox)
    fig.canvas.blit(scenario_select_ax)

scenario_select_radio.on_clicked(scenario_change)

bg_list = ['Plot Wind', 'Plot Solar', 'Leave Blank']
bg_dict = {
    'Plot Wind': 'wind',
    'Plot Solar': 'solar',
    'Leave Blank': 'blank'}


# def set_plot_background(val):
#     windcontour.set_bg(bg_dict[val])
# set_plot_bg_ax = plt.axes([0.05+1./6, 0.1/9, 1./6, 1.1/9], aspect='equal', frameon=False)
# set_plot_bg_radio = RadioButtons(set_plot_bg_ax, bg_list, activecolor=sns.xkcd_rgb['dark grey'])
# set_plot_bg_radio.on_clicked(set_plot_background)


###
# Animated areas controlled here
###

def init():
    pass


def animate(i):
    # windout = windcontour.animate(i)
    ProdConPlot.animate(i)
    netout = networkplot.animate(i)
    thePriceplot.animate(i)
    return ProdConPlot.areas + [ProdConPlot.curtime_line] + \
        [coastlines] + netout + thePriceplot.areas + thePriceplot.lines + [thePriceplot.curtime_line] # windout + \

ani = animation.FuncAnimation(fig, animate, frames=NUM_TIMESTEPS, interval=100, repeat=True, repeat_delay=1000, blit=True)

# global animate
# animate = True


plt.show()
