# coding=utf8

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import defaults
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Rectangle

NUM_TIMESTEPS = defaults.NUM_TIMESTEPS


class Production_Consumption_Plot:
    def __init__(self, ax, width=24*4):
        self.ax = ax
        self.width = width
        self.load_data()
        self.plot_init()

    def load_data(self):

        # Load price data...
        # TEMP: Load fixed wind/solar data
        resultstore = pd.HDFStore('Data/tempstore.h5')
        self.energydf = resultstore['energydf']
        resultstore.close()

        resultstore = pd.HDFStore('Data/outputdatastore.h5')
        scenarios = resultstore['scenarios']
        windpens = resultstore['windpens']
        solarpens = resultstore['solarpens']
        self.resdict = {}
        for scenario in scenarios[:2]:
            for windpen in windpens:
                for solarpen in solarpens:
                    resdir = '/'.join(map(str, (scenario, windpen, solarpen)))
                    self.resdict[scenario, windpen, solarpen] = resultstore[resdir + '/energydf']
        resultstore.close()

        self.wind = windpens[0]
        self.solar = solarpens[0]
        self.scenario = scenarios[0]

        self.energydf = self.resdict[self.scenario, self.wind, self.solar]

        self.xtimes = np.arange(len(self.energydf.index))
        self.yload = self.energydf.demand.values/1000.
        self.ywind = self.energydf.windused.values/1000.
        self.ysolar = self.energydf.solarused.values/1000.

        pass

    def plot_init(self):
        self.areas = self.draw_netloadgraph()
        self.curtime_line = self.ax.axvline(0, c='k', alpha=0.5, lw=3, zorder=10)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, 400)

    def animate(self, i):
        self.curtime_line.set_xdata([i, i])
        left_border = min(max(i-self.width/2, 0)+self.width, len(self.xtimes))-self.width
        self.ax.set_xlim((left_border, left_border+self.width))
        return [self.ax]

    def update_wind(self, val):
        self.wind = float(str(val))
        self.load_new_data()
        self.update_areaplot()

    def update_solar(self, val):
        self.solar = float(str(val))
        self.load_new_data()
        self.update_areaplot()

    def update_scenario(self, val):
        self.scenario = val
        self.load_new_data()
        self.update_areaplot()

    def load_new_data(self):
        self.energydf = self.resdict[self.scenario, self.wind, self.solar]
        self.ywind = self.energydf.windused.values/1000.
        self.ysolar = self.energydf.solarused.values/1000.

    def update_areaplot(self):
        fbw, fbs, fbl = self.areas
        update_fill_between(fbw, self.xtimes, 0*self.ywind, self.ywind)
        update_fill_between(fbs, self.xtimes, self.ywind, self.ywind + self.ysolar)
        update_fill_between(fbl, self.xtimes, self.ywind + self.ysolar, self.yload)

    def draw_netloadgraph(self):
        graphs = []
        graphs.append(self.ax.fill_between(self.xtimes, 0*self.ywind, self.ywind, color=sns.xkcd_rgb['sky blue'], zorder=0))
        graphs.append(self.ax.fill_between(self.xtimes, self.ywind, self.ywind + self.ysolar, color=sns.xkcd_rgb['pale yellow'], zorder=1))
        graphs.append(self.ax.fill_between(self.xtimes, self.ywind + self.ysolar, self.yload, color=sns.xkcd_rgb['pale red'], zorder=-1))
        self.areas = graphs
        return graphs


def update_fill_between(fb, x, y1, y2):
    import numpy as np
    p = fb.get_paths()[0]
    # Length of fill_between
    l = (p.vertices.shape[0]-3)/2.
    if not (len(x) == l and len(y1) == l and len(y2) == l):
        print l
        print len(x)
        print len(y1)
        print len(y2)
        raise ValueError('Length of arrays do not correspond to fill_between\'s vertices.')
    p.vertices[:, 0] = np.concatenate(([x[0]], x, [x[-1]], x[::-1], [x[0]]))
    p.vertices[:, 1] = np.concatenate(([y2[0]], y1, [y2[-1]], y2[::-1], [y2[0]]))


class WindMap:
    def __init__(self, ax):
        self.ax = ax
        self.i = 0
        self.load_data()
        self.plot_init()

    def load_data(self):
        store = pd.HDFStore('Data/presstore.h5')
        self.solarp = store['/bg/solar']
        self.windp = store['/bg/wind']
        store.close()
        self.times = self.solarp.items.values
        self.lats = self.solarp.major_axis.values
        self.lons = self.solarp.minor_axis.values
        self.loncut = [self.lons.searchsorted(defaults.LLCRNRLON), self.lons.searchsorted(defaults.URCRNRLON)+1]
        self.latcut = [self.lats.searchsorted(defaults.LLCRNRLAT), self.lats.searchsorted(defaults.URCRNRLAT)+1]
        self.lats = self.lats[self.latcut[0]:self.latcut[1]]
        self.lons = self.lons[self.loncut[0]:self.loncut[1]]
        self.windp = self.windp.iloc[:, self.latcut[0]:self.latcut[1], self.loncut[0]:self.loncut[1]]
        self.solarp = self.solarp.iloc[:, self.latcut[0]:self.latcut[1], self.loncut[0]:self.loncut[1]]
        self.curtype = 'wind'
        self.update_curarray()
        self.zeroarray = np.zeros_like(self.windp.values[0, :-1, :-1])

    def plot_init(self):
        self.bgcontour = self.ax.pcolormesh(
            self.lons, self.lats,
            self.windp.values[self.i], cmap=plt.cm.Blues,
            zorder=1, vmin=defaults.MINWIND, vmax=defaults.MAXWIND)

    def set_bg(self, val):
        self.curtype = val
        self.update_curarray()
        self.update_cmap()
        self.update_range()
        self.update_plot()

    def animate(self, i):
        self.i = i
        self.update_plot()
        if self.curtype in ['wind', 'solar']:
            return [self.bgcontour]
        else:
            return []

    def update_plot(self):
        if self.curtype == 'wind' or self.curtype == 'solar':
            self.bgcontour.set_array(self.curarray[self.i, :-1, :-1].ravel())
        else:
            self.bgcontour.set_array(self.zeroarray.ravel())

    def update_curarray(self):
        if self.curtype == 'wind':
            self.curarray = self.windp.values
        elif self.curtype == 'solar':
            self.curarray = self.solarp.values
        else:
            self.curarray = self.zeroarray

    def update_cmap(self):
        if self.curtype == 'wind':
            self.bgcontour.set_cmap(plt.cm.Blues)
        elif self.curtype == 'solar':
            self.bgcontour.set_cmap(plt.cm.YlOrBr_r)
        else:
            self.bgcontour.set_cmap(plt.cm.gray)

    def update_range(self):
        if self.curtype == 'wind':
            self.bgcontour.set_norm(Normalize(defaults.MINWIND, defaults.MAXWIND))
        elif self.curtype == 'solar':
            self.bgcontour.set_norm(Normalize(defaults.MINSOLAR, defaults.MAXSOLAR))
        else:
            self.bgcontour.set_norm(Normalize(-1, 0))


class Network_Plot:
    def __init__(self, ax):
        self.ax = ax
        self.i = 0
        self.load_data()
        self.plot_init()

    def load_data(self):
        import networkx as nx
        from load_fnct import load_network
        self.nx = nx
        self.G = load_network()
        self.pos = nx.get_node_attributes(self.G, 'pos')
        self.nodelist = self.G.nodes()
        self.edgelist = self.G.edges()

        resultstore = pd.HDFStore('Data/outputdatastore.h5')
        scenarios = resultstore['scenarios']
        windpens = resultstore['windpens']
        solarpens = resultstore['solarpens']
        self.pricesdict = {}
        for scenario in scenarios[:2]:
            for windpen in windpens:
                for solarpen in solarpens:
                    resdir = '/'.join(map(str, (scenario, windpen, solarpen)))
                    self.pricesdict[scenario, windpen, solarpen] = resultstore[resdir + '/prices'][self.nodelist]
        resultstore.close()

        self.wind = windpens[0]
        self.solar = solarpens[0]
        self.scenario = scenarios[0]

        self.prices = self.pricesdict[self.scenario, self.wind, self.solar]

    def plot_init(self):
        nx = self.nx
        self.nodeplot = nx.draw_networkx_nodes(
            self.G, pos=self.pos,
            node_color=self.prices.values[0], nodelist=self.nodelist,
            node_size=30, zorder=10, cmap=plt.cm.RdYlGn_r,
            vmin=defaults.MINPRICE, vmax=defaults.MAXPRICE)
        darkpal = sns.dark_palette(sns.xkcd_rgb['blood red'], as_cmap=True)
        self.edgeplot = nx.draw_networkx_edges(
            self.G, pos=self.pos,
            edge_color=sns.xkcd_rgb['dark grey'],
            # edge_color=self.relflows.values[0], edge_cmap=darkpal,
            edgelist=self.edgelist, zorder=9,
            width=2.0, vmin=0, vmax=1)

    def animate(self, i):
        self.nodeplot.set_array(self.prices.values[i])
        # self.edgeplot.set_array(self.relflows.values[i])
        return [self.edgeplot, self.nodeplot]

    def update_wind(self, val):
        self.wind = float(str(val))
        self.prices = self.pricesdict[self.scenario, self.wind, self.solar]

    def update_solar(self, val):
        self.solar = float(str(val))
        self.prices = self.pricesdict[self.scenario, self.wind, self.solar]

    def update_scenario(self, val):
        self.scenario = val
        self.prices = self.pricesdict[self.scenario, self.wind, self.solar]


class Network_Plot_Edge_Flows:
    def __init__(self, ax):
        self.ax = ax
        self.i = 0
        self.load_data()
        self.plot_init()

    def load_data(self):
        import networkx as nx
        from load_fnct import load_network
        self.nx = nx
        self.G = load_network()
        self.pos = nx.get_node_attributes(self.G, 'pos')
        self.nodelist = self.G.nodes()
        self.edgelist = self.G.edges()

        resultstore = pd.HDFStore('Data/tempstore.h5')
        self.prices = resultstore['prices'][self.nodelist]
        self.flows = resultstore['flows'][self.edgelist]
        self.flowmaxes = resultstore['flowmaxes'][self.edgelist]
        resultstore.close()

        self.relflows = self.flows.abs()/self.flowmaxes

    def plot_init(self):
        nx = self.nx
        self.nodeplot = nx.draw_networkx_nodes(
            self.G, pos=self.pos,
            node_color=self.prices.values[0], nodelist=self.nodelist,
            node_size=30, zorder=10, cmap=plt.cm.RdYlGn_r,
            vmin=defaults.MINPRICE, vmax=defaults.MAXPRICE)
        darkpal = sns.dark_palette(sns.xkcd_rgb['blood red'], as_cmap=True)
        self.edgeplot = nx.draw_networkx_edges(
            self.G, pos=self.pos,
            # edge_color=sns.xkcd_rgb['dark grey'],
            edge_color=self.relflows.values[0], edge_cmap=darkpal,
            edgelist=self.edgelist, zorder=9,
            width=2.0, vmin=0, vmax=1)

    def animate(self, i):
        self.nodeplot.set_array(self.prices.values[i])
        self.edgeplot.set_array(self.relflows.values[i])
        return [self.edgeplot, self.nodeplot]

    def update_wind(self, val):
        pass

    def update_solar(self, val):
        pass

    def update_scenario(self, val):
        pass


class Pieplots:
    def __init__(self, ax):
        self.ax = ax
        ax.set_ylim([0, 10])
        ax.set_xlim([-10, 32])
        self.fontsize = 16
        self.initial_offset = 90
        self.formatspec = '{:.00f}%'
        self.load_data()
        self.plot_init()

    def load_data(self):
        self.bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

        resultstore = pd.HDFStore('Data/outputdatastore.h5')
        scenarios = resultstore['scenarios']
        windpens = resultstore['windpens']
        solarpens = resultstore['solarpens']
        resultstore.close()
        meanstore = pd.HDFStore('Data/meandatastore.h5')
        self.energydfdict = {}
        self.profitdict = {}
        for scenario in scenarios[:2]:
            for windpen in windpens:
                for solarpen in solarpens:
                    resdir = '/'.join(map(str, (scenario, windpen, solarpen)))
                    self.energydfdict[scenario, windpen, solarpen] = meanstore[resdir + '/energydfsum']
                    self.profitdict[scenario, windpen, solarpen] = meanstore[resdir + '/profitsum']
        meanstore.close()
        self.wind = windpens[0]
        self.solar = solarpens[0]
        self.scenario = scenarios[0]
        self.load_new_dfs()
        pass

    def update_wind(self, val):
        self.wind = float(str(val))
        self.load_new_dfs()
        self.update_plots()

    def update_solar(self, val):
        self.solar = float(str(val))
        self.load_new_dfs()
        self.update_plots()

    def update_scenario(self, val):
        self.scenario = val
        self.load_new_dfs()
        self.update_plots()

    def load_new_dfs(self):
        self.energydfsum = self.energydfdict[self.scenario, self.wind, self.solar]
        self.profitsum = self.profitdict[self.scenario, self.wind, self.solar]

        self.windused_rel = min(self.energydfsum.windused, self.energydfsum.windavailable)/(self.energydfsum.windavailable + 0.1)
        self.solarused_rel = min(self.energydfsum.solarused, self.energydfsum.solaravailable)/(self.energydfsum.solaravailable + 0.1)
        self.solardemand_rel = self.energydfsum.solarused/self.energydfsum.demand
        self.winddemand_rel = self.energydfsum.windused/self.energydfsum.demand
        self.renewdemand_rel = (self.energydfsum.windused + self.energydfsum.solarused)/self.energydfsum.demand
        pass

    def update_plots(self):
        self.update_wedges()
        self.update_rects()
        self.update_texts()

    def get_artists(self):
        artists = [
            self.ax.patch,
            self.windtitle, self.solartitle, self.loadtitle, self.costtitle,
            self.windused_W, self.windspill_W,
            self.solarused_W, self.solarspill_W,
            self.loadused_wind_W, self.loadused_solar_W, self.loadspill_W,
            self.loadrect, self.fuelrect, self.gensrect, self.windrect, self.solarrect, self.tsorect,
            self.ploadtext, self.pfueltext, self.pgenstext, self.pwindtext, self.psolartext, self.ploadtext,
            self.windtext, self.solartext, self.loadtext]
        return artists

    def update_wedges(self):
        wtheta = self.initial_offset+360*self.windused_rel
        stheta = self.initial_offset+360*self.solarused_rel
        wltheta = self.initial_offset+360*self.winddemand_rel
        sltheta = self.initial_offset+360*(self.winddemand_rel + self.solardemand_rel)

        self.windused_W.set_theta2(wtheta)
        self.windspill_W.set_theta1(wtheta)

        self.solarused_W.set_theta2(stheta)
        self.solarspill_W.set_theta1(stheta)

        self.loadused_wind_W.set_theta2(wltheta)
        self.loadused_solar_W.set_theta1(wltheta)
        self.loadused_solar_W.set_theta2(sltheta)
        self.loadspill_W.set_theta1(sltheta)

    def update_rects(self):
        ptot, pfuel, pgens, pwind, psolar = self.profitsum.load, self.profitsum.gencost, self.profitsum.genrevenue-self.profitsum.gencost, self.profitsum.wind, self.profitsum.solar
        hfuel, hgens, hwind, hsolar = self.rect_height*pfuel/ptot, self.rect_height*pgens/ptot, self.rect_height*pwind/ptot, self.rect_height*psolar/ptot

        self.update_rectangle(self.fuelrect, [3.3+20, 10+-9.7], 2, hfuel)
        self.update_rectangle(self.gensrect, [3.3+20, 10+-9.7+hfuel], 2, hgens,)
        self.update_rectangle(self.windrect, [3.3+20, 10+-9.7+hfuel+hgens], 2, hwind)
        self.update_rectangle(self.solarrect, [3.3+20, 10+-9.7+hfuel+hgens+hwind], 2, hsolar)
        self.update_rectangle(self.tsorect, [3.3+20, 10+-9.7+hfuel+hgens+hwind+hsolar], 2, self.rect_height-(hfuel+hgens+hwind+hsolar))

    def update_rectangle(self, rect, xy, w, h):
        rect.set_xy(xy)
        rect.set_height(h)
        rect.set_width(w)

    def update_texts(self):
        ptot, pfuel, pgens, pwind, psolar = self.profitsum.load, self.profitsum.gencost, self.profitsum.genrevenue-self.profitsum.gencost, self.profitsum.wind, self.profitsum.solar

        self.pfueltext.set_text(u'Fuel {:.00f}M€'.format(pfuel/1e6))
        self.pgenstext.set_text(u'Gens {:.00f}M€'.format(pgens/1e6))
        self.pwindtext.set_text(u'Wind {:.00f}M€'.format(pwind/1e6))
        self.psolartext.set_text(u'Solar {:.00f}M€'.format(psolar/1e6))
        self.ploadtext.set_text(u'{:.00f}M€'.format(ptot/1e6))

        self.windtext.set_text(self.formatspec.format(100*self.windused_rel))
        self.solartext.set_text(self.formatspec.format(100*self.solarused_rel))
        self.loadtext.set_text(self.formatspec.format(100*self.renewdemand_rel))

    def plot_init(self):
        self.rect_height = 7.6
        self.plot_wind_pie()
        self.plot_solar_pie()
        self.plot_load_pie()
        self.plot_cost_share()

    def plot_wind_pie(self):
        wind_center = [-5, 4]
        winduse_color = sns.xkcd_rgb['sky blue']
        windspill_color = sns.xkcd_rgb['pale blue']
        self.windused_W = Wedge(
            center=wind_center, r=3.9,
            theta1=self.initial_offset, theta2=self.initial_offset+360*self.windused_rel,
            facecolor=winduse_color)
        self.windspill_W = Wedge(
            center=wind_center, r=3.7,
            theta1=self.initial_offset+360*self.windused_rel, theta2=360+self.initial_offset,
            facecolor=windspill_color)
        self.ax.add_patch(self.windused_W)
        self.ax.add_patch(self.windspill_W)
        self.windtext = self.ax.annotate(self.formatspec.format(100*self.windused_rel), xy=[20+-10+5, 1], xytext=[-5, 1], va='bottom', ha='center', fontsize=self.fontsize, bbox=self.bbox_props)
        self.windtitle = self.ax.annotate('Wind used', xy=[-5, 8.1], xytext=[-5, 8.1], va='bottom', ha='center', fontsize=self.fontsize)
        pass

    def plot_solar_pie(self):
        solar_center = [5, 4]
        solaruse_color = sns.xkcd_rgb['pale yellow']
        solarspill_color = sns.xkcd_rgb['sunflower']
        self.solarused_W = Wedge(
            center=solar_center, r=3.9,
            theta1=self.initial_offset, theta2=self.initial_offset+360*self.solarused_rel,
            facecolor=solaruse_color)
        self.solarspill_W = Wedge(
            center=solar_center, r=3.7,
            theta1=self.initial_offset+360*self.solarused_rel, theta2=360+self.initial_offset,
            facecolor=solarspill_color)
        self.ax.add_patch(self.solarused_W)
        self.ax.add_patch(self.solarspill_W)
        self.solartext = self.ax.annotate(self.formatspec.format(100*self.solarused_rel), xy=[5, 1], xytext=[5, 1], va='bottom', ha='center', fontsize=self.fontsize, bbox=self.bbox_props)
        self.solartitle = self.ax.annotate('Solar used', xy=[5, 8.1], xytext=[5, 8.1], va='bottom', ha='center', fontsize=self.fontsize)
        pass

    def plot_load_pie(self):
        load_center = [-5+20, -6+10]
        solaruse_color = sns.xkcd_rgb['pale yellow']
        winduse_color = sns.xkcd_rgb['sky blue']
        loaduse_color = sns.xkcd_rgb['pale green']
        loadspill_color = sns.xkcd_rgb['pale brown']
        self.loadused_wind_W = Wedge(
            center=load_center, r=3.9,
            theta1=self.initial_offset, theta2=self.initial_offset+360*self.winddemand_rel,
            facecolor=winduse_color)
        self.loadused_solar_W = Wedge(
            center=load_center, r=3.9,
            theta1=self.initial_offset+360*self.winddemand_rel, theta2=self.initial_offset+360*self.winddemand_rel+360*self.solardemand_rel,
            facecolor=solaruse_color)
        self.loadspill_W = Wedge(
            center=load_center, r=3.7,
            theta1=self.initial_offset+360*self.renewdemand_rel, theta2=360+self.initial_offset,
            facecolor=loadspill_color)
        self.ax.add_patch(self.loadused_wind_W)
        self.ax.add_patch(self.loadused_solar_W)
        self.ax.add_patch(self.loadspill_W)
        self.loadtext = self.ax.annotate(self.formatspec.format(100*self.renewdemand_rel), xy=[-5+20, -9+10], xytext=[-5+20, -9+10], va='bottom', ha='center', fontsize=self.fontsize, bbox=self.bbox_props)
        self.loadtitle = self.ax.annotate('Load share', xy=[-5+20, 8.1-10+10], xytext=[-5+20, 8.1-10+10], va='bottom', ha='center', fontsize=self.fontsize)
        pass

    def plot_cost_share(self):
        ptot, pfuel, pgens, pwind, psolar = self.profitsum.load, self.profitsum.gencost, self.profitsum.genrevenue-self.profitsum.gencost, self.profitsum.wind, self.profitsum.solar
        hfuel, hgens, hwind, hsolar = self.rect_height*pfuel/ptot, self.rect_height*pgens/ptot, self.rect_height*pwind/ptot, self.rect_height*psolar/ptot

        self.loadrect = self.ax.add_patch(Rectangle([0.3+20, 10+-9.7], 2, self.rect_height, facecolor=sns.xkcd_rgb['pale red']))
        self.fuelrect = self.ax.add_patch(Rectangle([3.3+20, 10+-9.7], 2, hfuel, facecolor=sns.xkcd_rgb['brown']))
        self.gensrect = self.ax.add_patch(Rectangle([3.3+20, 10+-9.7+hfuel], 2, hgens, facecolor=sns.xkcd_rgb['pale brown']))
        self.windrect = self.ax.add_patch(Rectangle([3.3+20, 10+-9.7+hfuel+hgens], 2, hwind, facecolor=sns.xkcd_rgb['pale blue']))
        self.solarrect = self.ax.add_patch(Rectangle([3.3+20, 10+-9.7+hfuel+hgens+hwind], 2, hsolar, facecolor=sns.xkcd_rgb['pale yellow']))
        self.tsorect = self.ax.add_patch(Rectangle([3.3+20, 10+-9.7+hfuel+hgens+hwind+hsolar], 2, 7.6-(hfuel+hgens+hwind+hsolar), facecolor=sns.xkcd_rgb['dark grey']))

        self.pfueltext = self.ax.annotate(u'Fuel {:.00f}M€'.format(pfuel/1e6), xy=[20+5.5, 10+-9.7], xytext=[20+5.5, 10+-9.7], va='bottom', ha='left', fontsize=self.fontsize)
        self.pgenstext = self.ax.annotate(u'Gens {:.00f}M€'.format(pgens/1e6), xy=[20+5.5, 10+-9.7+1.9], xytext=[20+5.5, 10+-9.7+1.9], va='bottom', ha='left', fontsize=self.fontsize)
        self.pwindtext = self.ax.annotate(u'Wind {:.00f}M€'.format(pwind/1e6), xy=[20+5.5, 10+-9.7+2*1.9], xytext=[20+5.5, 10+-9.7+2*1.9], va='bottom', ha='left', fontsize=self.fontsize)
        self.psolartext = self.ax.annotate(u'Solar {:.00f}M€'.format(psolar/1e6), xy=[20+5.5, 10+-9.7+3*1.9], xytext=[20+5.5, 10+-9.7+3*1.9], va='bottom', ha='left', fontsize=self.fontsize)
        self.ploadtext = self.ax.annotate(u'{:.00f}M€'.format(ptot/1e6), xy=[20+0.3, 10+-9.7], xytext=[20+0.3, 10+-9.7], va='bottom', ha='left', fontsize=self.fontsize, bbox=self.bbox_props)
        self.costtitle = self.ax.annotate(u'Payments', xy=[23, 8.1], va='bottom', ha='center', fontsize=self.fontsize)
        pass


class Priceplot:
    def __init__(self, ax, width=24*4):
        self.ax = ax
        self.width = width
        self.load_data()
        self.plot_init()

    def load_data(self):

        resultstore = pd.HDFStore('Data/outputdatastore.h5')
        scenarios = resultstore['scenarios']
        windpens = resultstore['windpens']
        solarpens = resultstore['solarpens']
        resultstore.close()
        meanstore = pd.HDFStore('Data/meandatastore.h5')
        self.pricemeandict = {}
        self.pricestddict = {}
        self.priceqsdict = {}
        for scenario in scenarios[:2]:
            for windpen in windpens:
                for solarpen in solarpens:
                    resdir = '/'.join(map(str, (scenario, windpen, solarpen)))
                    self.pricemeandict[scenario, windpen, solarpen] = meanstore[resdir + '/pricemean']
                    self.pricestddict[scenario, windpen, solarpen] = meanstore[resdir + '/pricemean']
                    self.priceqsdict[scenario, windpen, solarpen] = meanstore[resdir + '/priceqs']
        meanstore.close()

        self.wind = windpens[0]
        self.solar = solarpens[0]
        self.scenario = scenarios[0]

        self.yprice = self.pricemeandict[self.scenario, self.wind, self.solar]
        self.xtimes = np.arange(len(self.yprice))
        self.ypricestd = self.pricestddict[self.scenario, self.wind, self.solar]
        self.ypriceqs = self.priceqsdict[self.scenario, self.wind, self.solar]
        pass

    def load_new_prices(self):
        self.yprice = self.pricemeandict[self.scenario, self.wind, self.solar]
        self.ypricestd = self.pricestddict[self.scenario, self.wind, self.solar]
        self.ypriceqs = self.priceqsdict[self.scenario, self.wind, self.solar]
        pass

    def plot_init(self):
        self.areas = self.draw_quantile_areas()
        self.lines = self.draw_pricegraph()
        self.curtime_line = self.ax.axvline(0, c='k', alpha=0.5, lw=3, zorder=10)
        self.ax.set_xlim(0, self.width)

    def animate(self, i):
        self.curtime_line.set_xdata([i, i])
        left_border = min(max(i-self.width/2, 0)+self.width, len(self.xtimes))-self.width
        self.ax.set_xlim((left_border, left_border+self.width))
        return [self.ax]

    def update_wind(self, val):
        self.wind = float(str(val))
        self.load_new_prices()
        self.update_priceplot()

    def update_solar(self, val):
        self.solar = float(str(val))
        self.load_new_prices()
        self.update_priceplot()

    def update_scenario(self, val):
        self.scenario = val
        self.load_new_prices()
        self.update_priceplot()

    def update_priceplot(self):
        lm, lplus, lminus = self.lines
        lm.set_ydata(self.yprice)
        lplus.set_ydata(self.yprice + self.ypricestd)
        lminus.set_ydata(self.yprice - self.ypricestd)
        a10, a25 = self.areas
        update_fill_between(a10, self.xtimes, self.ypriceqs.q10, self.ypriceqs.q90)
        update_fill_between(a25, self.xtimes, self.ypriceqs.q25, self.ypriceqs.q75)

    def draw_pricegraph(self):
        lines = []
        lines.append(self.ax.plot(self.xtimes, self.yprice, ls='-', lw=3, color=sns.xkcd_rgb['charcoal'])[0])
        lines.append(self.ax.plot(self.xtimes, self.yprice+self.ypricestd, ls='-', lw=2, color=sns.xkcd_rgb['charcoal'], alpha=0.5)[0])
        lines.append(self.ax.plot(self.xtimes, self.yprice-self.ypricestd, ls='-', lw=2, color=sns.xkcd_rgb['charcoal'], alpha=0.5)[0])
        self.lines = lines
        return lines

    def draw_quantile_areas(self):
        areas = []
        areas.append(self.ax.fill_between(self.xtimes, self.ypriceqs.q10, self.ypriceqs.q90, color=sns.xkcd_rgb['deep green'], alpha=0.3))
        areas.append(self.ax.fill_between(self.xtimes, self.ypriceqs.q25, self.ypriceqs.q75, color=sns.xkcd_rgb['deep green'], alpha=0.3))
        self.areas = areas
        return areas
