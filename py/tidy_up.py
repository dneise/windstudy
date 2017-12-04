import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#plt.ion()
from matplotlib.colors import LogNorm

def hyst(x, th_lo, th_hi, initial = False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)



def remove_frozen(df, length_limit=30):
    """
    get rid of long runs of zeros in the data,
    
    I looked at the length-histogram, see `length`
    below. I decided everything with more than
    30 zeros in a row, is frozen.
    I checked for some of these, that the Temperature is low.
    and the humidity was high. not for all though. might be a nice plot.
    """
 
    v_is_zero = (df.v == 0).astype(int)
    starts = np.where(v_is_zero.diff() > 0)[0]
    ends = np.where(v_is_zero.diff() < 0)[0]
    lengths = ends - starts   

    df["frozen"] = False
    frozen_starts = starts[np.where(lengths > length_limit)[0]]
    frozen_ends = ends[np.where(lengths > length_limit)[0]]
    for s,e in zip(frozen_starts, frozen_ends):
        df["frozen"].iloc[s:e] = True

    df = df[~df.frozen]
    df = df.drop("frozen", axis=1)

    return df



def add_delta_t(df):
    delta_t = (df.Time.diff().astype(int) / 1e9)
    # delta_t[0] was NaT but is now a funny float.
    delta_t[0] = np.nan
    df["delta_t"] = delta_t
    # describe delta_t
    """
    count    3717241.000000
    mean          20.321321
    std          267.786783
    min            1.000000
    25%           14.000000
    50%           14.000000
    75%           29.999999
    max       235381.000000
    """
    # I exclude everyhing < 40seconds, i.e. I allow one missing report.
    return df

def add_delta_v(df, suitable_cut_in_seconds=40):
    """
    I want to find the typical noise delta_v per step.

    Sometimes the time between samples is very long.
    Thus the delta_v becomes very small.
    """

    # now I want to find the 'typical' delta_v/step
    # excluding long breaks or so.

    delta_v_suitable = df.delta_t < suitable_cut_in_seconds
    delta_v = df.v.diff().abs()
    delta_v[~delta_v_suitable] = np.nan

    df["delta_v"] = delta_v
    """
    plt.hist(delta_v[~np.isnan(delta_v)], log=True, histtype="stepfilled", bins=100)

    """
    """
    In [21]: delta_v.describe()
    Out[21]: 
    count    3715288.000000
    mean           2.701175
    std            3.120213
    min            0.000000
    25%            0.700000
    50%            1.700000
    75%            3.600000
    max           94.799995
    Name: v, dtype: float64

    """
    """
    g = (~np.isnan(delta_v)) & (df.v > 45)
    plt.hist2d(df.v[g], delta_v[g], cmap="viridis", norm=LogNorm())
    """

    """
    In [7]: delta_v[df.v > 40].describe()
    Out[7]: 
    count    34375.000000
    mean        11.548643
    std          8.034820
    min          0.000000
    25%          5.199997
    50%         10.400002
    75%         16.500000
    max         94.799995
    Name: v, dtype: float64

    """
    return df


def make_zones(hyst):
    zone_edges = np.nonzero(np.diff(hyst.astype(int)))[0]
    try:
        zones = zone_edges.reshape(-1, 2)
    except ValueError:
        zones = zone_edges[:-1].reshape(-1, 2)
    return zones

def calc_zone_sizes(zones):
    zone_sizes = pd.Series(zones[:,1] - zones[:,0])
    return zone_sizes

def calc_parktimes(zones, dft):
    starts = dft.index.values[zones[:,0]]
    ends = dft.index.values[zones[:,1]]
    return pd.DataFrame(data={
        "park_start": starts,
        "park_stop": ends,
        })
    



def analysis(df):
    limit = 50
    lower_limit = 15
    name = "pure_hyst_{0:d}_{1:d}".format(limit, lower_limit)
    dft[name] = hyst(dft.v.values, lower_limit, limit)
    zones = make_zones(dft[name])
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    parktimes = calc_parktimes(zones, dft)
    print("parktime:", parktimes.sum())
    print("-"*60)

    plt.figure(figsize=(11, 2.6))
    plt.hist(
        parktimes.astype(int)/3600e9, 
        bins=np.arange(0, 24, 0.2), 
        log=True, 
        histtype="stepfilled"
    )
    plt.ylim(1e-1, 1e3)
    plt.xlabel("parktime in hours")
    plt.title(name + "   $\Sigma$ parktime" + str(calc_parktimes(zones, dft).sum()))
    st = '\n'.join(str(parktimes.describe()).split("\n")[:4])
    st2 = '\n'.join(str(parktimes.describe()).split("\n")[4:-1])
    plt.text(2, 1e1, st, family="monospace")
    plt.text(12, 1e1, st2, family="monospace")
    plt.xticks(np.arange(24))
    plt.savefig(name+".png")
    plt.close("all")


    plt.figure(figsize=(11, 2.6))
    plt.title(name)
    s = slice("2014-01-21","2014-02-01")
    plt.plot(
        dft.v[s],
        'b-', 
        label="wind measurement"
    )
    plt.plot(
        dft[name][s]*50, 
        'r-', 
        lw=4,
        label="park! high wind"
    )
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(name+"_example.png")
    plt.close("all")


    limit = 50
    lower_limit = 40
    name = "MAGIC_w_hyst_{0:d}_{1:d}".format(limit, lower_limit)
    dft[name] = hyst(dft.v_max10.values, lower_limit, limit) 
    zones = make_zones(dft[name])
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    parktimes = calc_parktimes(zones, dft)
    print("parktime:", parktimes.sum())
    print("-"*60)

    plt.figure(figsize=(11, 2.6))
    plt.hist(
        parktimes.astype(int)/3600e9, 
        bins=np.arange(0, 24, 0.2), 
        log=True, 
        histtype="stepfilled"
    )
    plt.ylim(1e-1, 1e3)
    plt.xlabel("parktime in hours")
    plt.title(name + "   $\Sigma$ parktime" + str(calc_parktimes(zones, dft).sum()))
    st = '\n'.join(str(parktimes.describe()).split("\n")[:4])
    st2 = '\n'.join(str(parktimes.describe()).split("\n")[4:-1])
    plt.text(2, 1e1, st, family="monospace")
    plt.text(12, 1e1, st2, family="monospace")
    plt.xticks(np.arange(24))
    plt.savefig(name+".png")
    plt.close("all")

    plt.figure(figsize=(11, 2.6))
    plt.title(name)
    s = slice("2014-01-21","2014-02-01")
    plt.plot(
        dft.v[s],
        'b-', 
        label="wind measurement"
    )
    plt.plot(
        dft[name][s]*50, 
        'r-', 
        lw=4,
        label="park! high wind"
    )
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(name+"_example.png")
    plt.close("all")



    limit = 50
    name = "MAGIC_{0:d}".format(limit)
    dft[name] = dft.v_max10 > 50
    zones = make_zones(dft[name])
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    parktimes = calc_parktimes(zones, dft)
    print("parktime:", parktimes.sum())
    print("-"*60)

    plt.figure(figsize=(11, 2.6))
    plt.hist(
        parktimes.astype(int)/3600e9, 
        bins=np.arange(0, 24, 0.2), 
        log=True, 
        histtype="stepfilled"
    )
    plt.ylim(3e-1, 1e3)
    plt.xlabel("parktime in hours")
    plt.title(name + "   $\Sigma$ parktime" + str(calc_parktimes(zones, dft).sum()))
    st = '\n'.join(str(parktimes.describe()).split("\n")[:4])
    st2 = '\n'.join(str(parktimes.describe()).split("\n")[4:-1])
    plt.text(2, 1e1, st, family="monospace")
    plt.text(12, 1e1, st2, family="monospace")
    plt.xticks(np.arange(24))
    plt.savefig(name+".png")
    plt.close("all")


    plt.figure(figsize=(11, 2.6))
    plt.title(name)
    s = slice("2014-01-21","2014-02-01")
    plt.plot(
        dft.v[s],
        'b-', 
        label="wind measurement"
    )
    plt.plot(
        dft[name][s]*50, 
        'r-', 
        lw=4,
        label="park! high wind"
    )
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(name+"_example.png")
    plt.close("all")


def pure_hist(dft):
    limit = 50
    lower_limit = 15
    name = "pure_hyst_{0:d}_{1:d}".format(limit, lower_limit)
    dft[name] = hyst(dft.v.values, lower_limit, limit)
    zones = make_zones(dft[name])
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    parktimes = calc_parktimes(zones, dft)
    print("parktime:", parktimes.sum())
    print("-"*60)

    plt.figure(figsize=(11, 2.6))
    plt.hist(
        parktimes.astype(int)/3600e9, 
        bins=np.arange(0, 24, 0.2), 
        log=True, 
        histtype="stepfilled"
    )
    plt.ylim(1e-1, 1e3)
    plt.xlabel("parktime in hours")
    plt.title(name + "   $\Sigma$ parktime" + str(calc_parktimes(zones, dft).sum()))
    st = '\n'.join(str(parktimes.describe()).split("\n")[:4])
    st2 = '\n'.join(str(parktimes.describe()).split("\n")[4:-1])
    plt.text(2, 1e1, st, family="monospace")
    plt.text(12, 1e1, st2, family="monospace")
    plt.xticks(np.arange(24))
    plt.savefig(name+".png")
    plt.close("all")


    plt.figure(figsize=(11, 2.6))
    plt.title(name)
    s = slice("2014-01-21","2014-02-01")
    plt.plot(
        dft.v[s],
        'b-', 
        label="wind measurement"
    )
    plt.plot(
        dft[name][s]*50, 
        'r-', 
        lw=4,
        label="park! high wind"
    )
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(name+"_example.png")
    plt.close("all")


def parktimes_from_hysteresis(s, upper, lower):
    h = hyst(s.values, lower, upper)
    parktimes = calc_parktimes(make_zones(h), s)
    return parktimes

def plot_parktimes(parktimes, name):
    plt.figure(figsize=(11, 2.6))
    plt.hist(
        parktimes.astype(int)/3600e9, 
        bins=np.arange(0, 24, 0.2), 
        log=True, 
        histtype="stepfilled"
    )
    plt.ylim(1e-1, 1e3)
    plt.xlabel("parktime in hours")
    plt.title(name + "   $\Sigma$ parktime" + str(parktimes.sum()))
    st = '\n'.join(str(parktimes.describe()).split("\n")[:4])
    st2 = '\n'.join(str(parktimes.describe()).split("\n")[4:-1])
    plt.text(2, 1e1, st, family="monospace")
    plt.text(12, 1e1, st2, family="monospace")
    plt.xticks(np.arange(24))
    plt.savefig(name+".png")
    plt.close("all")



df = pd.read_hdf("MAGIC_WEATHER_DATA.h5")
df = remove_frozen(df)
#df = add_delta_t(df)
#df = add_delta_v(df, suitable_cut_in_seconds=40)

dft = df.set_index("Time")
#dft["v_mean10"] = pd.rolling_mean(dft.v, 10)
#dft["v_median5"] = pd.rolling_median(dft.v, 10)
#dft["v_max10"] = pd.rolling_max(dft.v, 80)

dftr = dft.resample("15S", how="first")
v_5min_mean = pd.rolling_mean(dftr.v, window=20, min_periods=10, center=True)
v_5min_std = pd.rolling_std(dftr.v, window=20, min_periods=10, center=True)
v_20min_max = pd.rolling_max(dftr.v, window=80, min_periods=40)

s = slice("2014-01-21","2014-02-01")

# time in minutes in arange(brackets)
for mean_window in np.arange(5, 10)*4:
    for max_window in np.arange(16, 40, 4)*4:
        average = pd.rolling_mean(dftr.v, window=mean_window, min_periods=mean_window/2, center=True)
        rolling_max = pd.rolling_max(average, window=max_window, min_periods=max_window/2)        

        plt.figure(figsize=(11, 7.5))
        plt.plot(dftr.v[s], 'o', label="measurement")
        plt.plot((rolling_max[s] > 40)*40, 'r-', label="roll max > 40")
        plt.grid()
        plt.legend(loc="upper right")
        plt.savefig("foo_plots/mean{0}_max{1}_limit_40.png".format(mean_window, max_window))
        plt.close("all")


"""
s = slice("2014-01-21","2014-02-01")
plt.figure(figsize=(11, 7.5))
plt.plot(dftr.v[s], 'o', label="measurement")
plt.plot(v_5min_mean[s], '.:', label="5min running mean")
plt.plot(v_20min_max[s], 'm-', label="20min running max")
plt.grid()
plt.legend(loc="upper right")
plt.savefig("example_mean_max.png")
plt.close("all")

s = slice("2014-01-22T17:00","2014-01-24T17:00")
plt.figure(figsize=(11, 7.5))
plt.plot(dftr.v[s], 'o', label="measurement")
plt.plot(v_5min_mean[s], '.:', label="5min running mean")
plt.plot(v_20min_max[s], 'm-', label="20min running max")
plt.grid()
plt.legend(loc="upper right")
plt.savefig("example_mean_max_two_days.png")
plt.close("all")


dftr.v.hist(histtype="stepfilled", log=True, bins=np.arange(0, 130, 1))
plt.title("2014 - today : direct 'v' from MAGIC_WEATHER")
plt.xlabel("wind speed [km/h]")
plt.savefig("distribution_of_v.png")
plt.close("all")

v_5min_mean.hist(histtype="stepfilled", log=True, bins=np.arange(0, 130, 1))
plt.title("2014 - today : 5minute running mean on 'v'")
plt.xlabel("wind speed [km/h]")
plt.savefig("distribution_of_v_5min_mean.png")
plt.close("all")


v_5min_std.hist(histtype="stepfilled", log=True, bins=np.arange(0, 40, .5))
plt.title("2014 - today : 5minute running std on 'v'")
plt.xlabel("wind speed [km/h]")
plt.savefig("distribution_of_v_5min_std.png")
plt.close("all")

v_20min_max.hist(histtype="stepfilled", log=True, bins=np.arange(0, 130, 1))
plt.title("2014 - today : 20 minute running max on 'v'")
plt.xlabel("wind speed [km/h]")
plt.savefig("distribution_of_v_20min_max.png")
plt.close("all")
"""

"""
parktimes = {}

uppers = np.arange(30, 50, 0.5)
lowers = np.arange(20, 50, 0.5)

sum_parktimes = np.zeros((len(uppers), len(lowers)))

for iu, upper in enumerate(uppers):
    for il, lower in enumerate(lowers):
        #if lower >= upper:
            #sum_parktimes[iu, il] = np.nan    
            #continue
        p = parktimes_from_hysteresis(v_5min_mean, upper, lower)
        lengths = p.park_stop - p.park_start
        sum_parktimes[iu, il] = lengths.sum().total_seconds()

sum_parktimes /= (3600 * 24)
plt.ion()

extent = [
    uppers[0], #left
    uppers[-1]+1, #right
    lowers[0], #bottom
    lowers[-1]+1, #top
    ]
plt.figure()
plt.imshow(
    sum_parktimes.T, 
    cmap="viridis", 
    extent=extent, 
    interpolation="nearest",
    origin="lower")
cbar = plt.colorbar()

CS = plt.contour(
    *np.meshgrid(uppers, lowers), 
    sum_parktimes.T, 
    cmap="magma",
    linewidths=3,
    )
plt.clabel(CS, inline=1, fotsize=12, fmt='%.0f')

plt.ylabel("lower limit [km/h]")
plt.xlabel("upper limit [km/h]")
cbar.set_label("total days parked")
plt.title("parktime vs hysteresis(5min mean, upper, lower)")
plt.savefig("parktime_contour.png")
plt.close("all")
"""

"""
h1 = pd.Series(hyst(v_5min_mean, 39.2, 25), index=v_5min_mean.index)
h2 = pd.Series(hyst(v_5min_mean, 39.2, 33), index=v_5min_mean.index)
 
s = slice("2014-01-21","2014-02-01")

plt.figure(figsize=(11, 7.5))
plt.plot(dftr.v[s], 'o', label="measurement")
plt.plot(v_5min_mean[s], '.:', label="5min running mean")
plt.plot(h1[s]*50, 'r-', label="hyst(39.2, 25)")
plt.grid()
plt.legend(loc="upper right")
plt.savefig("foo1.png")
plt.close("all")


plt.figure(figsize=(11, 7.5))
plt.plot(dftr.v[s], 'o', label="measurement")
plt.plot(v_5min_mean[s], '.:', label="5min running mean")
plt.plot(h2[s]*50, 'r-', label="hyst(39.2, 33)")
plt.grid()
plt.legend(loc="upper right")
plt.savefig("foo2.png")
plt.close("all")
"""


# =====================================================
"""
v_around_50 = (dftr.v > 48) & (dftr.v <= 52)
In [12]: dftr.v[v_around_50].describe()
Out[12]: 
count    4369.000000
mean       49.974594
std         1.239560
min        48.200001
25%        49.099998
50%        50.000000
75%        51.000000
max        52.000000
Name: v, dtype: float64

In [13]: v_5min_mean[v_around_50].describe()
Out[13]: 
count    4366.000000
mean       39.635415
std         6.531970
min        18.200000
25%        35.172878
50%        39.391666
75%        43.680382
max        61.261538
Name: v, dtype: float64
"""

