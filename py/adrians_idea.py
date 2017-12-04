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

def make_zones_and_gaps(hyst):
    zone_edges = np.nonzero(np.diff(hyst.astype(int)))[0]
    if len(zone_edges)%2 == 0:
        zones = zone_edges.reshape(-1, 2)
        gaps = zone_edges[1:-1].reshape(-1, 2)
    else:
        zones = zone_edges[:-1].reshape(-1, 2)
        gaps = zone_edges[1:].reshape(-1, 2)
    
    return zones, gaps


def calc_zone_sizes(zones):
    zone_sizes = pd.Series(zones[:,1] - zones[:,0])
    return zone_sizes

def calc_parktimes(zones, dft):
    starts = dft.index.values[zones[:,0]]
    ends = dft.index.values[zones[:,1]]
    return pd.DataFrame(data={
        "start": starts,
        "stop": ends,
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

s = slice("2014-01-21","2014-02-01")
dftr = dft.resample("15S", how="first")

burst = dftr.v_max >= 50

"""
plt.ion()
plt.plot(dftr.v_max[s],'.')
plt.plot(burst[s]*50,'r-')
"""

# check if at least two bursts in 20 min window
b_per_20min = pd.rolling_sum(burst, window=20*4, min_periods=int(20*4/10))

from tqdm import tqdm
hours = b_per_20min.index.hour 
p = np.zeros_like(b_per_20min, dtype=np.bool)
p4 = np.zeros_like(b_per_20min, dtype=np.int)
park_until = -1

ppps = [20, 30, 40, 50, 60]
b = np.zeros((len(burst), len(ppps)))
for i, pppi in enumerate(ppps):
    b[:,i] = pd.rolling_sum(burst, window=pppi*4, min_periods=int(pppi*4/10))


pppi = 0
for i in tqdm(range(len(b_per_20min))):
    ppp = ppps[pppi]
    x = b[i, pppi]
    if i <= park_until:
        p[i] = True
        #if x >= 1:
        #    park_until = i+20*4
    
    if x >= 2:
        # currently not parking
        if not p[i] and p[i-(60*4+1):i-1].any():
            # parked already during last hour:
            if pppi < len(ppps)-1:
                pppi += 1
        park_until = i+ppp*4
    if x < 1:
        park_until = i

    if hours[i-1:i] < 12 and hours[i:i+1] >= 12:
        pppi = 0

    p4[i] = ppp

p = pd.Series(p, index=b_per_20min.index)
p4 = pd.Series(p4, index=b_per_20min.index)

#more_t_2b =rma >= 2 
park = pd.Series(hyst(b_per_20min, th_lo=0, th_hi=2), index=b_per_20min.index)
#rma = pd.rolling_max(b_per_20min, window=20*4, min_periods=int(20*4/10))
#rmi = pd.rolling_min(b_per_20min, window=20*4, min_periods=int(20*4/10))
#park = (rma >= 2) | (rmi >= 1)

plt.ion()
plt.plot(dftr.v_max[s],'b:')
#plt.plot(burst[s]*50,'g.:')
plt.plot(b_per_20min[s],'k-')
#plt.plot(rma[s],'-', lw=2, color="orange")
#plt.plot(rmi[s],'-', lw=2, color="green")
plt.plot(park[s]*50,'r-', lw=2)
plt.plot(p[s]*51,'m-', lw=2)
#plt.plot(p4[s],'g-', lw=4)


#plt.plot(more_t_2b[s]*50,'m-')

#plt.ylim(0,10)
plt.grid()

zones, gaps = make_zones_and_gaps(park)
park_times = calc_parktimes(zones, dftr)
gap_times = calc_parktimes(gaps, dftr)
park_durations = park_times.stop - park_times.start
gap_durations = gap_times.stop - gap_times.start

print("park_durations")
print(park_durations.describe())
print(park_durations.sum())

print("gap_durations")
print(gap_durations.describe())
print(gap_durations.sum())

print(70*"-")

zones, gaps = make_zones_and_gaps(p)
park_times = calc_parktimes(zones, dftr)
gap_times = calc_parktimes(gaps, dftr)
park_durations = park_times.stop - park_times.start
gap_durations = gap_times.stop - gap_times.start

print("park_durations")
print(park_durations.describe())
print(park_durations.sum())

print("gap_durations")
print(gap_durations.describe())
print(gap_durations.sum())

