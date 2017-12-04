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
    delta_t[0] = np.nan
    df["delta_t"] = delta_t
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
    return df



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

import time
def regularize_dataframe(df, rule, columname, span_limit=None):
    df_index = df.index.values.astype(int)
    # empty frame with desired index
    rs = pd.DataFrame(index=df.resample(rule).iloc[1:].index)

    rs_index = rs.index.values.astype(int)
    # array of indexes corresponding with closest timestamp after resample
    idx_after = np.searchsorted(df.index.values, rs.index.values)
    # values and timestamp before/after resample
    
    after = df[columname].values[idx_after]
    before = df[columname].values[idx_after - 1]
    after_time = df_index[idx_after]
    before_time = df_index[idx_after - 1]

    #calculate new weighted value
    span = after_time - before_time
    after_weight = (after_time - rs_index) / span
    before_weight = 1 - after_weight
    
    rs[columname] = before * after_weight + after * before_weight
    rs["span_in_seconds"] = span / 1e9
    rs["after"] = after
    rs["before"] = before

    if span_limit:
        rs[columname][rs.span_in_seconds > span_limit] = np.nan
    
    return rs
    
s = slice("2014-01-01","2014-03-01")

df = pd.read_hdf("MAGIC_WEATHER_DATA.h5")
df["v"] = df[["v","v_max"]].max(axis=1)
df.drop("v_max", axis=1, inplace=True)
df = remove_frozen(df)

dft = df.set_index("Time")

print("resampling")
dftr = regularize_dataframe(dft, "15S", "v", span_limit=60)
print("resampling .. done")

burst = dftr.v >= 50
b_per_20min = pd.rolling_sum(burst, window=20*4, min_periods=int(20*4/10))

dftr["dvdt_1"] = dftr.v.diff(periods=1)
dftr["dvdt_2"] = dftr.v.diff(periods=2)

plt.ion()
"""
plt.plot(dft.v[s], 'o:', label="orig")
plt.plot(dftr.v[s], '.:', label="interp")
plt.grid()
plt.legend()

fig , ax= plt.subplots(4, sharex=True)
ax[0].plot(dftr.v[s], 'o:', label="dftr.v")
ax[1].plot(dft.v[s], 'o:', label="dft.v")
ax[2].plot(np.abs(dftr.dvdt_1[s]), 'o:', label="dftr.dvdt_1")
ax[3].plot(np.abs(dftr.dvdt_2[s]), 'o:', label="dftr.dvdt_2")
for a in ax: 
    a.legend()
    a.grid()
"""

fig , ax= plt.subplots(2, sharex=True)
ax[0].plot(dftr.v[s], 'b.', label="windspeed km/h")
ax[1].plot(np.abs(dftr.dvdt_1[s]), 'g.', label="wind acceleration [km/h /15sec]")
for a in ax: 
    a.legend()
    a.grid()
plt.tight_layout()


plt.figure()
plt.hist(
    dftr.v, 
    bins=np.arange(0,120), 
    histtype="step", 
    log=True, 
    label="windspeed km/h"
)
plt.hist(
    np.abs(dftr.dvdt_1), 
    bins=np.arange(0,150), 
    histtype="step", 
    log=True, 
    label="abs(wind acceleration [km/h /15sec])"
)
plt.legend()
plt.grid()