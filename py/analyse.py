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

df = pd.read_hdf("MAGIC_WEATHER_DATA.h5")

# get rid of long runs of zeros in the data,
# 

v_is_zero = (df.v == 0).astype(int)
starts = np.where(v_is_zero.diff() > 0)[0]
ends = np.where(v_is_zero.diff() < 0)[0]

lengths = ends - starts

# there are 18 'runs' of more than 200 zeros in a row
print(lengths[lengths > 200].shape)
s3 = starts[np.where(lengths > 200)[0]][3]
e3 = ends[np.where(lengths > 200)[0]][3]
s = slice(s3-1000, e3+1000)
"""
plt.plot(df.Time[s], df.v[s], '.:')

plt.figure()
plt.hist(lengths, bins=np.arange(0,200), log=True, histtype="stepfilled")
"""
# looking at this hostgram, I decide everything with more than
# 30 zeros in a row, is frozen.
# I checked for some of these, that the Temperature is low.
# and the humidity was high. not for all. might be a nice plot.

df["frozen"] = False
frozen_starts = starts[np.where(lengths > 30)[0]]
frozen_ends = ends[np.where(lengths > 30)[0]]
for s,e in zip(frozen_starts, frozen_ends):
    df["frozen"].iloc[s:e] = True

df = df[~df.frozen]
df = df.drop("frozen", axis=1)

# now I want to find the 'typical' delta_v/step
# excluding long breaks or so.
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

delta_v_suitable = delta_t < 40
delta_v = df.v.diff().abs()
delta_v[~delta_v_suitable] = np.nan

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
    return pd.Series(ends - starts)

dft = df.set_index("Time")
dft["v_mean10"] = pd.rolling_mean(dft.v, 10)
dft["v_median5"] = pd.rolling_median(dft.v, 10)
dft["v_max10"] = pd.rolling_max(dft.v, 80)

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





"""
limit = 50
for lower_limit in [40]:
    hyst1 = hyst(dft.v_max10.values, lower_limit, limit)
    name = "hyst_rolling_max{0:d}_{1:d}".format(limit, lower_limit)
    dft[name] = hyst1
    zones = make_zones(hyst1)
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    print("parktime:", calc_parktimes(zones, dft).sum())
    print("-"*60)

limit = 50
lower_limit = 15
hyst1 = hyst(dft.v.values, lower_limit, limit)
name = "hyst_{0:d}_{1:d}".format(limit, lower_limit)
dft[name] = hyst1
zones = make_zones(hyst1)
zone_sizes = calc_zone_sizes(zones)
print(name)
print(limit, lower_limit)
print(zone_sizes.describe())
print("parktime:", calc_parktimes(zones, dft).sum())
print("-"*60)

s = slice("2014-01-15","2014-01-28")
plt.plot(
    dft.v[s],
    'b-', 
    label="wind measurement"
)
#plt.plot(dft.v_median5[s],'k-')
plt.plot(
    dft.v_max10[s],
    'm-', 
    lw=1,
    label="running maximum"
)

dft["rolling_max_gt_50"] = dft.v_max10 > 50


plt.plot(
    dft.rolling_max_gt_50[s]*50, 
    'r-', 
    lw=4,
    label="park! high wind"
)
plt.grid()
plt.legend(loc="upper right")

parktimes = calc_parktimes(make_zones(dft["rolling_max_gt_50"]), dft)
plt.figure()
plt.hist(
    parktimes.astype(int)/3600e9, 
    bins=np.arange(0, 24, 0.2), 
    log=True, 
    histtype="stepfilled"
)
plt.xlabel("parktime in hours")
plt.title("total parktime in 2014-today: "+str(calc_parktimes(zones, dft).sum()))

st = '\n'.join(str(parktimes.describe()).split("\n")[:-1])
plt.text(2, 1e1, st, family="monospace")

plt.xticks(np.arange(24))
"""
"""
limit = 35
for lower_limit in [17, 21, 25]:
    hyst1 = hyst(dft.v_mean10.values, lower_limit, limit)
    name = "hyst_mean_{0:d}_{1:d}".format(limit, lower_limit) 
    dft[name] = hyst1
    zones = make_zones(hyst1)
    zone_sizes = calc_zone_sizes(zones)
    print(name)
    print(limit, lower_limit)
    print(zone_sizes.describe())
    print("parktime:", calc_parktimes(zones, dft).sum())
    print("-"*60)
"""

"""
plt.plot(dft.v[s],'b:')
plt.plot(dft.v_mean10[s],'k:')
plt.plot(dft["hyst_50_15"][s]*50, 'r-')
plt.plot(dft["hyst_mean_35_21"][s]*35, 'm-', lw=2)

choice = "hyst_mean_35_21"
zones = make_zones(dft[choice])
starts = dft.index.values[zones[:,0]]
ends = dft.index.values[zones[:,1]]

for s,e in zip(starts, ends):
  s = slice(s,e)
  d = dft[s]
  plt.cla()
  plt.plot(dft.v[s],'b:')
  plt.plot(dft.v_mean10[s],'k:')
  plt.plot(dft[choice][s]*35, 'm-', lw=2)
  plt.draw()
  input("next?")
"""
