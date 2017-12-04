# coding: utf-8
from astropy.io import fits
from glob import iglob
from astropy.table import Table
from tqdm import tqdm
import pandas as pd

file_paths = iglob("/fact/aux/201[456]/*/*/*.MAGIC_WEATHER_DATA.fits")
dfs = []
for fp in tqdm(file_paths):
  try:
    dfs.append( Table.read(fp).to_pandas() )
  except ValueError:
    pass
df = pd.concat(dfs)
del dfs
df.to_hdf("MAGIC_WEATHER_DATA.h5", "MAGIC_WEATHER_DATA")

