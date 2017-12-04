# coding: utf-8
from glob import glob
from tqdm import tqdm
import pandas as pd
from fact.auxservices import AuxService


def combine_fits_to_h5(
    glob_expr='aux/*.MAGIC_WEATHER_DATA.fits'
):
    df = pd.concat(
        [
            AuxService.read_file(path)
            for path in tqdm(sorted(glob(glob_expr)))
        ],
        ignore_index=True
    )
    df['Time'] = pd.to_datetime(df.Time, unit='d')
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)
    df.to_hdf("MAGIC_WEATHER_DATA.h5", "MAGIC_WEATHER_DATA")

if __name__ == "__main__":
    combine_fits_to_h5()
