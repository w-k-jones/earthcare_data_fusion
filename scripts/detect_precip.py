import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess
import tobac

auth = earthaccess.login(persist=True)

save_path = pathlib.Path.home() / "my-public-bucket" / "orcestra" / "features"
save_path.mkdir(exist_ok=True, parents=True)

def detect_features_imerg(start_date, end_date):
    results = earthaccess.search_data(
        short_name="GPM_3IMERGHH", 
        temporal=(start_date, end_date),
    )

    fileobjects = earthaccess.open(results)
    with xr.open_mfdataset(
        fileobjects, group="Grid", combine="nested", concat_dim="time"
    ) as imerg_ds:
        labels, features = tobac.feature_detection_multithreshold(
            imerg_ds.precipitation.sel(lat=slice(-40, 40)), 
            dxy=11100, 
            threshold=[5, 10, 20],
            return_labels=True, 
            statistic=dict(
                max_precip=np.nanmax
            )
        )

    features_dt = xr.DataTree(
        children=dict(
            labels=xr.DataTree(labels.to_dataset()), 
            features=xr.DataTree(xr.Dataset.from_dataframe(features))
        )
    )

    return features_dt

dates = pd.date_range(
    pd.Timestamp(year=2024,month=8,day=10), 
    pd.Timestamp(year=2024,month=10,day=1),
    freq="1d"
)

for start_date, end_date in zip(dates, dates[1:]):
    print(start_date)
    detect_features_imerg(start_date, end_date).to_netcdf(
        save_path / f'detected_features_s{start_date.strftime("%Y%m%d")}_e{end_date.strftime("%Y%m%d")}.nc'
    )
    