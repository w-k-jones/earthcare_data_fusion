import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import antimeridian
import pyproj
import shapely
import stratify

from scipy.spatial import ConvexHull
from sklearn.neighbors import BallTree

rad_1km = np.radians(360)/4e4


def regrid_height(ds, heights):
    new_ds = ds[[]].copy().assign_coords(ds.coords)
    new_ds = new_ds.assign_coords(
        height=xr.DataArray(
            heights, 
            dims=("height",), 
            attrs=dict(
                long_name="Height", 
                definition="Height of pixel centers above WGS84 ellipsoid", 
                units="m", 
            )
        )
    )
    for var in ds.data_vars:
        da = ds[var]
        if "height" in da.dims:
            new_ds[var] = (
                ("along_track", "height"), 
                stratify.interpolate(
                    new_ds.height.values,
                    ds.height.fillna(-np.inf).values,
                    da.values,
                    axis=1,
                    rising=False
                )
            )
            new_ds[var] = new_ds[var].assign_attrs(ds[var].attrs)
        elif "height_layer" in da.dims:
            new_ds[var] = (
                ("along_track", "height"), 
                stratify.interpolate(
                    new_ds.height.values,
                    ds.height_layers.fillna(-np.inf).values,
                    da.values,
                    axis=-1,
                    rising=False
                ).squeeze()
            )
            new_ds[var] = new_ds[var].assign_attrs(ds[var].attrs)
        elif "height_level" in da.dims:
            new_ds[var] = (
                ("along_track", "height"), 
                stratify.interpolate(
                    new_ds.height.values,
                    ds.height_levels.fillna(-np.inf).values,
                    da.values,
                    axis=-1,
                    rising=False
                ).squeeze()
            )
            new_ds[var] = new_ds[var].assign_attrs(ds[var].attrs)
        else:
            new_ds[var] = da
    return new_ds


def select_vars(ds, data_vars, coords=None, rename=None):
    if coords is not None:
        ds = ds.set_coords(
            [coord for coord in coords if coord in ds.data_vars]
        )
    ds = ds[data_vars]
    if rename is not None:
        ds = ds.rename_vars({k:v for k, v in rename.items() if k in ds})
    
    return ds

# def load_regrid_concat_ec_granules(granules, products):
#     datasets = {}
    
#     for product, variables in products.items():
#         temp_ds = []
#         for granule in granules:
#             try:
#                 filename = search_ec_filename(product, granule[:-1], granule[-1])
#             except ValueError as e:
#                 pass
#             else:
    
#                 with read_ec_file(filename) as ds:
#                     ds = select_vars(
#                         ds, 
#                         variables, 
#                         coords=[
#                             "time", "latitude", "longitude", "latitude_active", "longitude_active", "height", "height_level", "height_layer"
#                         ], 
#                         rename=dict(
#                             latitude_active="latitude", longitude_active="longitude"
#                         )
#                     ).load()
#                     temp_ds.append(
#                         regrid_height(ds, np.arange(50, 2e4, 100)[::-1])
#                     )
#         if len(temp_ds):
#             datasets[product] = xr.concat(temp_ds, "along_track")

#     return datasets


# def colocate_earthcare(ds, locations):
#     search_ll_tree = BallTree(
#         np.radians(
#             np.stack(
#                 [
#                     ds.latitude.values, 
#                     ds.longitude.values,
#                 ], axis=1
#             )
#         ), 
#         metric="haversine", 
#     )

#     distances, neighbours = search_ll_tree.query(locations)

#     return ds.isel(along_track=neighbours.ravel()).where(
#         xr.DataArray(distances.ravel() < rad_1km, dims="along_track")
#     )

# def create_merged_datatree(granules, PRODUCTS, tracking_bounds):
#     datasets = load_regrid_concat_ec_granules(granules, PRODUCTS)
    
#     cpr_gdf = gpd.GeoDataFrame(
#         data=[], 
#         geometry=[
#             shapely.Point(lon, lat) for lon, lat in zip(
#                 datasets["CPR_FMR_2A"].longitude, datasets["CPR_FMR_2A"].latitude
#             )
#         ], 
#         crs="EPSG:4326"
#     )
#     merged_ds = datasets["CPR_FMR_2A"].isel(along_track=cpr_gdf.sjoin(
#         gpd.GeoDataFrame(geometry=[tracking_bounds], crs="EPSG:4326")
#     ).drop("index_right", axis=1).index)

#     if len(merged_ds.along_track)==0:
#         raise ValueError("No intersecting track")
    
#     ec_ll = np.radians(np.stack(
#         [
#             merged_ds.latitude.values, 
#             merged_ds.longitude.values,
#         ], axis=1
#     ))
    
#     merged_dt = xr.DataTree(
#         dataset=xr.Dataset(coords=merged_ds.coords), 
#         children={
#             k:(xr.DataTree(colocate_earthcare(v, ec_ll)) if k!="CPR_FMR_2A" else xr.DataTree(merged_ds))
#             for k, v in datasets.items()
#         }
#     )
#     merged_dt.attrs["granules"] = " ".join(granules)
    
#     return merged_dt

# def colocate_tracks(merged_dt, tracking_path):
#     track_mask_filenames = []
#     for d in pd.date_range(
#         pd.Timestamp(merged_dt.time.min().values).floor("1d"), 
#         pd.Timestamp(merged_dt.time.max().values).floor("1d"), 
#         freq="1d"
#     ):
#         track_mask_filenames.extend(
#             sorted(list(tracking_path.rglob(f'detected_dccs_*S{d.strftime("%Y%m%d_%H%M%S")}*.nc')))
#         )

#     tracks_ds = xr.open_mfdataset(
#         track_mask_filenames, combine="nested", concat_dim="t", 
#         preprocess=lambda ds: ds[["core_label", "thick_anvil_label", "thin_anvil_label"]]
#     )

#     # SEVIRI projection
#     proj = pyproj.Proj('+proj=geos +lon_0 +h=035785831.0 +x_0=0 +y_0=0')
#     x, y = proj(merged_dt.longitude, merged_dt.latitude,)

#     for var, da in tracks_ds.sel(
#         x=xr.DataArray(x, dims="along_track"), 
#         y=xr.DataArray(y, dims="along_track"), 
#         t=merged_dt.time,
#         method="nearest",
#     ).reset_coords(drop=True).data_vars.items():
#         merged_dt[var] = da

#     merged_dt.attrs["track_mask_filenames"]=" ".join([f.name for f in track_mask_filenames])

#     return merged_dt