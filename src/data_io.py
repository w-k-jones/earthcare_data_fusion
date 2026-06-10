import pathlib
import requests
import fsspec
import pandas as pd
import geopandas as gpd
import xarray as xr
import antimeridian
import shapely
from pystac_client import Client


catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
catalog = Client.open(catalog_url)
EC_COLLECTION = ['EarthCAREL2Validated_MAAP']

CREDENTIALS_FILE = (pathlib.Path.home() / "credentials.txt" ).resolve()   # Insert the .txt path
io_params = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024
        }
    }
}

def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


# --- ESA MAAP API ---

def get_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    OFFLINE_TOKEN = creds.get("OFFLINE_TOKEN")
    CLIENT_ID = creds.get("CLIENT_ID")
    CLIENT_SECRET = creds.get("CLIENT_SECRET")
    # print(CLIENT_SECRET)

    if not all([OFFLINE_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": OFFLINE_TOKEN,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token

token = get_token()

fs = fsspec.filesystem(
    "https", 
    headers={"Authorization": f"Bearer {token}"}, 
    **io_params["fsspec_params"], 
)

def maap_search_to_gdf(search):
    df = pd.DataFrame(
        data={"stac":list(search.items())}
    )
    
    df["granule"] = [f.id[-6:] for f in df.stac]
    df["product"] = [f.id[9:19] for f in df.stac]
    df["baseline"] = [f.id[6:8] for f in df.stac]
    df["date"] = [
        pd.to_datetime(f.id[20:35], format="%Y%m%dT%H%M%S") for f in df.stac
    ]
    df["enclosure_h5"] = [f.assets.get('enclosure_h5').href for f in df.stac]
    df = df.sort_values(["date", "product"]).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df.drop("stac", axis=1), 
        geometry=[
            antimeridian.fix_line_string(
                shapely.LineString(row["stac"].geometry["coordinates"]), 
                great_circle=True,
            )
            for idx, row in df.iterrows()
        ], 
        crs="EPSG:4326"
    )

    return gdf


def search_ec_filename(product, orbit, frame):
    search = catalog.search(
        collections=EC_COLLECTION, 
        filter=f"(productType = '{product}') and orbitNumber = {orbit} and frame = '{frame}'", # For example filter by product type and orbitNumber. Use boolean logic for multi-filter queries
        method = 'GET', # This is necessary 
        max_items=1  # Adjust as needed, given the large amount of products it is recommended to set a limit if especially if you display results in pandas dataframe or similiar
    )
    items = list(search.items())
    if len(items):
        return items[0].assets.get('enclosure_h5').href

    raise ValueError(
        f'No EarthCARE files found for search {product=}, {orbit=}, {frame=}'
    )


from contextlib import contextmanager
@contextmanager
def read_ec_file(filename):
    try:
        f = fs.open(filename)
        dt = xr.open_datatree(
            f, 
            engine="h5netcdf", 
            **io_params["h5py_params"], 
        )
        ds = dt.ScienceData.to_dataset().assign_attrs(
            {
                k:v.item() for k, v in dt.HeaderData.FixedProductHeader.data_vars.items()
            }
        )
        yield ds
    finally:
        f.close()
        try:
            dt.close()
            ds.close()
        except UnboundLocalError:
            pass

def parse_dtime(ds):
    year = ds.Year.values
    wh_invalid = year == -9999
    year[wh_invalid] = 0
    time = (
        (ds.Year - 1970).astype("datetime64[Y]") 
        + (ds.Month - 1).astype("timedelta64[M]")
        + (ds.DayOfMonth - 1).astype("timedelta64[D]")
        + ds.Hour.astype("timedelta64[h]")
        + ds.Minute.astype("timedelta64[m]")
        + ds.Second.astype("timedelta64[s]")
        + ds.MilliSecond.astype("timedelta64[ms]")
    )
    time[wh_invalid] = np.datetime64(0, "Y")

    ds['time'] = time
    ds = ds.set_coords('time')
    #drop the variables we dont need 
    ds = ds.drop_vars(
        ['Year','Month','DayOfMonth','Hour','Minute','Second','MilliSecond','DayOfYear','SecondOfDay']
    )
    return ds


def read_dpr_l2a(filename, heavy=True):
    """
    This method unfolds all the groups into one combined xarray dataset for
    each radar (e.g., KuPR and KaPR). It will be lazily loaded to save on RAM.
    Note that this code was primarily developed for V7 DPR products. It will not 
    work for V6 data 

    """
    #######################################################################
    ################################ KuPR #################################
    #######################################################################

    if isinstance(filename, earthaccess.store.EarthAccessFile):
        dt = xr.open_datatree(filename, backend_kwargs={"phony_dims": "sort"}, decode_cf=False)
    else:
        dt = xr.open_datatree(filename, decode_cf=False, decode_times=False)
    geo = dt.FS.to_dataset()
    pre = dt.FS.PRE.to_dataset()
    slv = dt.FS.SLV.to_dataset()
    tim = dt.FS.ScanTime.to_dataset()

    #rename dims to proper names
    bad_dims = list(geo.dims)
    geo = geo.rename_dims({bad_dims[0]:'nscan',
                        bad_dims[1]:'nrayNS'})
    bad_dims = list(pre.dims)
    pre = pre.rename_dims({bad_dims[0]:'nscan',
                        bad_dims[1]:'nrayNS',
                        bad_dims[2]:'nfreq',
                        bad_dims[3]:'nbin'})
    bad_dims = list(slv.dims)
    slv = slv.rename_dims({bad_dims[0]:'nscan',
                        bad_dims[1]:'nrayNS',
                        bad_dims[2]:'nbin',
                        bad_dims[3]:'nfreq',
                        bad_dims[4]:'nNUBF'})
    bad_dims = list(tim.dims)
    tim = tim.rename_dims({bad_dims[0]:'nscan'})
    
    #if you want to load the whole thing in, turn the heavy flag on 
    if heavy: 
        ver = dt.FS.VER.to_dataset()
        srt = dt.FS.SRT.to_dataset()
        csf = dt.FS.CSF.to_dataset()
        exp = dt.FS.Experimental.to_dataset()
        flg = dt.FS.FLG.to_dataset()
        trg = dt.FS.TRG.to_dataset()

        #rename dims to proper names
        bad_dims = list(ver.dims)
        ver = ver.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'nbin',
                            bad_dims[3]:'nfreq',
                            bad_dims[4]:'nNP'})
        bad_dims = list(srt.dims)
        srt = srt.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'method',
                            bad_dims[3]:'foreBack',
                            bad_dims[4]:'nearFar',
                            bad_dims[5]:'nsdew'})
        bad_dims = list(csf.dims)
        csf = csf.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'nfreqHI'})
        bad_dims = list(exp.dims)
        exp = exp.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'nbinSZP',
                            bad_dims[3]:'nfreq'}) 
        bad_dims = list(flg.dims)
        flg = flg.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'nbin',
                            bad_dims[3]:'nfreq'})       

        bad_dims = list(trg.dims)
        trg = trg.rename_dims({bad_dims[0]:'nscan',
                            bad_dims[1]:'nrayNS',
                            bad_dims[2]:'nslope',})     

        #MERGE into one ds 
        ds = xr.merge([geo,pre,slv,ver,srt,csf,exp,flg,tim,trg])
    else:
        #MERGE into one ds 
        ds = xr.merge([geo,pre,slv,tim])
    

    #close uneeded xr datasets 
    dt.close()
    geo.close()
    pre.close()
    slv.close()
    tim.close()
    if heavy: 
      ver.close()
      srt.close()
      csf.close()
      exp.close()
      flg.close()
    
    #set lat,lon,height as the coords to allow for easy xr slicing
    ds = ds.set_coords(['Latitude','Longitude','height'])
    ds = parse_dtime(ds)
    return ds