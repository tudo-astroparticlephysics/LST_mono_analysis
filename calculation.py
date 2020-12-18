import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fact.io import read_h5py
from pyirf.cuts import evaluate_binned_cut

from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, AltAz, EarthLocation
import astropy.units as u
from astropy import table

from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator


erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


def calc_ontime(df):
    delta = np.diff(df.dragon_time.sort_values())
    delta = delta[np.abs(delta) < 10]
    return len(df) * delta.mean() * u.s


def calc_theta_off(source_coord: SkyCoord, reco_coord: SkyCoord, pointing_coord: SkyCoord, n_off=5):
    fov_frame = SkyOffsetFrame(origin=pointing_coord)
    source_fov = source_coord.transform_to(fov_frame)
    reco_fov = reco_coord.transform_to(fov_frame)
    
    r = source_coord.separation(pointing_coord)
    phi0 = np.arctan2(source_fov.lat, source_fov.lon).to_value(u.rad)
    
    theta_offs = []
    for off in range(1, n_off + 1):
        
        off_pos = SkyCoord(
            lon=r * np.cos(phi0 + 2 * np.pi * off / (n_off + 1)),
            lat=r * np.sin(phi0 + 2 * np.pi * off / (n_off + 1)),
            frame=fov_frame,
        )
        
        theta_offs.append(off_pos.separation(reco_fov))
        
    return reco_coord.separation(source_coord), np.concatenate(theta_offs)


def read_run_calculate_thetas(run, columns, threshold, source: SkyCoord, n_offs):

    df = read_h5py(run, key = 'events', columns=columns)

    ontime = calc_ontime(df).to(u.hour)

    if type(threshold) == float:
        df_selected = df.query(f'gammaness > {threshold}')
    else:
        df['selected_gh'] = evaluate_binned_cut(
            df.gammaness.to_numpy(), df.gamma_energy_prediction.to_numpy() * u.TeV, threshold, operator.ge
        )
        df_selected = df.query('selected_gh')

    location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
    obstime = Time(df_selected.dragon_time, format='unix')

    altaz = AltAz(obstime=obstime, location=location)

    pointing = SkyCoord(
        alt=u.Quantity(df_selected.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df_selected.az_tel.values, u.rad, copy=False),
        frame=altaz,
    )
    pointing_icrs = pointing.transform_to('icrs')

    prediction_icrs = SkyCoord(
        df_selected.source_ra_prediction.values * u.rad, 
        df_selected.source_dec_prediction.values * u.rad, 
        frame='icrs'
    )

    theta, theta_off = calc_theta_off(
        source_coord=source,
        reco_coord=prediction_icrs,
        pointing_coord=pointing_icrs,
        n_off=n_offs,
    )

    # generate df containing corresponding energies etc for theta_off
    df_selected5 = df_selected
    for i in range(n_offs-1):
        df_selected5 = df_selected5.append(df_selected)

    return df_selected, ontime, theta, df_selected5, theta_off