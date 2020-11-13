import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u

from fact.analysis.statistics import li_ma_significance
from ctapipe.coordinates import CameraFrame

from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator


erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


def calc_dist(x, y):
    dist = x**2 + y**2
    return dist


def calc_theta2(dist, focal_length):
    theta2 = np.rad2deg(
        np.sqrt(dist) / focal_length
    )**2
    return theta2


def total_t(df):
    delta = np.diff(df.dragon_time.sort_values())
    delta = delta[np.abs(delta) < 10]
    return len(df) * delta.mean()


def theta2(theta2_on, theta2_off, scaling, cut, threshold, source, total_time=None, ax=None, window=[0,1]):

    ax = ax or plt.gca()

    ax.hist(theta2_on, bins=100, range=window, histtype='step', color='r', label='ON')
    ax.hist(theta2_off, bins=100, range=window, histtype='stepfilled', color='tab:blue', alpha=0.5, label='OFF', weights=np.full_like(theta2_off, scaling))

    n_off = np.count_nonzero(theta2_off < cut)
    n_on = np.count_nonzero(theta2_on < cut)
    li_ma = li_ma_significance(n_on, n_off, scaling)
    n_exc_mean = n_on - scaling * n_off
    n_exc_std = np.sqrt(n_on + scaling**2 * n_off)

    text_pos = 0.9 * theta2_on[theta2_on < 0.01].size 
    text = (
        rf'Source: {source}, $t_\mathrm{{obs}} = {total_time:.2f} \mathrm{{h}}$' + '\n'
        + rf'$N_\mathrm{{on}} = {n_on},\, N_\mathrm{{off}} = {n_off},\, \alpha = {scaling:.2f}$' + '\n' 
        + rf'$N_\mathrm{{exc}} = {n_exc_mean:.0f} \pm {n_exc_std:.0f},\, S_\mathrm{{Li&Ma}} = {li_ma:.2f}$'
    )
    ax.text(0.3 * window[1], text_pos, text)
    ax.axvline(x=cut, color='k', alpha=0.6, lw=1.5, ls=':')
    ax.annotate(
        rf'$\theta_\mathrm{{max}}^2 = {cut} \mathrm{{deg}}^2$' + '\n' + rf'$(\, t_\gamma = {threshold} \,)$', 
        (cut + window[1]/100, 0.8 * text_pos)
    )

    ax.set_xlabel(r'$\theta^2 \,\, / \,\, \mathrm{deg}^2$')
    ax.set_xlim(window)
    ax.legend()
    ax.figure.tight_layout()
    return ax


def theta_astropy(df, source):
    obstime = Time(df.dragon_time, format='unix')
    location = EarthLocation.of_site('Roque de los Muchachos')
    altaz = AltAz(obstime=obstime, location=location)
    
    pointing = SkyCoord(
        alt=u.Quantity(df.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df.az_tel.values, u.rad, copy=False),
        frame=altaz,
    )
    
    camera_frame = CameraFrame(telescope_pointing=pointing, location=location, obstime=obstime, focal_length=28 * u.m)

    prediction_cam = SkyCoord(
        x=u.Quantity(df.source_x_prediction.values, u.m, copy=False),
        y=u.Quantity(df.source_y_prediction.values, u.m, copy=False),
        frame=camera_frame,
    )
    
    prediction_icrs = prediction_cam.transform_to('icrs')

    theta = prediction_icrs.separation(source)
    
    return theta.dms
