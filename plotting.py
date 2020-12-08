import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

from fact.analysis.statistics import li_ma_significance


erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


def calc_dist(x, y):
    return np.sqrt(x**2 + y**2)


def calc_theta2(dist, focal_length):
    return np.rad2deg(dist / focal_length)**2


def ontime(df):
    delta = np.diff(df.dragon_time.sort_values())
    delta = delta[np.abs(delta) < 10]
    return len(df) * delta.mean() * u.s


def theta2(theta2_on, theta2_off, scaling, cut, threshold, source, ontime=None, ax=None, window=[0,1]):

    ax = ax or plt.gca()

    ax.hist(theta2_on, bins=100, range=window, histtype='step', color='r', label='ON')
    ax.hist(theta2_off, bins=100, range=window, histtype='stepfilled', color='tab:blue', alpha=0.5, label='OFF', weights=np.full_like(theta2_off, scaling))

    n_off = np.count_nonzero(theta2_off < cut)
    n_on = np.count_nonzero(theta2_on < cut)
    li_ma = li_ma_significance(n_on, n_off, scaling)
    n_exc_mean = n_on - scaling * n_off
    n_exc_std = np.sqrt(n_on + scaling**2 * n_off)

    txt = rf'''Source: {source}, $t_\mathrm{{obs}} = {ontime.to_value(u.hour):.2f} \mathrm{{h}}$
    $\theta_\mathrm{{max}}^2 = {cut} \mathrm{{deg}}^2,\, t_\gamma = {threshold}$
    $N_\mathrm{{on}} = {n_on},\, N_\mathrm{{off}} = {n_off},\, \alpha = {scaling:.2f}$
    $N_\mathrm{{exc}} = {n_exc_mean:.0f} \pm {n_exc_std:.0f},\, S_\mathrm{{Li&Ma}} = {li_ma:.2f}$
    '''
    ax.text(0.5, 0.95, txt, transform=ax.transAxes, va='top', ha='center')
    ax.axvline(cut, color='k', alpha=0.6, lw=1, ls='--')

    ax.set_xlabel(r'$\theta^2 \,\, / \,\, \mathrm{deg}^2$')
    ax.set_xlim(window)
    ax.legend()
    ax.figure.tight_layout()
    return ax


def calc_theta_off(source_coord: SkyCoord, reco_coord: SkyCoord, pointing_coord: SkyCoord, theta_save=None, n_off=5):
    fov_frame = SkyOffsetFrame(origin=pointing_coord)
    source_fov = source_coord.transform_to(fov_frame)
    reco_fov = reco_coord.transform_to(fov_frame)
    
    r = source_coord.separation(pointing_coord)
    phi0 = np.arctan2(source_fov.lat, source_fov.lon).to_value(u.rad)
    
    theta_offs = []
    for off in range(1, n_off + 1):
        
        off_pos = SkyCoord(
            lon=r * np.sin(phi0 + 2 * np.pi * off / (n_off + 1)),
            lat=r * np.cos(phi0 + 2 * np.pi * off / (n_off + 1)),
            frame=fov_frame,
        )
        
        theta_offs.append(off_pos.separation(reco_fov))
        if theta_save is not None:
            theta_save[f'astropy_off_{off}'] = off_pos.separation(reco_fov)

    if theta_save is not None:
        theta_save['astropy_on'] = reco_coord.separation(source_coord)
        
    return reco_coord.separation(source_coord), np.concatenate(theta_offs)