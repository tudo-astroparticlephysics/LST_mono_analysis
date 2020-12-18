import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import astropy.units as u
from astropy import table

from pyirf.utils import calculate_source_fov_offset

from fact.analysis.statistics import li_ma_significance


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
    return ax


def to_astropy_table(df, column_map, unit_map, theta, t_obs):
    df_renamed = df.rename(columns=column_map)
    tab = table.QTable.from_pandas(df_renamed, units=unit_map)

    tab['reco_source_fov_offset'] = calculate_source_fov_offset(tab, prefix='reco')
    tab['weight'] = 50 / t_obs.value
    tab['theta'] = theta
    return tab


def plot_sensitivity(sensitivity, label=None, ax=None, magic=False):

    ax = ax or plt.gca()

    unit = u.Unit('erg cm-2 s-1')

    if magic:
        e = .5 * (sensitivity['e_max'].to(u.TeV) + sensitivity['e_min'].to(u.TeV))
        w = (sensitivity['e_max'].to(u.TeV) - sensitivity['e_min'].to(u.TeV))
        s = (e**2 * sensitivity['sensitivity_lima_5off'])
    else:
        e = sensitivity['reco_energy_center']
        w = (sensitivity['reco_energy_high'] - sensitivity['reco_energy_low'])
        s = (e**2 * sensitivity['flux_sensitivity'])

    ax.errorbar(
        e.to_value(u.TeV),
        s.to_value(unit),
        xerr=w.to_value(u.TeV) / 2,
        ls='',
        label=label
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed energy / TeV")
    ax.set_ylabel(rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})")
    
    ax.grid(which='both')
    ax.legend()
    return ax