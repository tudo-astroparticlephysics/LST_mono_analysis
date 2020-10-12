import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u

from fact.analysis.statistics import li_ma_significance
from ctapipe.coordinates import CameraFrame, TelescopeFrame

def calc_dist(x, y):
    dist = x**2 + y**2
    return dist


def total_t(df):
    delta = np.diff(df.dragon_time.sort_values())
    delta = delta[np.abs(delta) < 10]
    return len(df) * delta.mean()


def theta2(df, cut, threshold, n_offs, source, ax=None, window=[0,1]):

    ax = ax or plt.gca()

    src = SkyCoord.from_name(source)
    altaz = AltAz(
        location = EarthLocation.of_site('Roque de los Muchachos'),
        obstime = Time(df.dragon_time, format='unix')
    )
    telescope_pointing = SkyCoord(
        alt = u.Quantity(df.alt_tel.to_numpy(), u.rad, copy=False),
        az = u.Quantity(df.az_tel.to_numpy(), u.rad, copy=False),
        frame = altaz
    )
    camera_frame = CameraFrame(
        focal_length = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False),
        telescope_pointing = telescope_pointing,
        location = EarthLocation.of_site('Roque de los Muchachos'),
        obstime = Time(df.dragon_time, format='unix')
    )
    src_cf = src.transform_to(camera_frame)


    dist_on = calc_dist(
        df.source_x_prediction - src_cf.x.to_value(u.m), 
        df.source_y_prediction - src_cf.y.to_value(u.m)
    )

    dist_off = pd.Series()
    r = np.sqrt(src_cf.x.to_value(u.m)**2 + src_cf.y.to_value(u.m)**2)
    phi = np.arctan2(src_cf.y.to_value(u.m), src_cf.x.to_value(u.m))
    for i in range(1, n_offs + 1):
        x_off = r * np.cos(phi + i * 2 * np.pi / (n_offs + 1)) 
        y_off = r * np.sin(phi + i * 2 * np.pi / (n_offs + 1))
        dist_off = dist_off.append(
            calc_dist(
                df.source_x_prediction - x_off,
                df.source_y_prediction - y_off
            )
        )


    theta2_on = np.rad2deg(np.sqrt(dist_on) / df.focal_length)**2
    theta2_off = np.rad2deg(np.sqrt(dist_off) / df.focal_length)**2
    scaling = 1 / n_offs
    
    ax.hist(theta2_off, bins=100, range=window, histtype='stepfilled', color='tab:blue', alpha=0.5, label='OFF', weights=np.full_like(theta2_off, scaling))
    ax.hist(theta2_on, bins=100, range=window, histtype='step', color='r', label='ON')

    n_off = np.count_nonzero(theta2_off < cut)
    n_on = np.count_nonzero(theta2_on < cut)
    li_ma = li_ma_significance(n_on, n_off, scaling)
    n_exc_mean = n_on - scaling * n_off
    n_exc_std = np.sqrt(n_on + scaling**2 * n_off)

    total_time = total_t(df) / 3600
    text_pos = 0.9 * theta2_on[theta2_on < 0.01].size 
    text = (
        rf'Source: {source}, $t_\mathrm{{obs}} = {total_time:.2f}$' + '\n'
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


def angular_res(df, true_energy_column, ax=None, label=r'$68^{\mathrm{th}}$ Percentile'):

    df = df.copy()
    edges = 10**np.arange(
        np.log10(df[true_energy_column].min()),
        np.log10(df[true_energy_column].max()),
        0.2     #cta convention: 5 bins per energy decade
    )
    df['bin_idx'] = np.digitize(df[true_energy_column], edges)

    binned = pd.DataFrame({
        'e_center': 0.5 * (edges[1:] + edges[:-1]),
        'e_low': edges[:-1],
        'e_high': edges[1:],
        'e_width': np.diff(edges),
    }, index=pd.Series(np.arange(1, len(edges)), name='bin_idx'))

    df['diff'] = np.rad2deg(
        np.sqrt((df.source_x_prediction - df.src_x)**2 + (df.source_y_prediction - df.src_y)**2)
        / df.focal_length
    )

    def f(group):
        group = group.sort_values('diff')
        group = group.dropna(axis='index', subset=['diff'])
        group68 = group.quantile(q=0.68)
        return group68['diff']

    grouped = df.groupby('bin_idx')
    counts = grouped.size()
    binned['ang_res'] = grouped.apply(f)
    binned['counts'] = counts
    binned = binned.query('counts > 50') # at least 50 events per bin
    
    ax = ax or plt.gca()

    ax.errorbar(
        binned.e_center, binned.ang_res,
        xerr=binned.e_width / 2,
        ls='',
        label=label
    )
    ax.set_ylabel(r'$\theta_{68\%} \,\, / \,\, \mathrm{deg}$')
    ax.set_xlabel(
        r'$E_{\mathrm{true}} \,\,/\,\, \mathrm{TeV}$'
    )
    ax.set_xscale('log')
    ax.legend()

    return ax