import click

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from fact.io import read_h5py
from fact.io import to_h5py
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from ctapipe.coordinates import CameraFrame

import plotting
from fact.analysis.statistics import li_ma_significance

from astropy import table
from pyirf.cuts import evaluate_binned_cut
import operator

import matplotlib
if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator


erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))


columns = [
        'source_x_prediction', 
        'source_y_prediction', 
        'source_ra_prediction',
        'source_dec_prediction',
        'dragon_time', 
        'gammaness',
        'focal_length',
        'alt_tel',
        'az_tel',
        'gamma_energy_prediction'
    ]


@click.command()
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.argument('data', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument('source', type=str)
@click.argument('cuts_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('theta2_cut', type=float)
@click.argument('threshold', type=float)
@click.option(
    '--n_offs', '-n', type=int, default=5,
    help='Number of OFF regions (default = 5)'
)
def main(output, data, source, cuts_file, theta2_cut, threshold, n_offs):

    df = pd.DataFrame()
    for i, run in enumerate(data):
        df = pd.concat( [
                df,
                read_h5py(run, key = 'events', columns=columns)
            ],
            ignore_index=True
        )

    ontime = plotting.ontime(df).to(u.hour)

    df_selected = df.query(f'gammaness > {threshold}')
    df_selected = df_selected.query('gamma_energy_prediction > 0.15') # e_reco > 150GeV cut

    # theta/ distance to source/ off position in icrs 
    src = SkyCoord.from_name(source)

    obstime = Time(df_selected.dragon_time, format='unix')
    location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)

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

    df_thetas = pd.DataFrame() # save thetas for comparison

    theta, theta_off = plotting.calc_theta_off(
        source_coord=src,
        reco_coord=prediction_icrs,
        pointing_coord=pointing_icrs,
        theta_save=df_thetas,
        n_off=n_offs,
    )
    

    # theta/ distance to source/ off position in camera frame
    camera_frame = CameraFrame(telescope_pointing=pointing, location=location, obstime=obstime, focal_length=28 * u.m)

    src_cam = src.transform_to(camera_frame)

    dist_on = plotting.calc_dist(
        df_selected.source_x_prediction - src_cam.x.to_value(u.m), 
        df_selected.source_y_prediction - src_cam.y.to_value(u.m)
    )
    theta2_on = plotting.calc_theta2(dist_on, df_selected.focal_length)

    df_thetas['camera_frame_on'] = u.Quantity(np.sqrt(theta2_on), u.deg)

    r = np.sqrt(src_cam.x.to_value(u.m)**2 + src_cam.y.to_value(u.m)**2)
    phi = np.arctan2(src_cam.y.to_value(u.m), src_cam.x.to_value(u.m))

    theta2_off = pd.Series(dtype = 'float64')
    for i in range(1, n_offs + 1):
        x_off = r * np.cos(phi + i * 2 * np.pi / (n_offs + 1)) 
        y_off = r * np.sin(phi + i * 2 * np.pi / (n_offs + 1))
        dist_off = plotting.calc_dist(
            df_selected.source_x_prediction - x_off,
            df_selected.source_y_prediction - y_off
        )
        theta2_off = theta2_off.append(
            plotting.calc_theta2(dist_off, df_selected.focal_length)
        )
        df_thetas[f'camera_frame_off_{i}'] = u.Quantity(
            np.sqrt(plotting.calc_theta2(dist_off, df_selected.focal_length)),
            u.deg
        )

    to_h5py(df_thetas, 'build/theta_comparison.h5', key='thetas_compare', mode = 'w') # save thetas for comparison 


    ##############################################################################################################
    # use pyirf cuts
    gh_cuts = table.QTable.read(cuts_file, hdu='GH_CUTS')
    theta_cuts = table.QTable.read(cuts_file, hdu='THETA_CUTS_OPT')
    
    df['selected_gh'] = evaluate_binned_cut(
        df.gammaness.to_numpy(), df.gamma_energy_prediction.to_numpy() * u.TeV, gh_cuts, operator.ge
    )
    df_pyirf = df.query('selected_gh')
    df_pyirf = df_pyirf.query('gamma_energy_prediction > 0.15') # e_reco > 150GeV cut

    obstime = Time(df_pyirf.dragon_time, format='unix')

    altaz = AltAz(obstime=obstime, location=location)

    pointing = SkyCoord(
        alt=u.Quantity(df_pyirf.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df_pyirf.az_tel.values, u.rad, copy=False),
        frame=altaz,
    )
    pointing_icrs = pointing.transform_to('icrs')

    prediction_icrs = SkyCoord(
        df_pyirf.source_ra_prediction.values * u.rad, 
        df_pyirf.source_dec_prediction.values * u.rad, 
        frame='icrs'
    )

    theta_pyirf, theta_off_pyirf = plotting.calc_theta_off(
        source_coord=src,
        reco_coord=prediction_icrs,
        pointing_coord=pointing_icrs,
        n_off=n_offs,
    )

    n_on = np.count_nonzero(
        evaluate_binned_cut(
            theta_pyirf, df_pyirf.gamma_energy_prediction.to_numpy() * u.TeV, theta_cuts, operator.le
        )
    )
    # generate array containing corresponding energies for theta_off_pyirf
    reco_energy5 = df_pyirf.gamma_energy_prediction
    for i in range(4):
        reco_energy5 = reco_energy5.append(df_pyirf.gamma_energy_prediction)
    
    n_off = np.count_nonzero(
        evaluate_binned_cut(
            theta_off_pyirf, reco_energy5.to_numpy() * u.TeV, theta_cuts, operator.le
        )
    )
    li_ma = li_ma_significance(n_on, n_off, 1/n_offs)
    n_exc_mean = n_on - (1/n_offs) * n_off
    n_exc_std = np.sqrt(n_on + (1/n_offs)**2 * n_off)


    ##############################################################################################################
    # plots
    figures = []

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta.deg**2, 
        theta_off.deg**2, 
        1/n_offs, theta2_cut, threshold, 
        source, ontime=ontime,
        ax=ax
    )
    ax.set_title('Theta calculated in ICRS using astropy')

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta2_on, 
        theta2_off, 
        1/n_offs, theta2_cut, threshold, 
        source, ontime=ontime,
        ax=ax
    )
    ax.set_title('Theta calculated in camera frame')

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta_pyirf.deg**2, 
        theta_off_pyirf.deg**2, 
        1/n_offs, theta2_cut, threshold, 
        source, ontime=ontime,
        ax=ax
    )
    ax.set_title('Gh cut optimised using pyirf')

    # plot using pyirf theta cuts
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    ax.hist(
        theta_pyirf.deg**2, 
        bins=100, range=[0,1], histtype='step', color='r', label='ON'
    )
    ax.hist(
        theta_off_pyirf.deg**2, 
        bins=100, range=[0,1], histtype='stepfilled', color='tab:blue', alpha=0.5, label='OFF', 
        weights=np.full_like(theta_off_pyirf.deg**2, 1/n_offs)
    )

    txt = rf'''Source: {source}, $t_\mathrm{{obs}} = {ontime.to_value(u.hour):.2f} \mathrm{{h}}$
    $N_\mathrm{{on}} = {n_on},\, N_\mathrm{{off}} = {n_off},\, \alpha = {1/n_offs:.2f}$
    $N_\mathrm{{exc}} = {n_exc_mean:.0f} \pm {n_exc_std:.0f},\, S_\mathrm{{Li&Ma}} = {li_ma:.2f}$
    '''
    ax.text(0.5, 0.95, txt, transform=ax.transAxes, va='top', ha='center')

    ax.set_xlabel(r'$\theta^2 \,\, / \,\, \mathrm{deg}^2$')
    ax.set_xlim(0,1)
    ax.legend()
    ax.figure.tight_layout()
    ax.set_title('Gh and theta cuts optimised using pyirf')


    # saving
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()