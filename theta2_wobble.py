import click
import operator

import plotting
import calculation

from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fact.io import read_h5py

import astropy.units as u
from astropy import table
from astropy.io import fits

from astropy.coordinates import SkyCoord

from fact.analysis.statistics import li_ma_significance

from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.spectral import CRAB_HEGRA

import matplotlib
if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


columns = [
    'obs_id',
    'event_id',
    'source_az_prediction',
    'source_alt_prediction',
    'source_ra_prediction',
    'source_dec_prediction',
    'dragon_time', 
    'gammaness',
    'focal_length',
    'alt_tel',
    'az_tel',
    'gamma_energy_prediction'
]

COLUMN_MAP = {
    'obs_id': 'obs_id',
    'event_id': 'event_id',
    'gamma_energy_prediction': 'reco_energy',
    'source_alt_prediction': 'reco_alt',
    'source_az_prediction': 'reco_az',
    'alt_tel': 'pointing_alt',
    'az_tel': 'pointing_az',
    'gammaness': 'gh_score',
}

UNIT_MAP = {
    'reco_energy': u.TeV,
    'reco_alt': u.rad,
    'reco_az': u.rad,
    'pointing_alt': u.rad,
    'pointing_az': u.rad
}

MAX_BG_RADIUS = 1 * u.deg


@click.command()
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.argument('data', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument('source', type=str)
@click.argument('cuts_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('theta2_cut', type=float)
@click.argument('threshold', type=float)
@click.option(
    '--n_offs', type=int, default=5,
    help='Number of OFF regions (default = 5)'
)
@click.option(
    '--n_jobs', type=int, default=-1,
    help='Number of processors used (default = -1)'
)
def main(output, data, source, cuts_file, theta2_cut, threshold, n_offs, n_jobs):
    outdir = output.split('/')[0]

    src = SkyCoord.from_name(source)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as pool:
        results = np.array(
            pool.starmap(
                calculation.read_run_calculate_thetas, 
                [(run, columns, threshold, src, n_offs) for run in data]
            ), dtype=object
        )

    df_selected = pd.concat(results[:,0], ignore_index=True)
    ontime = np.sum(results[:,1])
    theta = np.concatenate(results[:,2])
    df_selected5 = pd.concat(results[:,3], ignore_index=True)
    theta_off = np.concatenate(results[:,4])

    # use pyirf cuts
    gh_cuts = table.QTable.read(cuts_file, hdu='GH_CUTS')
    theta_cuts_opt = table.QTable.read(cuts_file, hdu='THETA_CUTS_OPT')
    
    with Pool(n_jobs) as pool:
        results = np.array(
            pool.starmap(
                calculation.read_run_calculate_thetas, 
                [(run, columns, gh_cuts, src, n_offs) for run in data]
            ), dtype=object
        )

    df_pyirf = pd.concat(results[:,0], ignore_index=True)
    theta_pyirf = np.concatenate(results[:,2])
    df_pyirf5 = pd.concat(results[:,3], ignore_index=True)
    theta_off_pyirf = np.concatenate(results[:,4])

    n_on = np.count_nonzero(
        evaluate_binned_cut(
            theta_pyirf, df_pyirf.gamma_energy_prediction.to_numpy() * u.TeV, theta_cuts_opt, operator.le
        )
    )
    n_off = np.count_nonzero(
        evaluate_binned_cut(
            theta_off_pyirf, df_pyirf5.gamma_energy_prediction.to_numpy() * u.TeV, theta_cuts_opt, operator.le
        )
    )
    li_ma = li_ma_significance(n_on, n_off, 1/n_offs)
    n_exc_mean = n_on - (1/n_offs) * n_off
    n_exc_std = np.sqrt(n_on + (1/n_offs)**2 * n_off)


    ##############################################################################################################
    # plots
    ##############################################################################################################
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
        theta_pyirf.deg**2, 
        theta_off_pyirf.deg**2, 
        1/n_offs, theta2_cut, r'\mathrm{energy-dependent}', 
        source, ontime=ontime,
        ax=ax
    )
    ax.set_title(r'Energy-dependent $t_\gamma$ optimised using pyirf')

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
    ax.set_title(r'Energy-dependent $t_\gamma$ and $\theta_\mathrm{max}^2$ optimised using pyirf')


    ##############################################################################################################
    # sensitivity
    ##############################################################################################################
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.8 * u.TeV, 10 ** 2.41 * u.TeV, bins_per_decade=5
        )
    )
    # gh_cuts and theta_cuts_opt in line 119f

    gammas = plotting.to_astropy_table(
        df_pyirf, # df_pyirf has pyirf gh cuts already applied
        column_map=COLUMN_MAP,
        unit_map=UNIT_MAP,
        theta=theta_pyirf,
        t_obs=ontime
    )
    background = plotting.to_astropy_table(
        df_pyirf5,
        column_map=COLUMN_MAP,
        unit_map=UNIT_MAP,
        theta=theta_off_pyirf,
        t_obs=ontime
    )
    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    background["selected_theta"] = evaluate_binned_cut(
        background["theta"], background["reco_energy"], theta_cuts_opt, operator.le
    )

    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected_theta"]], bins=sensitivity_bins
    )
    background_hist = estimate_background(
        background[background["selected_theta"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=1/n_offs,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(
        signal_hist, background_hist, alpha=1/n_offs
    )

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = CRAB_HEGRA
    sensitivity["flux_sensitivity"] = (
        sensitivity["relative_sensitivity"] * spectrum(sensitivity['reco_energy_center'])
    )

    ##############################################################################################################
    # sensitivity using unoptimised cuts
    ##############################################################################################################
    gammas_unop = plotting.to_astropy_table(
        df_selected, # df_selected has gammaness > threshold already applied
        column_map=COLUMN_MAP,
        unit_map=UNIT_MAP,
        theta=theta,
        t_obs=ontime
    )
    background_unop = plotting.to_astropy_table(
        df_selected5,
        column_map=COLUMN_MAP,
        unit_map=UNIT_MAP,
        theta=theta_off,
        t_obs=ontime
    )

    gammas_unop["selected_theta"] = gammas_unop["theta"].to_value(u.deg) <= np.sqrt(0.03)
    background_unop["selected_theta"] = background_unop["theta"].to_value(u.deg) <= np.sqrt(0.03)

    theta_cut_unop = theta_cuts_opt
    theta_cut_unop['cut'] = np.sqrt(0.03) * u.deg

    # calculate sensitivity
    signal_hist_unop = create_histogram_table(
        gammas_unop[gammas_unop["selected_theta"]], bins=sensitivity_bins
    )
    background_hist_unop = estimate_background(
        background_unop[background_unop["selected_theta"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cut_unop,
        alpha=1/n_offs,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity_unop = calculate_sensitivity(
        signal_hist_unop, background_hist_unop, alpha=1/n_offs
    )

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    sensitivity_unop["flux_sensitivity"] = (
        sensitivity_unop["relative_sensitivity"] * spectrum(sensitivity_unop['reco_energy_center'])
    )

    # write fits file and create plot
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_unop, name="SENSITIVITY_UNOP")
    ]
    fits.HDUList(hdus).writeto(f'{outdir}/sensitivity_{source}.fits.gz', overwrite=True)

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    for s, label in zip(
        [sensitivity, sensitivity_unop], 
        ['pyirf optimised cuts', rf'$\theta^2 < {theta2_cut}$ and gh_score$> {threshold}$']
    ): plotting.plot_sensitivity(s, label=label, ax=ax)

    # plot Magic sensitivity for reference
    magic = table.QTable.read('notebooks/magic_sensitivity_2014.ecsv')
    plotting.plot_sensitivity(magic, label='MAGIC 2014', ax=ax, magic=True)

    ax.set_title(f'Minimal Flux Satisfying Requirements for 50 hours \n(based on {ontime.to_value(u.hour):.2f}h of {source} observations)')


    # save plots
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()