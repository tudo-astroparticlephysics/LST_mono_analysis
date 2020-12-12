import logging
import operator

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits
from fact.io import read_h5py

from pyirf.simulations import SimulatedEventsInfo
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut

from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)

from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)

import click


log = logging.getLogger(__name__)


COLUMN_MAP = {
    "obs_id": "obs_id",
    "event_id": "event_id",
    "mc_energy": "true_energy",
    "gamma_energy_prediction": "reco_energy",
    "mc_alt": "true_alt",
    "mc_az": "true_az",
    "source_alt_prediction": "reco_alt",
    "source_az_prediction": "reco_az",
    "alt_tel": "pointing_alt",
    "az_tel": "pointing_az",
    "gammaness": "gh_score",
}

UNIT_MAP = {
    "true_energy": u.TeV,
    "reco_energy": u.TeV,
    "true_alt": u.rad,
    "true_az": u.rad,
    "reco_alt": u.rad,
    "reco_az": u.rad,
    "pointing_alt": u.rad,
    "pointing_az": u.rad
}

T_OBS = 50 * u.hour

ALPHA = 0.2

# Radius to use for calculating bg rate
MAX_BG_RADIUS = 1 * u.deg
MAX_GH_CUT_EFFICIENCY = 1
GH_CUT_EFFICIENCY_STEP = 0.01

# gh cut used for first calculation of the binned theta cuts
INITIAL_GH_CUT_EFFICENCY = 0.4


def read_file(infile):
    log.debug(f"Reading {infile}")
    events = read_h5py(infile, key='events', columns=list(COLUMN_MAP.keys()))
    sim_runs = read_h5py(infile, key='corsika_runs')

    events.rename(columns=COLUMN_MAP, inplace=True)
    
    n_showers = np.sum(sim_runs.num_showers * sim_runs.shower_reuse)
    log.debug(f"Number of events from corsika_runs: {n_showers}")

    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        energy_min=u.Quantity(sim_runs["energy_range_min"][0], u.TeV),
        energy_max=u.Quantity(sim_runs["energy_range_max"][0], u.TeV),
        max_impact=u.Quantity(sim_runs["max_scatter_range"][0], u.m),
        spectral_index=sim_runs["spectral_index"][0],
        viewcone=u.Quantity(sim_runs["max_viewcone_radius"][0] - sim_runs["min_viewcone_radius"][0], u.deg),
    )
    return table.QTable.from_pandas(events, units=UNIT_MAP), sim_info


@click.command()
@click.argument('gammafile', type=click.Path(exists=True, dir_okay=False))
@click.argument('protonfile', type=click.Path(exists=True, dir_okay=False))
@click.argument('electronfile', type=click.Path(exists=True, dir_okay=False))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False))
def main(gammafile, protonfile, electronfile, outputfile):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    particles = {
        "gamma": {
            "file": gammafile,
            "target_spectrum": CRAB_HEGRA,
        },
        "proton": {
            "file": protonfile,
            "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        },
        "electron": {
            "file": electronfile,
            "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        },
    }

    for particle_type, p in particles.items():
        log.info(f"Simulated {particle_type.title()} Events:")
        p["events"], p["simulation_info"] =  read_file(p["file"])
        p["events"]["particle_type"] = particle_type

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        for prefix in ('true', 'reco'):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)

        log.info(p["simulation_info"])
        log.info("")

    gammas = particles["gamma"]["events"]
    background = table.vstack(
        [particles["proton"]["events"], particles["electron"]["events"]]
    )

    # calculate theta / distance between reco and assuemd source position
    gammas["theta"] = calculate_theta(
        gammas,
        assumed_source_az=gammas["true_az"],
        assumed_source_alt=gammas["true_alt"],
    )

    INITIAL_GH_CUT = np.quantile(gammas['gh_score'], (1 - INITIAL_GH_CUT_EFFICENCY))
    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    theta_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.8 * u.TeV, 10 ** 2.41 * u.TeV, bins_per_decade=25
        )
    )
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.8 * u.TeV, 10 ** 2.41 * u.TeV, bins_per_decade=5
        )
    )

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=theta_bins,
        min_value=0.05 * u.deg,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        percentile=68,
    )

    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        GH_CUT_EFFICIENCY_STEP,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP
    )
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        gammas,
        background,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        op=operator.ge,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    log.info('Recalculating theta cut for optimized GH Cuts')
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_cuts_opt = calculate_percentile_cut(
        gammas[gammas['selected_gh']]["theta"],
        gammas[gammas['selected_gh']]["reco_energy"],
        theta_bins,
        percentile=68,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        min_value=0.05 * u.deg,
    )

    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    background_hist = estimate_background(
        background[background["selected_gh"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(
        signal_hist, background_hist, alpha=ALPHA
    )

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles['gamma']['target_spectrum']
    for s in (sensitivity_step_2, sensitivity):
        s["flux_sensitivity"] = (
            s["relative_sensitivity"] * spectrum(s['reco_energy_center'])
        )

    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]


    # calculate sensitivity using unoptimised cuts (temporary, just for comparison)
    gammas["theta_unop"] = gammas["theta"].to_value(u.deg) <= np.sqrt(0.03)
    gammas["gh_unop"] = gammas["gh_score"] > 0.85

    theta_cut_unop = table.QTable()
    theta_cut_unop['low'] = theta_cuts_opt['low']
    theta_cut_unop['high'] = theta_cuts_opt['high']
    theta_cut_unop['center'] = theta_cuts_opt['center']
    theta_cut_unop['cut'] = np.sqrt(0.03) * u.deg

    signal_hist_unop = create_histogram_table(
        gammas[gammas["theta_unop"] & gammas["gh_unop"]], bins=sensitivity_bins
    )
    background_hist_unop = estimate_background(
        background[background["gh_score"] > 0.85],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cut_unop,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity_unop = calculate_sensitivity(
        signal_hist_unop, background_hist_unop, alpha=ALPHA
    )
    sensitivity_unop["flux_sensitivity"] = (
        sensitivity_unop["relative_sensitivity"] * spectrum(sensitivity_unop['reco_energy_center'])
    )
    hdus.append(
        fits.BinTableHDU(sensitivity_unop, name="SENSITIVITY_UNOP")
    )


    log.info('Calculating IRFs')
    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.8 * u.TeV, 10 ** 2.41 * u.TeV, bins_per_decade=10
        )
    )
    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(
            10 ** -1.8 * u.TeV, 10 ** 2.41 * u.TeV, bins_per_decade=5
        )
    )
    fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], true_energy_bins,
    )
    ang_res = angular_resolution(
        gammas[gammas["selected_gh"]], true_energy_bins,
    )
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )
    background_rate = background_2d(
        background[background['selected_gh']],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(
        create_background_2d_hdu(
            background_rate,
            reco_energy_bins,
            fov_offset_bins=np.arange(0, 11) * u.deg,
        )
    )
    hdus.append(
        create_psf_table_hdu(
            psf, true_energy_bins, source_offset_bins, fov_offset_bins,
        )
    )
    hdus.append(
        create_rad_max_hdu(
            theta_cuts_opt["cut"][:, np.newaxis], theta_bins, fov_offset_bins
        )
    )
    hdus.append(
        fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION")
    )
    hdus.append(
        fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION")
    )

    log.info('Writing outputfile')
    fits.HDUList(hdus).writeto(outputfile, overwrite=True)


if __name__ == '__main__':
    main()