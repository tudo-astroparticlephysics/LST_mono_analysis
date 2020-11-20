import numpy as np
import matplotlib.pyplot as plt
from fact.io import read_h5py
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import plotting
import click

import matplotlib
if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

from ctapipe.coordinates import CameraFrame
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
        'az_tel'
    ]


@click.command()
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.argument('source', type=str)
@click.argument('theta2_cut', type=float)
@click.argument('threshold', type=float)
@click.option(
    '--data', '-d', multiple=True, type=click.Path(exists=True, dir_okay=True),
    help='DL2 data to be analysed'
)
@click.option(
    '--n_offs', '-n', type=int, default=5,
    help='Number of OFF regions (default = 5)'
)
def main(output, source, theta2_cut, threshold, data, n_offs):
    
    if not data:
        exit('No data given!')

    df = pd.DataFrame()
    for i, run in enumerate(data):
        df = pd.concat( [
                df,
                read_h5py(run, key = 'events', columns=columns)
            ],
            ignore_index=True
        )

    df_selected = df.query(f'gammaness > {threshold}')

    ontime = plotting.ontime(df_selected).to(u.hour)

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

    theta, theta_off = plotting.calc_theta_off(
        source_coord=src,
        reco_coord=prediction_icrs,
        pointing_coord=pointing_icrs,
        n_off=n_offs,
    )

    # theta/ distance to source/ off position in camera frame
    camera_frame = CameraFrame(telescope_pointing=pointing, location=location, obstime=obstime, focal_length=28 * u.m)

    src_cam = src.transform_to(camera_frame)

    dist_on = plotting.calc_dist(
        df_selected.source_x_prediction - src_cam.x.to_value(u.m), 
        df_selected.source_y_prediction - src_cam.y.to_value(u.m)
    )

    r = np.sqrt(src_cam.x.to_value(u.m)**2 + src_cam.y.to_value(u.m)**2)
    phi = np.arctan2(src_cam.y.to_value(u.m), src_cam.x.to_value(u.m))

    dist_off = pd.Series(dtype = 'float64')
    for i in range(1, n_offs + 1):
        x_off = r * np.cos(phi + i * 2 * np.pi / (n_offs + 1)) 
        y_off = r * np.sin(phi + i * 2 * np.pi / (n_offs + 1))
        dist_off = dist_off.append(
            plotting.calc_dist(
                df_selected.source_x_prediction - x_off,
                df_selected.source_y_prediction - y_off
            )
        )

    theta2_on = plotting.calc_theta2(dist_on, df_selected.focal_length)
    theta2_off = plotting.calc_theta2(dist_off, df_selected.focal_length)

    # plots
    figures = []

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta.deg**2, theta_off.deg**2, 1/n_offs, theta2_cut, 
        threshold, source, ontime=ontime,
        ax=ax
    )
    ax.set_title('Theta calculated in ICRS using astropy')

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta2_on, theta2_off, 1/n_offs, theta2_cut, 
        threshold, source, ontime=ontime,
        ax=ax
    )
    ax.set_title('Theta calculated in camera frame')

    # saving
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()