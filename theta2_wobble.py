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

    columns = [
        'source_x_prediction', 
        'source_y_prediction', 
        'dragon_time', 
        'gammaness',
        'focal_length',
        'alt_tel',
        'az_tel'
    ]

    df = pd.DataFrame()
    for i, run in enumerate(data):
        df = pd.concat( [
                df,
                read_h5py(run, key = 'events', columns=columns)
            ],
            ignore_index=True
        )

    df_selected = df.query(f'gammaness > {threshold}')
    scaling = 1 / n_offs
    total_time = plotting.total_t(df_selected) / 3600

    # define camera frame
    obstime = Time(df_selected.dragon_time, format='unix')
    location = EarthLocation.of_site('Roque de los Muchachos')
    altaz = AltAz(obstime=obstime, location=location)
    
    pointing = SkyCoord(
        alt=u.Quantity(df_selected.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df_selected.az_tel.values, u.rad, copy=False),
        frame=altaz,
    )
    
    camera_frame = CameraFrame(telescope_pointing=pointing, location=location, obstime=obstime, focal_length=28 * u.m)

    src = SkyCoord.from_name(source)
    src_cam = src.transform_to(camera_frame)

    r = np.sqrt(src_cam.x.to_value(u.m)**2 + src_cam.y.to_value(u.m)**2)
    phi = np.arctan2(src_cam.y.to_value(u.m), src_cam.x.to_value(u.m))


    # distance in camera frame
    dist_on = plotting.calc_dist(
        df_selected.source_x_prediction - src_cam.x.to_value(u.m), 
        df_selected.source_y_prediction - src_cam.y.to_value(u.m)
    )
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


    # distance in icrs
    theta_on = plotting.theta_astropy(df_selected, src) # returns astropy.coordinates.Angle.dms 
    theta_on_series = pd.Series(
        theta_on.d + theta_on.m / 60 + theta_on.s / 3600
    )

    # first off position
    x_off = r * np.cos(phi + 2 * np.pi / (n_offs + 1)) 
    y_off = r * np.sin(phi +  2 * np.pi / (n_offs + 1))

    offPos_cam = SkyCoord(x_off * u.m, y_off * u.m, frame=camera_frame)
    offPos_icrs = offPos_cam.transform_to('icrs')

    theta_off =  plotting.theta_astropy(df_selected, offPos_icrs)
    theta_off_series = pd.Series( 
        theta_off.d + theta_off.m / 60 + theta_off.s / 3600
    )
    # further off postions
    for i in range(2, n_offs + 1):
        x_off = r * np.cos(phi + i * 2 * np.pi / (n_offs + 1)) 
        y_off = r * np.sin(phi + i * 2 * np.pi / (n_offs + 1))

        offPos_cam = SkyCoord(x_off * u.m, y_off * u.m, frame=camera_frame)
        offPos_icrs = offPos_cam.transform_to('icrs')

        theta_off_temp = plotting.theta_astropy(df_selected, offPos_icrs)
        theta_off_series_temp = pd.Series(
            theta_off_temp.d + theta_off_temp.m / 60 + theta_off_temp.s / 3600
        )
        theta_off_series = theta_off_series.append(theta_off_series_temp)


    # plots
    figures = []

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta2_on, theta2_off, scaling, theta2_cut, 
        threshold, source, total_time=total_time,
        ax=ax
    )

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta_on_series**2, theta_off_series**2, scaling, theta2_cut, 
        threshold, source, total_time=total_time,
        ax=ax
    )
    ax.set_title('astropy')

    # saving
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()