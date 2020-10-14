import numpy as np
import matplotlib.pyplot as plt
from fact.io import read_h5py
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import plotting
import click

import matplotlib
if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages

columns = [
    'source_x_prediction', 
    'source_y_prediction', 
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

    src = SkyCoord.from_name(source)
    src_cf = plotting.to_camera_frame(df_selected, src)

    dist_on = plotting.calc_dist(
        df_selected.source_x_prediction - src_cf.x.to_value(u.m), 
        df_selected.source_y_prediction - src_cf.y.to_value(u.m)
    )

    dist_off = pd.Series(dtype = 'float64')
    r = np.sqrt(src_cf.x.to_value(u.m)**2 + src_cf.y.to_value(u.m)**2)
    phi = np.arctan2(src_cf.y.to_value(u.m), src_cf.x.to_value(u.m))
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
    scaling = 1 / n_offs

    total_time = plotting.total_t(df_selected) / 3600
    

    figures = []
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(
        theta2_on, theta2_off, scaling, theta2_cut, 
        threshold, source, total_time=total_time,
        ax=ax
    )

    #saving
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()