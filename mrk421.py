import numpy as np
import matplotlib.pyplot as plt
from fact.io import read_h5py
import pandas as pd
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
@click.option(
    '--data', '-d', multiple=True, type=click.Path(exists=True, dir_okay=True),
    help='DL2 data to be analysed'
    )
def main(output, data):
    
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

    figures = []
    theta2_cut = 0.04
    gammaness_threshold = 0.6

    df_selected = df.query(f'gammaness > {gammaness_threshold}')

    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plotting.theta2(df_selected, theta2_cut, gammaness_threshold, n_offs=5, source='Mrk 421', ax=ax)

    #saving
    with PdfPages(output) as pdf:
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == '__main__':
    main()