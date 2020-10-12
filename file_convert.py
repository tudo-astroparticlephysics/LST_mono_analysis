import pandas as pd
from fact.io import to_h5py
import numpy as np
import click

@click.command()
@click.argument('infile', type=click.Path(exists=True, dir_okay=False))
@click.argument('outfile', type=click.Path(exists=False, dir_okay=False))
@click.argument('tel_name')
def main(infile, outfile, tel_name):
    
    parameters = pd.read_hdf(infile, key = f'dl1/event/telescope/parameters/{tel_name}')
    focal_length = pd.read_hdf(infile, key = 'instrument/telescope/optics').drop_duplicates().set_index('name').loc['LST', 'equivalent_focal_length']

    # renaming for simulations
    if 'mc_az' in parameters.columns:
        parameters['az_tel'] = parameters.mc_az_tel
        parameters['alt_tel'] = parameters.mc_alt_tel

    parameters['focal_length'] = focal_length
    to_h5py(parameters, outfile, key='events', mode = 'w')

if __name__ == '__main__':
    main()