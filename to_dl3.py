import operator
import click

import calculation

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

from pyirf.cuts import evaluate_binned_cut


columns = [
    'obs_id',
    'event_id',
    'source_ra_prediction',
    'source_dec_prediction',
    'dragon_time', 
    'gammaness',
    'alt_tel',
    'az_tel',
    'pointing_ra',
    'pointing_dec',
    'gamma_energy_prediction'
]
event_map = {
    'source_ra_prediction': 'RA',
    'source_dec_prediction': 'DEC',
    'event_id': 'EVENT_ID',
    'dragon_time': 'TIME',
    'gamma_energy_prediction': 'ENERGY',
    'theta_on': 'THETA_ON'
}
pointing_map = {
    'dragon_time': 'TIME',
    'pointing_ra': 'RA_PNT',
    'pointing_dec': 'DEC_PNT'
}
unit_map = {
    'RA': u.rad,
    'DEC': u.rad,
    'TIME': u.s,
    'ENERGY': u.TeV,
    'RA_PNT': u.rad,
    'DEC_PNT': u.rad,
    'THETA_ON': u.deg
}
DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["HDUCLASS"] = 'GADF'
DEFAULT_HEADER["HDUDOC"] = ('https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html')
DEFAULT_HEADER["HDUVERS"] = '0.2'
DEFAULT_HEADER["CREATOR"] = 'Lukas Beiske'


@click.command()
@click.argument('outdir', type=click.Path(exists=True, dir_okay=True))
@click.argument('data', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument('source', type=str)
@click.argument('cuts_file', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--n_offs', type=int, default=5,
    help='Number of OFF regions (default = 5)'
)
@click.option(
    '--n_jobs', type=int, default=-1,
    help='Number of processors used (default = -1)'
)
def main(outdir, data, source, cuts_file, n_offs, n_jobs):

    gh_cuts = QTable.read(cuts_file, hdu='GH_CUTS')
    theta_cuts = QTable.read(cuts_file, hdu='THETA_CUTS_OPT')
    src = SkyCoord.from_name(source)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as pool:
        results = np.array(
            pool.starmap(
                calculation.read_run_calculate_thetas, # applies gh cut
                [(run, columns, gh_cuts, src, n_offs) for run in data]
            ), dtype=object
        )

    dfs = results[:,0]
    ontimes = results[:,1]
    thetas = results[:,2]
    df5s = results[:,3] # only necessary for theta_off (not currently included)
    theta_offs = results[:,4] # not currently included

    observations = []
    index = []
    for df, ontime, theta, df5, theta_off in zip(dfs, ontimes, thetas, df5s, theta_offs):
        tstart = df.dragon_time.min()
        tstop = df.dragon_time.max()
        df['theta_on'] = theta.deg
        df.dropna(
            subset=['gamma_energy_prediction', 'gammaness', 'source_ra_prediction', 'source_dec_prediction'],
            inplace=True
        )
        # Apply theta cuts here?
        #theta_mask = evaluate_binned_cut(
        #    theta, df.gamma_energy_prediction.to_numpy() * u.TeV, theta_cuts, operator.le
        #)

        df_events = df[event_map.keys()]
        df_events.rename(columns=event_map, inplace=True)
        df_pointings = df[pointing_map.keys()]
        df_pointings.rename(columns=pointing_map, inplace=True)

        events = QTable.from_pandas(df_events, units=unit_map)
        events['RA'] = events['RA'].to(u.deg)
        events['DEC'] = events['DEC'].to(u.deg)

        pointings = QTable.from_pandas(df_pointings, units=unit_map)
        pointings['RA_PNT'] = pointings['RA_PNT'].to(u.deg)
        pointings['DEC_PNT'] = pointings['DEC_PNT'].to(u.deg)

        event_header = DEFAULT_HEADER.copy()
        event_header['HDUCLAS1'] = 'EVENTS'
        event_header['OBS_ID'] = df.obs_id.iloc[0]
        event_header['TSTART'] = tstart
        event_header['TSTOP'] = tstop
        event_header['ONTIME'] = ontime.to_value(u.s)
        event_header['LIVETIME'] = event_header['ONTIME'] # Fix this?
        event_header['DEADC'] = 1.0

        event_header['RA_PNT'] = np.mean(pointings['RA_PNT']).to_value(u.deg)
        event_header['DEC_PNT'] = np.mean(pointings['DEC_PNT']).to_value(u.deg)

        event_header['EQUINOX'] = 2000.0
        event_header['RADECSYS'] = 'ICRS'
        event_header['ORIGIN'] = 'CTA'
        event_header['TELESCOP'] = 'LST1'
        event_header['INSTRUME'] = 'LST1'

        gtis = QTable(
            [[tstart] * u.s, [tstop] * u.s],
            names=('START', 'STOP')
        )
        gti_header = DEFAULT_HEADER.copy()
        gti_header['MJDREFI'] = 40587 # ref time is this correct? 01.01.1970?
        gti_header['MJDREFF'] = .0
        gti_header['TIMEUNIT'] = 's'
        gti_header['TIMESYS'] = 'UTC'  # ?
        gti_header['TIMEREF'] = 'TOPOCENTER' # ?

        pointing_header = gti_header.copy()
        pointing_header['OBSGEO-L'] = -17.89139
        pointing_header['OBSGEO-B'] = 28.76139
        pointing_header['OBSGEO-H'] = 2184.0

        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(events, header=event_header, name="EVENTS"),
            fits.BinTableHDU(pointings, header=pointing_header, name="POINTING"),
            fits.BinTableHDU(gtis, header=gti_header, name="GTI")
        ]
        fits.HDUList(hdus).writeto(f'{outdir}/{df.obs_id.iloc[0]}.fits.gz', overwrite=True)

        observations.append((
            df.obs_id.iloc[0],
            (df_pointings.RA_PNT.mean(axis=0) * u.rad).to_value(u.deg),
            (df_pointings.DEC_PNT.mean(axis=0) * u.rad).to_value(u.deg),
            tstart,
            tstop,
            1.0
        ))
        index.append((
            df.obs_id.iloc[0],
            'events',
            'events',
            '.',
            f'{df.obs_id.iloc[0]}.fits.gz',
            'EVENTS'
        ))
        index.append((
            df.obs_id.iloc[0],
            'gti',
            'gti',
            '.',
            f'{df.obs_id.iloc[0]}.fits.gz',
            'GTI'
        ))
        index.append((
            df.obs_id.iloc[0],
            'aeff',
            'aeff_2d',
            '.',
            cuts_file.split('/')[-1],
            'EFFECTIVE_AREA'
        ))
        index.append((
            df.obs_id.iloc[0],
            'psf',
            'psf_table',
            '.',
            cuts_file.split('/')[-1],
            'PSF'
        ))
        index.append((
            df.obs_id.iloc[0],
            'edisp',
            'edisp_2d',
            '.',
            cuts_file.split('/')[-1],
            'ENERGY_DISPERSION'
        ))
        index.append((
            df.obs_id.iloc[0],
            'bkg',
            'bkg_2d',
            '.',
            cuts_file.split('/')[-1],
            'BACKGROUND'
        ))

    observations_table = QTable(
        rows=observations,
        names=['OBS_ID', 'RA_PNT', 'DEC_PNT', 'TSTART', 'TSTOP', 'DEADC'],
        units=['', 'deg', 'deg', 's', 's', '']
    )
    obs_header = DEFAULT_HEADER.copy()
    obs_header['HDUCLAS1'] = 'INDEX'
    obs_header['HDUCLAS2'] = 'OBS'
    obs_header['MJDREFI'] = 40587 # ref time is this correct? 01.01.1970?
    obs_header['MJDREFF'] = .0 
    obs_header['TIMEUNIT'] = 's'
    obs_header['TIMESYS'] = 'UTC'  # ?
    obs_header['TIMEREF'] = 'TOPOCENTER' # ?
    obs_header['OBSGEO-L'] = -17.89139
    obs_header['OBSGEO-B'] = 28.76139
    obs_header['OBSGEO-H'] = 2184.0

    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(observations_table, header=obs_header, name="OBS_INDEX")
    ]
    fits.HDUList(hdus).writeto(f'{outdir}/obs-index.fits.gz', overwrite=True)

    index_table = QTable(
        rows=index,
        names=['OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'FILE_DIR', 'FILE_NAME', 'HDU_NAME'],
    )
    index_header = DEFAULT_HEADER.copy()
    index_header['HDUCLAS1'] = 'INDEX'
    index_header['HDUCLAS2'] = 'HDU'

    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(index_table, header=index_header, name="HDU_INDEX"),
    ]
    fits.HDUList(hdus).writeto(f'{outdir}/hdu-index.fits.gz', overwrite=True)


if __name__ == '__main__':
    main()