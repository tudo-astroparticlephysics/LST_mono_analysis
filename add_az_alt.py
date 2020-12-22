from fact.io import read_h5py
from aict_tools.io import append_column_to_hdf5
import h5py
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import click

from ctapipe.coordinates import CameraFrame
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator


erfa_astrom.set(ErfaAstromInterpolator(10 * u.min))
location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
columns_obs = [
    'dragon_time',
    'az_tel',
    'alt_tel',
    'source_x_prediction',
    'source_y_prediction'
]
columns_sim = [
    'az_tel',
    'alt_tel',
    'source_x_prediction',
    'source_y_prediction'
]


@click.command()
@click.argument('infile', type=click.Path(exists=True, dir_okay=False))
def main(infile):

    with h5py.File(infile, mode='r') as f:
        is_simulation = 'corsika_runs' in f

    if is_simulation:
        df = read_h5py(infile, key='events', columns=columns_sim)
        obstime = None
    else:
        df = read_h5py(infile, key='events', columns=columns_obs)
        obstime = Time(df.dragon_time, format='unix')
    
    altaz = AltAz(obstime=obstime, location=location)

    pointing = SkyCoord(
        alt=u.Quantity(df.alt_tel.values, u.rad, copy=False),
        az=u.Quantity(df.az_tel.values, u.rad, copy=False),
        frame=altaz,
    )
    
    camera_frame = CameraFrame(telescope_pointing=pointing, location=location, obstime=obstime, focal_length=28 * u.m)

    prediction_cam = SkyCoord(
        x=u.Quantity(df.source_x_prediction.values, u.m, copy=False),
        y=u.Quantity(df.source_y_prediction.values, u.m, copy=False),
        frame=camera_frame,
    )

    prediction_altaz = prediction_cam.transform_to(altaz)

    append_column_to_hdf5(infile, prediction_altaz.alt.rad, 'events', 'source_alt_prediction')
    append_column_to_hdf5(infile, prediction_altaz.az.rad, 'events', 'source_az_prediction')

    if not is_simulation:
        prediction_icrs = prediction_altaz.transform_to('icrs')
        pointing_icrs = pointing.transform_to('icrs')
        
        append_column_to_hdf5(infile, prediction_icrs.ra.rad, 'events', 'source_ra_prediction')
        append_column_to_hdf5(infile, prediction_icrs.dec.rad, 'events', 'source_dec_prediction')
        append_column_to_hdf5(infile, pointing_icrs.ra.rad, 'events', 'pointing_ra')
        append_column_to_hdf5(infile, pointing_icrs.dec.rad, 'events', 'pointing_dec')



if __name__ == '__main__':
    main()