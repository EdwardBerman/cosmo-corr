import galsim
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os

from dataclasses import dataclass

@dataclass
class PSF_fit:
    sig_vignet: float
    g1_vignet: float
    g2_vignet: float
    sig_psfex: float
    g1_psfex: float
    g2_psfex: float

# Loop over all visits
for visit_number in range(1, 39):  # from 001 to 038
    visit_str = f'visit_{visit_number:03d}'
    fits_path = f'../../mccleary/real_images/gal_cutout_20mas_working/f150w/{visit_str}_nircam_f150w_COSMOS-Web_20mas_v0_6_i2d_valid_starcat.fits'

    if not os.path.exists(fits_path):
        print(f"File {fits_path} does not exist, skipping.")
        continue

    f = fits.open(fits_path)
    psf_fits = []

    for i in range(len(f[2].data)):
        try:
            a_vignet = f[2].data['VIGNET'][i]
            img_vignet = galsim.Image(a_vignet, wcs=galsim.PixelScale(0.02), xmin=0, ymin=0)
            result_vignet = img_vignet.FindAdaptiveMom(guess_sig=1.0)
            a_psfex = f[2].data['PSFEX_VIGNET'][i]
            img_psfex = galsim.Image(a_psfex, wcs=galsim.PixelScale(0.02), xmin=0, ymin=0)
            result_psfex = img_psfex.FindAdaptiveMom(guess_sig=1)
            s_d, g1_d, g2_d = result_vignet.moments_sigma, result_vignet.observed_shape.g1, result_vignet.observed_shape.g2
            s_p, g1_p, g2_p = result_psfex.moments_sigma, result_psfex.observed_shape.g1, result_psfex.observed_shape.g2
            psf_fits.append(PSF_fit(s_d, g1_d, g2_d, s_p, g1_p, g2_p))
        except Exception as e:
            print(f"Error processing object {i} in {visit_str}: {e}")
            continue

    # Create a table of PSF fits for the current visit
    dtype = [('sig_vignet', np.float64), ('g1_vignet', np.float64), ('g2_vignet', np.float64),
             ('sig_psfex', np.float64), ('g1_psfex', np.float64), ('g2_psfex', np.float64)]
    psf_fits_array = np.array([tuple(psf.__dict__.values()) for psf in psf_fits], dtype=dtype)
    column_names = ['sig_vignet', 'g1_vignet', 'g2_vignet', 'sig_psfex', 'g1_psfex', 'g2_psfex']
    table = Table(psf_fits_array, names=column_names)

    # Data Cleaning - remove rows with NaN values
    table = table[~np.isnan(table['sig_vignet'])]
    table = table[~np.isnan(table['g1_vignet'])]
    table = table[~np.isnan(table['g2_vignet'])]
    table = table[~np.isnan(table['sig_psfex'])]
    table = table[~np.isnan(table['g1_psfex'])]
    table = table[~np.isnan(table['g2_psfex'])]

    # Apply threshold for sig_vignet
    threshold = 3*np.std(table['sig_vignet']) + np.mean(table['sig_vignet'])
    table = table[table['sig_vignet'] < threshold]

    # Save the table to a FITS file for the current visit
    output_fits_path = f'psf_fits_{visit_str}.fits'
    table.write(output_fits_path, overwrite=True)
    print(f"Saved {output_fits_path}")
