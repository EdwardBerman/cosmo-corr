import galsim
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PSF_fit:
    ra: float
    dec: float
    sig_vignet: float
    g1_vignet: float
    g2_vignet: float
    sig_psfex: float
    g1_psfex: float
    g2_psfex: float
    #chi2: float


psf_fits = []

file_names = ['f115w_validation_split_augmented_resized.fits', 'f150w_validation_split_augmented_resized.fits']

fits_file_name = file_names[1]

f = fits.open(fits_file_name)

def count_nan_inf(arr):
            np_arr = np.array(arr)
            nan_count = np.count_nonzero(np.isnan(np_arr))
            inf_count = np.count_nonzero(np.isinf(np_arr))
            return nan_count, inf_count

for i in range(len(f[2].data)):
    try:
        a_vignet = f[2].data['VIGNET_CROPPED'][i]
        img_vignet = galsim.Image(a_vignet, wcs=galsim.PixelScale(0.03), xmin=0, ymin=0)
        result_vignet = img_vignet.FindAdaptiveMom(guess_sig=1.0)
        a_psfex = f[2].data['VIGNET_PSFEX_CROPPED'][i]
        img_psfex = galsim.Image(a_psfex, wcs=galsim.PixelScale(0.03), xmin=0, ymin=0)
        result_psfex = img_psfex.FindAdaptiveMom(guess_sig=1)
        s_d, g1_d, g2_d = result_vignet.moments_sigma, result_vignet.observed_shape.g1, result_vignet.observed_shape.g2
        s_p, g1_p, g2_p = result_psfex.moments_sigma, result_psfex.observed_shape.g1, result_psfex.observed_shape.g2
        ra = f[2].data['alphawin_j2000'][i]
        dec = f[2].data['deltawin_j2000'][i]
        psf_fit = PSF_fit(ra, dec, s_d, g1_d, g2_d, s_p, g1_p, g2_p)
        psf_fits.append(psf_fit)
    except Exception as e:
        print(f"Error processing object {i}: {e}")
        continue

dtype = [('ra', np.float64), ('dec', np.float64), ('sig_vignet', np.float64), ('g1_vignet', np.float64), ('g2_vignet', np.float64), ('sig_psfex', np.float64), ('g1_psfex', np.float64), ('g2_psfex', np.float64)]
psf_fits_array = np.array([tuple(psf.__dict__.values()) for psf in psf_fits], dtype=dtype)
column_names = ['ra', 'dec', 'sig_vignet', 'g1_vignet', 'g2_vignet', 'sig_psfex', 'g1_psfex', 'g2_psfex']
table = Table(psf_fits_array, names=column_names)

table = table[~np.isnan(table['sig_vignet'])]
table = table[~np.isnan(table['g1_vignet'])]
table = table[~np.isnan(table['g2_vignet'])]
table = table[~np.isnan(table['sig_psfex'])]
table = table[~np.isnan(table['g1_psfex'])]
table = table[~np.isnan(table['g2_psfex'])]
threshold = 3*np.std(table['sig_vignet']) + np.mean(table['sig_vignet'])
table = table[table['sig_vignet'] < threshold]
output_fits_path = fits_file_name.replace('.fits', '_hsm_info.fits')
table.write(output_fits_path, overwrite=True)


