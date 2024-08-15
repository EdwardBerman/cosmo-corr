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
    chi2: float


polydeg = 1
polydim = 2
nparams = np.prod([polydeg+i+1 for i in range(polydim)])/polydim

psf_fits = []
fits_file_name = '../../cweb_psf/new_f277_apr_mosaic_combined_catalog.fits'
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
        chi2_map = np.divide(np.square(a_vignet - a_psfex), f[2].data['ERR_VIGNET_CROPPED'][i])
        chi2_finite = chi2_map[np.isfinite(chi2_map)]
        nan_count, inf_count = count_nan_inf(chi2_map)
        ddof = chi2_finite.size - nparams - (nan_count + inf_count)
        chi2 = np.sum(chi2_finite)/ddof
        ra = f[2].data['alphawin_j2000'][i]
        dec = f[2].data['deltawin_j2000'][i]
        psf_fit = PSF_fit(ra, dec, s_d, g1_d, g2_d, s_p, g1_p, g2_p, chi2)
        psf_fits.append(psf_fit)
    except Exception as e:
        print(f"Error processing object {i}: {e}")
        continue

dtype = [('ra', np.float64), ('dec', np.float64), ('sig_vignet', np.float64), ('g1_vignet', np.float64), ('g2_vignet', np.float64), ('sig_psfex', np.float64), ('g1_psfex', np.float64), ('g2_psfex', np.float64), ('chi2', np.float64)]
psf_fits_array = np.array([tuple(psf.__dict__.values()) for psf in psf_fits], dtype=dtype)
column_names = ['ra', 'dec', 'sig_vignet', 'g1_vignet', 'g2_vignet', 'sig_psfex', 'g1_psfex', 'g2_psfex', 'chi2']
table = Table(psf_fits_array, names=column_names)

table = table[~np.isnan(table['sig_vignet'])]
table = table[~np.isnan(table['g1_vignet'])]
table = table[~np.isnan(table['g2_vignet'])]
table = table[~np.isnan(table['sig_psfex'])]
table = table[~np.isnan(table['g1_psfex'])]
table = table[~np.isnan(table['g2_psfex'])]
threshold = 3*np.std(table['sig_vignet']) + np.mean(table['sig_vignet'])
table = table[table['sig_vignet'] < threshold]
table = table[table['chi2'] < 100]
output_fits_path = f'psf_fits_f277w.fits'
table.write(output_fits_path, overwrite=True)


fig, ax = plt.subplots(1, 1, figsize=(8, 7), tight_layout=True)
plt.rcParams.update({'xtick.labelsize': 18})
plt.rcParams.update({'ytick.labelsize': 18})
bins = np.logspace(np.log10(0.001), np.log10(100), 20)
ax.hist(table['chi2'], bins=bins, label='PSFex', alpha=0.5, histtype='bar') 
ax.set_xscale('log')
ax.set_xlim(0.001, 100)
ax.set_xlabel(r'$\chi^2$', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
ax.legend(fontsize=18)
plt.savefig('chi2_hist.png')

