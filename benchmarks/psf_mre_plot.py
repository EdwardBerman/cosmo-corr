import glob
from astropy.io import fits
from astropy.table import Table, vstack
from catalogaugmenter import catalog, psf
from catalogaugmenter import webb_psf, epsfex, shopt, piff_psf 
#from catalogplotter import ResidPlots
import os
import re 
from datetime import datetime, timedelta
import catplot as ctp 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
#Make augmented catalogs with columns for each psf fitter than use the plotter with these new catalogs
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None) # or a large number like 1000

def sem_with_nans(data):        
    """        
    Calculate the standard error of the mean (SEM) for a dataset that might contain NaNs.        
            
    Args:        
    - data (list or np.array): The dataset        
            
    Returns:        
    - float: The standard error of the mean        
    """        
    # Remove NaNs                                    
    filtered_data = np.array(data)[~np.isnan(data)]        
            
    # Calculate standard deviation        
    sd = np.std(filtered_data, ddof=1)  # Using ddof=1 for sample standard deviation        
            
    # Calculate SEM        
    sem = sd / np.sqrt(len(filtered_data))        
            
    return sem       

f115w_cat_name = 'f115w_validation_split_augmented.fits'
f115w_catalog = catalog(f115w_cat_name)

f150w_cat_name = 'f150w_validation_split_augmented.fits'
f150w_catalog = catalog(f150w_cat_name)

def extract_3_numbers(filename):
    pattern = r'\d{3}'
    matches = re.findall(pattern, filename)
    return matches


mre_name = "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/f115w_mosaic_mock_mre_psfex.png"

mean_relative_error_plot_psfex = ctp.mean_relative_error_plot(catalog(f115w_cat_name), epsfex(''))
mean_relative_error_plot_psfex.preprocessing()
mean_relative_error_plot_psfex.set_residuals()
sum_residuals_mre_psfex = mean_relative_error_plot_psfex.return_residuals_sum()
std_mr_psfex = mean_relative_error_plot_psfex.return_sem()
mean_relative_error_plot_psfex.set_titles([f'Median Star F115W', f'Median PSFex PSF', f'MRE'])
mean_relative_error_plot_psfex.save_figure(outname=mre_name)


mre_name = "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/f150w_mosaic_mock_mre_psfex.png"

mean_relative_error_plot_psfex = ctp.mean_relative_error_plot(catalog(f150w_cat_name), epsfex(''))
mean_relative_error_plot_psfex.preprocessing()
mean_relative_error_plot_psfex.set_residuals()
sum_residuals_mre_psfex = mean_relative_error_plot_psfex.return_residuals_sum()
std_mr_psfex = mean_relative_error_plot_psfex.return_sem()
mean_relative_error_plot_psfex.set_titles([f'Median Star F150W', f'Median PSFex PSF', f'MRE'])
mean_relative_error_plot_psfex.save_figure(outname=mre_name)

