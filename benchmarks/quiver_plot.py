from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})

    return

set_rc_params(fontsize=14)

f115w = fits.open('f115w_validation_split_augmented_resized_hsm_info.fits')
f150w = fits.open('f150w_validation_split_augmented_resized_hsm_info.fits')

f115w_hsm = f115w[1].data

sig_vignet = f115w_hsm['sig_vignet']
g1_vignet = f115w_hsm['g1_vignet']
g2_vignet = f115w_hsm['g2_vignet']

sig_psfex = f115w_hsm['sig_psfex']
g1_psfex = f115w_hsm['g1_psfex']
g2_psfex = f115w_hsm['g2_psfex']

sig_residual = sig_vignet - sig_psfex
g1_residual = g1_vignet - g1_psfex
g2_residual = g2_vignet - g2_psfex

g_residual = np.sqrt(g1_residual**2 + g2_residual**2)
theta_residual = np.arctan2(g2_residual, g1_residual)

e1_residual = g_residual * np.cos(theta_residual)
e2_residual = g_residual * np.sin(theta_residual)
e_residual = np.sqrt(e1_residual**2 + e2_residual**2)

median_residual_g = np.median(g_residual)
median_residual_e = np.median(e_residual)
median_residual_sigma = np.median(sig_residual)
median_sigma_vignet = np.median(sig_vignet)
median_sigma_psfex = np.median(sig_psfex)

theta_vignet = np.arctan2(g2_vignet, g1_vignet)
e1_vignet = np.sqrt(g1_vignet**2 + g2_vignet**2) * np.cos(theta_vignet)
e2_vignet = np.sqrt(g1_vignet**2 + g2_vignet**2) * np.sin(theta_vignet)
e_vignet = np.sqrt(e1_vignet**2 + e2_vignet**2)

theta_psfex = np.arctan2(g2_psfex, g1_psfex)
e1_psfex = np.sqrt(g1_psfex**2 + g2_psfex**2) * np.cos(theta_psfex)
e2_psfex = np.sqrt(g1_psfex**2 + g2_psfex**2) * np.sin(theta_psfex)
e_psfex = np.sqrt(e1_psfex**2 + e2_psfex**2)

median_e_vignet = np.median(e_vignet)
median_e_psfex = np.median(e_psfex)

mean_residual_sigma = np.mean(sig_residual)
std_residual_sigma = np.std(sig_residual)

scale_units = 'width' # For quiver plots
norm = colors.CenteredNorm(vcenter=median_sigma_vignet, halfrange=0.06)
div_norm = colors.CenteredNorm(vcenter=vc2, halfrange=0.05)

qkey_scale = 0.05
qkey_label = r'$e_{HSM} = {%.2f}$' % qkey_scale
fontprops = {'size':14, 'weight':'bold'}

q_dict = dict(
        cmap='cool', #cividis
        width=90,
        units='xy',
        pivot='mid',
        headaxislength=0,
        headwidth=0,
        headlength=0,
        norm=norm,
        scale=scale,
        scale_units=scale_units
        )

qkey_dict = dict(
    X=0.2,
    Y=0.02,
    U=qkey_scale,
    labelpos='N',
    label=qkey_label,
    fontproperties=fontprops
    )

plt.rcParams.update({'xtick.direction': 'out'})
plt.rcParams.update({'ytick.direction': 'out'})
plt.rcParams.update({'legend.fontsize': 14})

star_title = \
            'median $\sigma^{*}_{HSM} = %.2f$ mas; $e^{*}_{HSM} = %.5f$'\
                        % (median_sigma_vignet*1000, median_e_vignet)
psf_title = \
    'median $\sigma^{PSF}_{HSM} = %.2f$ mas; $e^{PSF}_{HSM} = %.5f$'\
                % (median_sigma_psfex*1000, median_e_psfex)
resid_title = \
    'median $\sigma^{resid}_{HSM} = %.2f$ mas; $e^{resid}_{HSM} = %.5f$'\
                % (median_residual_sigma*1000, median_residual_e)
'''
        for i, dc in enumerate(dicts):
            # Set up masks for really circular objects
            min_ellip = 0.01

            # Define the minimum ellipse size and create a mask of circles
            mask = dc.e <= min_ellip
            round = axs[i].scatter(self.x[mask], self.y[mask], s=9,
                        facecolor='black', edgecolors='black')

            mask = dc.e > min_ellip
            q = axs[i].quiver(
                self.x[mask], self.y[mask],
                dc.e1[mask], dc.e2[mask], dc.sigma[mask],
                angles=np.rad2deg(dc.theta[mask]), **quiver_dict
            )
            # adjust x, y limits
            lx, rx = axs[i].get_xlim()
            axs[i].set_xlim(lx-2000, rx+2000)
            ly, ry = axs[i].get_ylim()
            axs[i].set_ylim(ly-500, ry+500)
            key = axs[i].quiverkey(q, **qkey_dict)
            ax_divider = make_axes_locatable(axs[i])
            cax = ax_divider.append_axes("bottom", size="5%", pad="7%")
            cbar = fig.colorbar(q, cax=cax, orientation="horizontal")
            axs[i].set_title(titles[i])
'''
