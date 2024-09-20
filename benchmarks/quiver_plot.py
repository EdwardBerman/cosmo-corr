from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

f115w = fits.open('f115w_validation_split_augmented_resized_xy_info.fits')
f150w = fits.open('revised_apr_f115w_shopt_xy_info.fits')

f115w_hsm = f150w[1].data

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
div_norm = colors.CenteredNorm(vcenter=0, halfrange=0.05)

qkey_scale = 0.05
scale=1
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

min_ellip = 0.01
mask = e_vignet <= min_ellip
fig, ax = plt.subplots(1, 3, figsize=(16, 16))
round = ax[0].scatter(f115w_hsm['x'][mask], f115w_hsm['y'][mask], s=9,
            facecolor='black', edgecolors='black')
mask = e_vignet > min_ellip
q = ax[0].quiver(
    f115w_hsm['x'][mask], f115w_hsm['y'][mask],
    e1_vignet[mask], e2_vignet[mask], sig_vignet[mask],
    angles=np.rad2deg(theta_vignet[mask]), **q_dict
)
key = ax[0].quiverkey(q, **qkey_dict)
lx, rx = ax[0].get_xlim()
ly, ry = ax[0].get_ylim()
ax[0].set_xlim(lx-2000, rx+2000)
ax[0].set_ylim(ly-500, ry+500)
ax[0].set_title(star_title)
ax_divider = make_axes_locatable(ax[0])
cax = ax_divider.append_axes("bottom", size="5%", pad="7%")
cbar = fig.colorbar(q, cax=cax, orientation="horizontal")

mask = e_psfex <= min_ellip
ax[1].scatter(f115w_hsm['x'][mask], f115w_hsm['y'][mask], s=9,
            facecolor='black', edgecolors='black')
mask = e_psfex > min_ellip

q = ax[1].quiver(
    f115w_hsm['x'][mask], f115w_hsm['y'][mask],
    e1_psfex[mask], e2_psfex[mask], sig_psfex[mask],
    angles=np.rad2deg(theta_psfex[mask]), **q_dict
)

key = ax[1].quiverkey(q, **qkey_dict)
lx, rx = ax[1].get_xlim()
ly, ry = ax[1].get_ylim()
ax[1].set_xlim(lx-2000, rx+2000)
ax[1].set_ylim(ly-500, ry+500)
ax[1].set_title(psf_title)
ax_divider = make_axes_locatable(ax[1])
cax = ax_divider.append_axes("bottom", size="5%", pad="7%")
cbar = fig.colorbar(q, cax=cax, orientation="horizontal")

mask = e_residual <= min_ellip
ax[2].scatter(f115w_hsm['ra'][mask], f115w_hsm['dec'][mask], s=9,
            facecolor='black', edgecolors='black')
mask = e_residual > min_ellip
q = ax[2].quiver(
    f115w_hsm['x'][mask], f115w_hsm['y'][mask],
    e1_residual[mask], e2_residual[mask], sig_residual[mask],
    angles=np.rad2deg(theta_residual[mask]), **q_dict
)
key = ax[2].quiverkey(q, **qkey_dict)
lx, rx = ax[2].get_xlim()
ly, ry = ax[2].get_ylim()
ax[2].set_xlim(lx-2000, rx+2000)
ax[2].set_ylim(ly-500, ry+500)
ax[2].set_title(resid_title)
ax_divider = make_axes_locatable(ax[2])
cax = ax_divider.append_axes("bottom", size="5%", pad="7%")
cbar = fig.colorbar(q, cax=cax, orientation="horizontal")

print("Vignet:")
print(f'g1 = {np.mean(g1_vignet)} +/- {np.std(g1_vignet)}, g2 = {np.mean(g2_vignet)} +/- {np.std(g2_vignet)}, sigma = {np.mean(sig_vignet)} +/- {np.std(sig_vignet)}')
print("PSFEx:")
print(f'g1 = {np.mean(g1_psfex)} +/- {np.std(g1_psfex)}, g2 = {np.mean(g2_psfex)} +/- {np.std(g2_psfex)}, sigma = {np.mean(sig_psfex)} +/- {np.std(sig_psfex)}')

plt.savefig('/home/eddieberman/research/mcclearygroup/AstroCorr/assets/f115w_hsm_quiver.png', dpi=300)


