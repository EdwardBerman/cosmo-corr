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
        fontsize=24
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 3.9})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 24})
    plt.rcParams.update({'xtick.major.width': 3.9})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 3.})
    plt.rcParams.update({'xtick.minor.size': 18})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 3.9})
    plt.rcParams.update({'ytick.major.size': 24})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 3.})
    plt.rcParams.update({'ytick.minor.size':18})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})

    return

set_rc_params(fontsize=28)

f115w = fits.open('revised_apr_f115w_shopt_xy_info.fits')

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
plt.rcParams.update({'legend.fontsize': 28})
plt.rcParams.update({'font.family': 'monospace'})

star_title = \
            'F115W median $\sigma^{*}_{HSM} = %.2f$ mas; $e^{*}_{HSM} = %.5f$'\
                        % (median_sigma_vignet*1000, median_e_vignet)

min_ellip = 0.01
mask = e_vignet <= min_ellip
fig, ax = plt.subplots(figsize=(16, 16))
round = ax.scatter(f115w_hsm['x'][mask], f115w_hsm['y'][mask], s=9,
            facecolor='black', edgecolors='black')
mask = e_vignet > min_ellip
q = ax.quiver(
    f115w_hsm['x'][mask], f115w_hsm['y'][mask],
    e1_vignet[mask], e2_vignet[mask], sig_vignet[mask],
    angles=np.rad2deg(theta_vignet[mask]), **q_dict
)
key = ax.quiverkey(q, **qkey_dict)
lx, rx = ax.get_xlim()
ly, ry = ax.get_ylim()
ax.set_xlim(lx-2000, rx+2000)
ax.set_ylim(ly-500, ry+500)
ax.set_title(star_title, fontsize=32)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("bottom", size="5%", pad="7%")
cbar = fig.colorbar(q, cax=cax, orientation="horizontal")


#print("Vignet:")
#print(f'g1 = {np.mean(g1_vignet)} +/- {np.std(g1_vignet)}, g2 = {np.mean(g2_vignet)} +/- {np.std(g2_vignet)}, sigma = {np.mean(sig_vignet)} +/- {np.std(sig_vignet)}')
#print(f'Ra = {np.mean(f115w_hsm["ra"])} +/- {np.std(f115w_hsm["ra"])}')
#print(f'Dec = {np.mean(f115w_hsm["dec"])} +/- {np.std(f115w_hsm["dec"])}')
plt.savefig('/home/eddieberman/research/mcclearygroup/AstroCorr/assets/vignet_f115w_hsm_quiver.png', dpi=300)


