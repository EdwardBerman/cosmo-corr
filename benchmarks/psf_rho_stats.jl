include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
using UnicodePlots
using Base.Threads
using Statistics
using Distributions

println(nthreads())

file_name = "Aardvark.fit"

fits = pyimport("astropy.io.fits")
f = fits.open(file_name)
print(f[2].data)

ra = f[2].data["RA"]
dec = f[2].data["DEC"]
σ_D = f[2].data["sig_vignet"]
g1_D = f[2].data["g1_vignet"]
g2_D = f[2].data["g2_vignet"]
σ_psf = f[2].data["sig_psfex"]
g1_psf = f[2].data["g1_psfex"]
g2_psf = f[2].data["g2_psfex"]
χ2 = f[2].data["chi2"]

ra = convert(Vector{Float64}, ra)
dec = convert(Vector{Float64}, dec)
σ_D = convert(Vector{Float64}, σ_D)
g1_D = convert(Vector{Float64}, g1_D)
g2_D = convert(Vector{Float64}, g2_D)
σ_psf = convert(Vector{Float64}, σ_psf)
g1_psf = convert(Vector{Float64}, g1_psf)
g2_psf = convert(Vector{Float64}, g2_psf)
χ2 = convert(Vector{Float64}, χ2)

# NOTE: Shear is not a vector, but a complex number. However, we represent is as a vector of two components for quick computation and readibility.
e_D = [[g1_D[i], g2_D[i]] for i in 1:length(g1_D)]
e_psf = [[g1_psf[i], g2_psf[i]] for i in 1:length(g1_psf)]
δ_e = [[g1_D[i] - g1_psf[i], g2_D[i] - g2_psf[i]] for i in 1:length(g1_D)]

T_D .= 2.0 .* σ_D.^2
T_psf .= 2.0 .* σ_psf.^2
δ_T .= T_D .- T_psf
δ_TT .= δ_T ./ T_psf


ρ1 = corr(ra, dec, δ_e, δ_e, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ2 = corr(ra, dec, e_psf, δ_e, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ3 = corr(ra, dec, e_psf.*δ_TT, e_psf.*δ_TT, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ4 = corr(ra, dec, δ_e, e_psf.*δ_TT, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ5 = corr(ra, dec, e_psf, e_psf.*δ_TT, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)

ρ6 = corr(ra, dec, χ2, δg1, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ7 = corr(ra, dec, χ2, δg2, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
ρ8 = corr(ra, dec, χ2, δ_TT, 200.0*0.03/3600, 10, 5000.0*0.03/3600; correlator=naivecorr, verbose=true)
