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

ra = convert(Vector{Float64}, ra)
dec = convert(Vector{Float64}, dec)
σ_D = convert(Vector{Float64}, σ_D)
g1_D = convert(Vector{Float64}, g1_D)
g2_D = convert(Vector{Float64}, g2_D)
σ_psf = convert(Vector{Float64}, σ_psf)
g1_psf = convert(Vector{Float64}, g1_psf)
g2_psf = convert(Vector{Float64}, g2_psf)

δ_e .= g1_D .+ im(g2_D) .- (g1_psf .+ im(g2_psf))
δ_e_star = conj.(δ_e)

e = g1_D .+ im(g2_D)
e_star = conj.(e)

T_D .= 2.0 .* σ_D.^2
T_psf .= 2.0 .* σ_psf.^2
δ_T .= T_D .- T_psf
δ_TT .= δ_T ./ T_psf

ρ1 = corr(ra, dec, δ_e_star, δ_e, 200.0, 10, 5000.0; correlator=naivecorr, verbose=true)
ρ2 = corr(ra, dec, e_star, δ_e_star, 200.0, 10, 5000.0; correlator=naivecorr, verbose=true)
ρ3 = corr(ra, dec, e_star.*δ_TT, e.*δ_TT, 200.0, 10, 5000.0; correlator=naivecorr, verbose=true)
ρ4 = corr(ra, dec, δ_e_star, e.*δ_TT, 200.0, 10, 5000.0; correlator=naivecorr, verbose=true)
ρ5 = corr(ra, dec, e_star, e.*δ_TT, 200.0, 10, 5000.0; correlator=naivecorr, verbose=true)
