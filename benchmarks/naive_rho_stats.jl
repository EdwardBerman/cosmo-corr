include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
using UnicodePlots
using Base.Threads
using Statistics
using Distributions

println(nthreads())
struct fuzzy_shear
    shear::Vector{Float64}
end


file_name = "revised_apr_f115w_shopt_xy_info.fits"

fits = pyimport("astropy.io.fits")
f = fits.open(file_name)

ra = f[1].data["ra"]
dec = f[1].data["dec"]
σ_D = f[1].data["sig_vignet"]
g1_D = f[1].data["g1_vignet"]
g2_D = f[1].data["g2_vignet"]
σ_psf = f[1].data["sig_psfex"]
g1_psf = f[1].data["g1_psfex"]
g2_psf = f[1].data["g2_psfex"]

ra = convert(Vector{Float64}, ra)
dec = convert(Vector{Float64}, dec)
σ_D = convert(Vector{Float64}, σ_D)
g1_D = convert(Vector{Float64}, g1_D)
g2_D = convert(Vector{Float64}, g2_D)
σ_psf = convert(Vector{Float64}, σ_psf)
g1_psf = convert(Vector{Float64}, g1_psf)
g2_psf = convert(Vector{Float64}, g2_psf)

# NOTE: Shear is not a vector, but a complex number. However, we represent is as a vector of two components for quick computation and readibility.
e_D = [[g1_D[i], g2_D[i]] for i in 1:length(g1_D)]

e_psf = [[g1_psf[i], g2_psf[i]] for i in 1:length(g1_psf)]
e_psf_conj = [[g1_psf[i], -g2_psf[i]] for i in 1:length(g1_psf)]

δ_e = [[g1_D[i] - g1_psf[i], g2_D[i] - g2_psf[i]] for i in 1:length(g1_D)]
δ_e_conj = [[g1_D[i] - g1_psf[i], -g2_D[i] + g2_psf[i]] for i in 1:length(g1_D)]

T_D = 2.0 .* σ_D.^2
T_psf = 2.0 .* σ_psf.^2
δ_T = T_D .- T_psf
δ_TT = δ_T ./ T_psf

ra_dec = hcat(ra, dec)
n_clusters = 50
nrows, ncols = size(data)
initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec', n_clusters, rand(length(ra_dec)), 0.5)
initial_weights = rand(ncols, n_clusters)

println("Computing ρ1")
fuzzy_shear_one = [fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ1 = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)

println("Computing ρ2")
fuzzy_shear_one = [fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ2 = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)

println("Computing ρ3")
fuzzy_shear_one = [fuzzy_shear(e_psf_conj[i] * δ_TT[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ3 = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)

println("Computing ρ4")
fuzzy_shear_one = [fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ4 = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)

println("Computing ρ5")
fuzzy_shear_one = [fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ5 = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
