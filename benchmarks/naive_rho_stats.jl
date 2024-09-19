include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
using UnicodePlots
using Base.Threads
using Statistics
using Distributions
using FITSIO

println(nthreads())
struct fuzzy_shear
    shear::Vector{Float64}
end


f = FITS("revised_apr_f115w_shopt_xy_info.fits")

ra = read(f[2], "ra")
dec = read(f[2], "dec")
σ_D = read(f[2], "sig_vignet")
g1_D = read(f[2], "g1_vignet")
g2_D = read(f[2], "g2_vignet")
σ_psf = read(f[2], "sig_psfex")
g1_psf = read(f[2], "g1_psfex")
g2_psf = read(f[2], "g2_psfex")

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
nrows, ncols = size(ra_dec)

initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec, n_clusters, rand(nrows), 0.5)
initial_centers = initial_centers'
initial_weights = rand(nrows, n_clusters)

println("Computing ρ1")
fuzzy_shear_one = [astrocorr.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ1, distances = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
plt = lineplot(log10.(distances), log10.(abs.(ρ1)), title="ρ1", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
println(plt)

println("Computing ρ2")
fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ2, distances = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
plt = lineplot(log10.(distances), log10.(abs.(ρ2)), title="ρ2", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
println(plt)

println("Computing ρ3")
fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i] * δ_TT[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ3, distances = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
plt = lineplot(log10.(distances), log10.(abs.(ρ3)), title="ρ3", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
println(plt)

println("Computing ρ4")
fuzzy_shear_one = [astrocorr.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ4, distances = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
plt = lineplot(log10.(distances), log10.(abs.(ρ4)), title="ρ4", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
println(plt)

println("Computing ρ5")
fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ5, distances = fuzzy_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, n_clusters, 200.0*60*0.03/3600, 10, 5000.0*60*0.03/3600; spacing="log", verbose=true)
plt = lineplot(log10.(distances), log10.(abs.(ρ5)), title="ρ5", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
println(plt)
