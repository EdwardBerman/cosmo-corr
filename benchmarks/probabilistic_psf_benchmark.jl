include("../src/corr.jl")
using .astrocorr

using PyCall
using NaNStatistics
using Statistics
using Random
using UnicodePlots
using Base.Threads
using Statistics
using StatsBase
using Distributions
using FITSIO
using CairoMakie

println(nthreads())
struct fuzzy_shear
    shear::Vector{Float64}
end


f = FITS("f115w_shopt_corr_xy_info.fits")

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
n_clusters = 250
nrows, ncols = size(ra_dec)

initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec, n_clusters, rand(nrows), 0.5)
initial_centers = initial_centers'
initial_weights = rand(nrows, n_clusters)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ1_list = []
distances = []
for i in 1:10
    ρ1, distances1 = probabilistic_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 50, 200.0 * 60 * 0.03 / 3600, 10, 100 * 5000.0 * 60 * 0.03 / 3600; spacing="log", verbose=false)
    push!(ρ1_list, ρ1)
    push!(distances, distances1)
end
ρ1_list = hcat(ρ1_list...)
distances = hcat(distances...)
ρ_means = vec(nanmean(ρ1_list, dims=2))
ρ_stds = vec(nanstd(ρ1_list, dims=2))
distances = vec(nanmean(distances, dims=2))

f = Figure()
Axis(f[1, 1], xlabel="log₁₀(θ) [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ1 Sampling")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool) 
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho1_sampling_finer.png", f)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ2_list = []
distances = []
for i in 1:10
    ρ2, distances1 = probabilistic_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 50, 200.0 * 60 * 0.03 / 3600, 10, 100 * 5000.0 * 60 * 0.03 / 3600; spacing="log", verbose=false)
    push!(ρ2_list, ρ2)
    push!(distances, distances1)
end
ρ2_list = hcat(ρ2_list...)
distances = hcat(distances...)
ρ_means = vec(nanmean(ρ2_list, dims=2))
ρ_stds = vec(nanstd(ρ2_list, dims=2))
distances = vec(nanmean(distances, dims=2))

f2 = Figure()
Axis(f2[1, 1], xlabel="log₁₀(θ) [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ2 Sampling")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool)
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho2_sampling_finer.png", f2)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i] * δ_TT[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ3_list = []
distances = []
for i in 1:10
    ρ3, distances1 = probabilistic_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 50, 200.0 * 60 * 0.03 / 3600, 10, 100 * 5000.0 * 60 * 0.03 / 3600; spacing="log", verbose=false)
    push!(ρ3_list, ρ3)
    push!(distances, distances1)
end
ρ3_list = hcat(ρ3_list...)
distances = hcat(distances...)
ρ_means = vec(nanmean(ρ3_list, dims=2))
ρ_stds = vec(nanstd(ρ3_list, dims=2))
distances = vec(nanmean(distances, dims=2))

f3 = Figure()
Axis(f3[1, 1], xlabel="log₁₀(θ) [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ3 Sampling")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool)
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho3_sampling_finer.png", f3)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ4_list = []
distances = []
for i in 1:10
    ρ4, distances1 = probabilistic_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 50, 200.0 * 60 * 0.03 / 3600, 10, 100 * 5000.0 * 60 * 0.03 / 3600; spacing="log", verbose=false)
    push!(ρ4_list, ρ4)
    push!(distances, distances1)
end
ρ4_list = hcat(ρ4_list...)
distances = hcat(distances...)
ρ_means = vec(nanmean(ρ4_list, dims=2))
ρ_stds = vec(nanstd(ρ4_list, dims=2))
distances = vec(nanmean(distances, dims=2))

f4 = Figure()
Axis(f4[1, 1], xlabel="log₁₀(θ) [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ4 Sampling")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool)
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho4_sampling_finer.png", f4)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ5_list = []
distances = []
for i in 1:10
    ρ5, distances1 = probabilistic_correlator(ra, dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 50, 200.0 * 60 * 0.03 / 3600, 10, 100 * 5000.0 * 60 * 0.03 / 3600; spacing="log", verbose=false)
    push!(ρ5_list, ρ5)
    push!(distances, distances1)
end
ρ5_list = hcat(ρ5_list...)
distances = hcat(distances...)
ρ_means = vec(nanmean(ρ5_list, dims=2))
ρ_stds = vec(nanstd(ρ5_list, dims=2))
distances = vec(nanmean(distances, dims=2))

f5 = Figure()
Axis(f5[1, 1], xlabel="log₁₀(θ) [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ5 Sampling")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool)
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho5_sampling_finer.png", f5)

