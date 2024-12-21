include("../src/corr.jl")
using .astrocorr

using PyCall
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

np = pyimport("numpy")

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
n_clusters = 50
nrows, ncols = size(ra_dec)

initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec, n_clusters, rand(nrows), 0.5)
initial_centers = initial_centers'
initial_weights = rand(nrows, n_clusters)

function bootstrap(ra, dec, shear_one, shear_two, n_clusters, subset_size=50, n_iterations=10)
    nrows = length(ra)
    ρ = []
    distances = []
    for _ in 1:n_iterations
        leave_out_indices = randperm(nrows)[1:subset_size]
        subset_ra = vcat(ra[1:leave_out_indices[1] - 1], ra[leave_out_indices[end] + 1:end])
        subset_dec = vcat(dec[1:leave_out_indices[1] - 1], dec[leave_out_indices[end] + 1:end])
        subset_shear_one = vcat(shear_one[1:leave_out_indices[1] - 1], shear_one[leave_out_indices[end] + 1:end])
        subset_shear_two = vcat(shear_two[1:leave_out_indices[1] - 1], shear_two[leave_out_indices[end] + 1:end])
        ra_dec_subset = hcat(subset_ra, subset_dec)
        initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec_subset, n_clusters, rand(length(subset_ra)), 0.5)
        initial_weights = rand(length(subset_ra), n_clusters)
        ρ_now, distance = probabilistic_correlator(
            subset_ra, subset_dec, subset_shear_one, subset_shear_two, 
            initial_centers, initial_weights, n_clusters, 
            0.0, 9, 45.0;
            spacing="linear", verbose=false
        )
        push!(ρ, ρ_now)
        push!(distances, distance)
    end
    
    return ρ, distances
end

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ1_list, distances = bootstrap(ra, dec, fuzzy_shear_two, fuzzy_shear_two, n_clusters)
ρ_means, ρ_stds = mean_and_std(ρ1_list)
np.save("rho1_means_bootstrap.npy", ρ_means)
np.save("rho1_stds_bootstrap.npy", ρ_stds)
println(size(ρ_means), size(ρ_stds))
distances = distances[1]
np.save("distances_bootstrap.npy", distances)

absolute_ρ_means = abs.(ρ_means)
lower_bounds = absolute_ρ_means .- ρ_stds
upper_bounds = absolute_ρ_means .+ ρ_stds

#log_lower_bounds = log10.(lower_bounds)
#log_upper_bounds = log10.(upper_bounds)

#f = Figure(fontsize = 30)
#Axis(f[1, 1], xlabel="θ [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ1 Bootstrap")
#ylims!(current_axis(), -15, 2)
#errorbars!(ax, distances, log10.(absolute_ρ_means), log10.(upper_bounds) .- log10.(absolute_ρ_means), log10.(absolute_ρ_means) .- log10.(lower_bounds), colormap = :cool, linewidth=2, whiskerwidth=10)
#errorbars!(distances, log10.(abs.(ρ_means)), log10.(ρ_stds),  colormap = :cool, linewidth=2, whiskerwidth = 10)
#scatter!(distances, log10.(abs.(ρ_means)), colormap = :cool)
#save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho1_bootstrap.png", f)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ2_list, distances2 = bootstrap(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = mean_and_std(ρ2_list)
np.save("rho2_means_bootstrap.npy", means)
np.save("rho2_stds_bootstrap.npy", stds)
distances2 = distances2[1]
np.save("distances2_bootstrap.npy", distances2)
println(means)
println(stds)
println(distances2)

f2 = Figure(fontsize = 30)
Axis(f2[1, 1], xlabel="θ [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ2 Bootstrap")
ylims!(current_axis(), -15, 2)
errorbars!(distances2, log10.(abs.(means)), log10.(stds),  colormap = :cool, linewidth=2, whiskerwidth = 10)
scatter!(distances2, log10.(abs.(means)), colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho2_bootstrap.png", f2)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i] * δ_TT[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ3_list, distances3 = bootstrap(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = mean_and_std(ρ3_list)
np.save("rho3_means_bootstrap.npy", means)
np.save("rho3_stds_bootstrap.npy", stds)
distances3 = distances3[1]
np.save("distances3_bootstrap.npy", distances3)

f3 = Figure(fontsize = 30)
Axis(f3[1, 1], xlabel="θ [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ3 Bootstrap")
ylims!(current_axis(), -15, 2)
errorbars!(distances3, log10.(abs.(means)), log10.(stds),  colormap = :cool, linewidth=2, whiskerwidth = 10)
scatter!(distances3, log10.(abs.(means)),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho3_bootstrap.png", f3)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ4_list, distances4 = bootstrap(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = mean_and_std(ρ4_list)
np.save("rho4_means_bootstrap.npy", means)
np.save("rho4_stds_bootstrap.npy", stds)
distances4 = distances4[1]
np.save("distances4_bootstrap.npy", distances4)

f4 = Figure(fontsize = 30)
Axis(f4[1, 1], xlabel="θ [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ4 Bootstrap")
ylims!(current_axis(), -15, 2)
errorbars!(distances4, log10.(abs.(means)), log10.(stds),  colormap = :cool, linewidth=2, whiskerwidth = 10)
scatter!(distances4, log10.(abs.(means)), colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho4_bootstrap.png", f4)

fuzzy_shear_one = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.probabilistic_fuzzy.fuzzy.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ5_list, distances5 = bootstrap(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = mean_and_std(ρ5_list)
np.save("rho5_means_bootstrap.npy", means)
np.save("rho5_stds_bootstrap.npy", stds)
distances5 = distances5[1]
np.save("distances5_bootstrap.npy", distances5)

f5 = Figure(fontsize = 30)
Axis(f5[1, 1], xlabel="θ [arcmin]", ylabel="log₁₀(|ξ(θ)|)", title="ρ5 Bootstrap")
ylims!(current_axis(), -15, 2)
errorbars!(distances5, log10.(abs.(means)), log10.(stds),  colormap = :cool, linewidth=2, whiskerwidth = 10)
scatter!(distances5, log10.(abs.(means)),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho5_bootstrap.png", f5)



