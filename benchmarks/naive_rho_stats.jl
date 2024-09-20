include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
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

function jackknife(ra, dec, shear_one, shear_two, n_clusters, subset_size=10)
    nrows = length(ra)
    ρ = []
    distances = []
    for i in 1:subset_size:nrows
        leave_out_indices = i:min(i + subset_size - 1, nrows)
        subset_ra = vcat(ra[1:leave_out_indices[1] - 1], ra[leave_out_indices[end] + 1:end])
        subset_dec = vcat(dec[1:leave_out_indices[1] - 1], dec[leave_out_indices[end] + 1:end])
        subset_shear_one = vcat(shear_one[1:leave_out_indices[1] - 1], shear_one[leave_out_indices[end] + 1:end])
        subset_shear_two = vcat(shear_two[1:leave_out_indices[1] - 1], shear_two[leave_out_indices[end] + 1:end])
        ra_dec_subset = hcat(subset_ra, subset_dec)
        initial_centers = kmeans_plusplus_weighted_initialization_vincenty(ra_dec_subset, n_clusters, rand(length(subset_ra)), 0.5)
        initial_weights = rand(length(subset_ra), n_clusters)
        ρ_now, distance = fuzzy_correlator(
            subset_ra, subset_dec, subset_shear_one, subset_shear_two, 
            initial_centers, initial_weights, n_clusters, 
            200.0 * 60 * 0.03 / 3600, 10, 5000.0 * 60 * 0.03 / 3600;
            spacing="log", verbose=false
        )
        push!(ρ, ρ_now)
        push!(distances, distance)
    end
    
    return ρ, distances
end

fuzzy_shear_one = [astrocorr.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ1_list, distances = jackknife(ra, dec, fuzzy_shear_two, fuzzy_shear_two, n_clusters)
ρ_means, ρ_stds = mean_and_std(ρ1_list)
println(ρ_stds)
distances = distances[1]
println(typeof(ρ_means))
println(typeof(ρ_stds))
println(typeof(distances))
println(size(ρ_means))
println(size(ρ_stds))
println(size(distances))

f = Figure()
Axis(f[1, 1], xlabel="log₁₀(θ)", ylabel="log₁₀(|ξ(θ)|)", title="ρ₁")
errorbars!(log10.(distances), log10.(abs.(ρ_means)), log10.(ρ_stds), color = abs.(ρ_means),  colormap = :cool) 
scatter!(log10.(distances), log10.(abs.(ρ_means)),  color = abs.(ρ_means),  colormap = :cool)
save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho1_jackknife.png", f)

#plot1 = Plots.plot(log10.(distances), log10.(abs.(ρ_means)), yerr=ρ_stds, title="ρ1", xlabel="θ", ylabel="ξ(θ)", label="", color=:cool, titlefont=font("Courier New", 20, weight=:bold, italic=true), guidefont=font("Courier New", 16) , tickfont=font("Courier New", 10), legendfont=font("Courier New", 10), lw=5, size=(1200,600), bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px, errorbarcolor=:pink)
#Plots.savefig(plot1, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho1_jackknife.png")

fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(δ_e[i]) for i in 1:length(δ_e)]
ρ2_list, distances2 = jackknife(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = 
plot2 = Plots.plot(distances2, means, yerr=stds, title="ρ2", xlabel="θ", ylabel="ξ(θ)", label="", color=:cool, titlefont=font("Courier New", 20, weight=:bold, italic=true), guidefont=font("Courier New", 16) , tickfont=font("Courier New", 10), legendfont=font("Courier New", 10), lw=5, size=(1200,600), bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)
Plots.savefig(plot2, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho2_jackknife.png")


fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i] * δ_TT[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ3_list, distances3 = jackknife(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = 
plot3 = Plots.plot(distances3, means, yerr=stds, title="ρ3", xlabel="θ", ylabel="ξ(θ)", label="", color=:cool, titlefont=font("Courier New", 20, weight=:bold, italic=true), guidefont=font("Courier New", 16) , tickfont=font("Courier New", 10), legendfont=font("Courier New", 10), lw=5, size=(1200,600), bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)
Plots.savefig(plot3, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho3_jackknife.png")

fuzzy_shear_one = [astrocorr.fuzzy_shear(δ_e_conj[i]) for i in 1:length(δ_e)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ4_list, distances4 = jackknife(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = 
plot4 = Plots.plot(distances4, means, yerr=stds, title="ρ4", xlabel="θ", ylabel="ξ(θ)", label="", color=:cool, titlefont=font("Courier New", 20, weight=:bold, italic=true), guidefont=font("Courier New", 16) , tickfont=font("Courier New", 10), legendfont=font("Courier New", 10), lw=5, size=(1200,600), bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)
Plots.savefig(plot4, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho4_jackknife.png")

fuzzy_shear_one = [astrocorr.fuzzy_shear(e_psf_conj[i]) for i in 1:length(e_psf)]
fuzzy_shear_two = [astrocorr.fuzzy_shear(e_psf[i] * δ_TT[i]) for i in 1:length(e_psf)]
ρ5_list, distances5 = jackknife(ra, dec, fuzzy_shear_one, fuzzy_shear_two, n_clusters)
means, stds = 
plot5 = Plots.plot(distances5, means, yerr=stds, title="ρ5", xlabel="θ", ylabel="ξ(θ)", label="", color=:cool, titlefont=font("Courier New", 20, weight=:bold, italic=true), guidefont=font("Courier New", 16) , tickfont=font("Courier New", 10), legendfont=font("Courier New", 10), lw=5, size=(1200,600), bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)
Plots.savefig(plot5, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/rho5_jackknife.png")



