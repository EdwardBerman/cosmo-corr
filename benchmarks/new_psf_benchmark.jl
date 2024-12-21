using DifferentialEquations, Plots
using Plots.PlotMeasures
using Zygote, SciMLSensitivity
using Statistics
using LinearAlgebra
using UnicodePlots
using Distances
using StatsBase
using Distributions
using CairoMakie
using FITSIO

function Vincenty_Formula(coord1::Vector{Float64}, coord2::Vector{Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Tuple{Float64, Float64}, coord2::Tuple{Float64, Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60 
end

function kmeans_plusplus_weighted_initialization_vincenty(data, k, random_weights, weight_factor=0.5)
    n, d = size(data)
    centers = zeros(k, d)
    centers[1, :] = data[rand(1:n), :]
    for i in 2:k
        distances = map(x -> minimum([Vincenty_Formula(collect(x), collect(center)) for center in eachrow(centers[1:i-1, :])]), eachrow(data))
        combined_weights = weight_factor .* distances .+ (1 .- weight_factor) .* random_weights
        probs = combined_weights / sum(combined_weights)
        centers[i, :] = data[rand(Categorical(probs)), :]
    end

    return centers'
end

function gravitational_ode_3d!(du, u, p, t)
    x, y, z, vx, vy, vz = u
    G = p[1]
    M = p[2]  

    r = sqrt(x^2 + y^2 + z^2)

    ax = -G * M * x / r^3
    ay = -G * M * y / r^3
    az = -G * M * z / r^3

    du[1] = vx            # dx/dt = vx
    du[2] = vy            # dy/dt = vy
    du[3] = vz            # dz/dt = vz
    du[4] = ax            # dvx/dt = ax
    du[5] = ay            # dvy/dt = ay
    du[6] = az            # dvz/dt = az
end

function calculate_weights(current_weights, data, centers, fuzziness, dist_metric=Vincenty_Formula)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(current_weights)
    ϵ = 1e-10
    dists = [dist_metric(data[:,i], centers[:,j]) for i in 1:size(data,2), j in 1:size(centers,2)]
    weights = [1.0 / sum(( (dists[i,j] + ϵ) /(dists[i,k] + ϵ))^pow for k in 1:ncols) for i in 1:nrows, j in 1:ncols]
    return weights
end

function calculate_centers(current_centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    centers = hcat([sum(weights[i,j]^fuzziness * data[:,i] for i in 1:nrows) / sum(weights[i,j]^fuzziness for i in 1:nrows) for j in 1:ncols]...)
    return centers
end

function fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, fuzziness, dist_metric=Vincenty_Formula, tol=1e-6, max_iter=1000)
    centers = initial_centers
    weights = initial_weights
    current_iteration = 0
    while current_iteration < max_iter
        old_centers = copy(centers)
        old_weights = copy(weights)
        centers = calculate_centers(centers, data, weights, fuzziness)
        weights = calculate_weights(weights, data, centers, fuzziness, dist_metric)
        current_iteration += 1
        if sum(abs2, weights - old_weights) < tol
            break
        end
    end
    return centers, weights, current_iteration
end

function weighted_average(quantity, weights)
    weighted_sum = quantity' * weights  
    sum_weights = sum(weights, dims=1)
    weighted_average = weighted_sum ./ sum_weights
    return weighted_average
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


data = [[ra[i], dec[i]] for i in 1:length(ra)]
data = hcat(data...)
println(size(data), "data")
n_clusters = 100
nrows, ncols = size(data)
initial_centers = rand(nrows, n_clusters)
println(size(initial_centers))
initial_weights = rand(ncols, n_clusters)
centers, weights, iterations = fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, 2.0)
@info "Converged in $iterations iterations"

weight_matrix_plot = Plots.heatmap(weights', title="Weight Matrix", xlabel="Source", ylabel="Cluster Center", color=:cool, size=(1200, 600), titlefont=font("Courier New", 16, weight=:bold, italic=true), guidefont=font("Courier New", 14), tickfont=font("Courier New", 12), colorbar_titlefont=font("Courier New", 10), margin=10mm)
Plots.savefig(weight_matrix_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/weight_matrix_heatmap.png")

#=
distances = [Vincenty_Formula(data[:,i], data[:,j]) for i in 1:length(ra), j in 1:length(ra) if i < j]
fuzz_distances = [Vincenty_Formula(centers[:,i], centers[:,j]) for i in 1:n_clusters, j in 1:n_clusters if i < j]

Plots.histogram(distances, normalize=true, nbins=100, label="Simulated Vincenty Distances", color=:blue, alpha=0.5, lw=2, xlabel="Distance", ylabel="Frequency", legend=:topright)
Plots.histogram!(fuzz_distances, normalize=true, nbins=100, label="Fuzzy Cluster Vincenty Distances", color=:pink, alpha=0.5, lw=2)

# Display the plot
Plots.savefig("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/distance_histograms.png")  # Optionally save the plot to a file

println(UnicodePlots.histogram(fuzz_distances, nbins = 10))

ra_edges = range(minimum(data[1, :]), maximum(data[1, :]), length=10)
dec_edges = range(minimum(data[2, :]), maximum(data[2, :]), length=10)
ra_dec_hist = fit(Histogram, (data[1, :], data[2, :]), (ra_edges, dec_edges))
ϵ = 1e-10
normalized_hist = (ra_dec_hist.weights .+ ϵ)  ./ sum(ra_dec_hist.weights .+ ϵ)
ra_edges_centers = range(minimum(centers[1, :]), maximum(centers[1, :]), length=10)
dec_edges_centers = range(minimum(centers[2, :]), maximum(centers[2, :]), length=10)
ra_dec_hist_centers = fit(Histogram, (centers[1, :], centers[2, :]), (ra_edges_centers, dec_edges_centers))
normalized_hist_centers = (ra_dec_hist_centers.weights .+ ϵ) ./ sum(ra_dec_hist_centers.weights .+ ϵ)
normalized_hist = vec(normalized_hist)
normalized_hist_centers = vec(normalized_hist_centers)
kl_divergence = round(Distances.kl_divergence(normalized_hist, normalized_hist_centers), digits=2)

kmeans_plusplus_centers = kmeans_plusplus_weighted_initialization_vincenty(data', n_clusters, weights[:, 1], 0.5)
ra_edges_kmpp = range(minimum(kmeans_plusplus_centers[:, 1]), maximum(kmeans_plusplus_centers[:, 1]), length=10)
dec_edges_kmpp = range(minimum(kmeans_plusplus_centers[:, 2]), maximum(kmeans_plusplus_centers[:, 2]), length=10)
ra_dec_hist_kmpp = fit(Histogram, (kmeans_plusplus_centers[:, 1], kmeans_plusplus_centers[:, 2]), (ra_edges_kmpp, dec_edges_kmpp))
normalized_hist_kmpp = (ra_dec_hist_kmpp.weights .+ ϵ) ./ sum(ra_dec_hist_kmpp.weights .+ ϵ)
normalized_hist_kmpp = vec(normalized_hist_kmpp)

aug_centers, weights, iterations = fuzzy_c_means(data, n_clusters, kmeans_plusplus_centers', initial_weights, 2.0)
@info "Converged in $iterations iterations"
=#
#=
kl_divergence_kmpp = round(Distances.kl_divergence(normalized_hist, normalized_hist_kmpp), digits=2)

scatter_plot = Plots.scatter([ra_coord for ra_coord in ra], [dec_coord for dec_coord in dec],
                                 title="COSMOS Field Point Sources and the Cluster Centers", 
    xlabel="Ra", ylabel="Dec", # xlabel size 
    size=(1200, 600),  # Increase the size of the plot
    titlefont=font("Courier New", 12, weight=:bold, italic=true), guidefont=font("Courier New", 12),
    label="COSMOS Field", legend=:topright,
    markersize=3, color=:lightblue,
    bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)  # Increase the margins

# Plot centers on top of it
Plots.scatter!(scatter_plot, centers[1, :], centers[2, :], 
    label="FCM Centers", markersize=5, color="#FF1493")


#Plots.scatter!(scatter_plot, kmeans_plusplus_centers[:, 1], kmeans_plusplus_centers[:, 2], 
#    label="KMPP Centers", markersize=5, color="#FFD700")

scatter_plot_alt = Plots.scatter([ra_coord for ra_coord in ra], [dec_coord for dec_coord in dec],
                                 title="COSMOS Field Point Sources and the Cluster Centers", 
    xlabel="RA", ylabel="Dec", # xlabel size 
    size=(1200, 600),  # Increase the size of the plot
    titlefont=font("Courier New", 12, weight=:bold, italic=true), guidefont=font("Courier New", 12),
    label="Simulated Points", legend=:topright,
    markersize=3, color=:lightblue,
    bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)  # Increase the margins

# Plot centers on top of it
Plots.scatter!(scatter_plot_alt, centers[1, :], centers[2, :], 
    label="FCM Centers", markersize=5, color="#FF1493")

Plots.scatter!(scatter_plot_alt, aug_centers[1, :], aug_centers[2, :], 
    label="FCM + KMPP Centers", markersize=5, color="#00FF00")

# Save the scatter plot with cool colorscheme
Plots.savefig(scatter_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/cosmos_field_cluster_scatter.png")
Plots.savefig(scatter_plot_alt, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/cosmos_field_cluster_scatter_alt.png")
=#

#=
weight_matrix_plot = Plots.heatmap(weights, title="Weight Matrix", xlabel="Cluster Center", ylabel="Source", color=:cool, size=(1200, 600), bottom_margin=50px, left_margin=100px, right_margin=100px, top_margin=10px, titlefont=font("Courier New", 16, weight=:bold, italic=true), guidefont=font("Courier New", 14), tickfont=font("Courier New", 12), colorbar_titlefont=font("Courier New", 10))
Plots.savefig(weight_matrix_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/weight_matrix_heatmap.png")

println([maximum(weights[i, :]) for i in 1:size(weights, 1)])
println([sum(-1 .* weights[i, :] .* log.(weights[i, :])) for i in 1:size(weights, 1)])

entropy_plot = Plots.bar([sum(-1 .* weights[i, :] .* log.(weights[i, :])) for i in 1:size(weights, 1)], title="Entropy of Each Source Assignment Distribution", xlabel="Galaxy", ylabel="Entropy", color=:cool, size=(1200, 600), bottom_margin=50px, left_margin=100px, right_margin=100px, top_margin=10px, titlefont=font("Courier New", 16, weight=:bold, italic=true), guidefont=font("Courier New", 14), tickfont=font("Courier New", 12))
Plots.savefig(entropy_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/entropy_barplot.png")

entropy_histogram = Plots.histogram([sum(-1 .* weights[i, :] .* log.(weights[i, :])) for i in 1:size(weights, 1)], nbins=100, title="Entropy of Each Galaxy Assignment Distribution", xlabel="Entropy", ylabel="Frequency", color=:cool, size=(1200, 600), bottom_margin=50px, left_margin=100px, right_margin=100px, top_margin=10px, titlefont=font("Courier New", 16, weight=:bold, italic=true), guidefont=font("Courier New", 14), tickfont=font("Courier New", 12))
Plots.savefig(entropy_histogram, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/entropy_histogram.png")

f = CairoMakie.Figure(fonts = (; regular = "Courier New", weird = "Blackchancery"))
CairoMakie.Axis(f[1, 1], xlabel="Entropy", title="Entropy of Each Source Assignment Distribution", ylabel="Frequency", titlefont = "Courier New")
CairoMakie.hist!([sum(-1 .* weights[i, :] .* log.(weights[i, :])) for i in 1:size(weights, 1)], nbins=1000, color = :lightblue)
CairoMakie.save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/cairo_entropy_histogram.png", f)

f2 = CairoMakie.Figure(fonts = (; regular = "Courier New", weird = "Blackchancery"))
CairoMakie.Axis(f2[1, 1], xlabel="Max Probability", title="Max Probability of Each Source Assignment Distribution", ylabel="Frequency", titlefont = "Courier New")
CairoMakie.hist!([maximum(weights[i, :]) for i in 1:size(weights, 1)], nbins=1000, color = :pink)
CairoMakie.save("/home/eddieberman/research/mcclearygroup/AstroCorr/assets/cairo_max_probability_histogram.png", f2)
=#
