using DifferentialEquations, Plots
using Plots.PlotMeasures
using Zygote, SciMLSensitivity
using Statistics
using LinearAlgebra
using UnicodePlots
using Distances
using StatsBase

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

#u0_3d = [x0, y0, z0, vx0, vy0, vz0]

function generate_initial_conditions(nbodies)
    return [[randn(), randn(), randn(), randn(), randn(), randn()] for i in 1:nbodies]
end

initial_conditions = generate_initial_conditions(500)
tspan = (0.0, 10.0)

G = [1.0]
M = 1.0
p = [G[1], M]  

function xyz_to_ra_dec(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    dec = acos(z / r) * (180 / π )
    ra = atan(y/x) * (180 / π)
    return [ra, dec]
end

function gravity_sim(p; u0_3d = initial_conditions, nbodies = 500)
    G = p[1]
    M = p[2]
    p = [G, M]
    prob_3d_list = [ODEProblem(gravitational_ode_3d!, u0, tspan, p) for u0 in u0_3d]
    solulations_3d = [solve(prob_3d) for prob_3d in prob_3d_list]
    end_states = [sol_3d.u[end] for sol_3d in solulations_3d]
    end_states_ra_dec = [xyz_to_ra_dec(end_state[1], end_state[2], end_state[3]) for end_state in end_states]
    return end_states_ra_dec
end

end_states_ra_dec = gravity_sim(p)

println(size(end_states_ra_dec))

distances = [Vincenty_Formula(end_states_ra_dec[i], end_states_ra_dec[j]) for i in 1:500, j in 1:500 if i < j]
println(UnicodePlots.histogram(distances, nbins = 100))

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

function fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, fuzziness, dist_metric=Vincenty_Formula, tol=1e-6, max_iter=15)
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


data = [[end_states_ra_dec[i][1], end_states_ra_dec[i][2]] for i in 1:500]
data = hcat(data...)

n_clusters = 100
nrows, ncols = size(data)
initial_centers = rand(nrows, n_clusters)
initial_weights = rand(ncols, n_clusters)
centers, weights, iterations = fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, 2.0)
@info "Converged in $iterations iterations"

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

scatter_plot = Plots.scatter([coord[1] for coord in end_states_ra_dec], [coord[2] for coord in end_states_ra_dec],
    title="Simulated Points and the Fuzzy C Means Cluster Centers (KL Divergence: $kl_divergence)", 
    xlabel="RA", ylabel="Dec", # xlabel size 
    size=(1200, 600),  # Increase the size of the plot
    titlefont=font("Courier New", 12, weight=:bold, italic=true), guidefont=font("Courier New", 12),
    label="Simulated Points", legend=:topright,
    markersize=3, color=:lightblue,
    bottom_margin=100px, left_margin=100px, right_margin=100px, top_margin=50px)  # Increase the margins

# Plot centers on top of it
Plots.scatter!(scatter_plot, centers[1, :], centers[2, :], 
    label="Cluster Centers", markersize=5, color="#FF1493")

# Save the scatter plot with cool colorscheme
Plots.savefig(scatter_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/end_states_and_centers_scatter.png")

weight_matrix_plot = Plots.heatmap(weights, title="Weight Matrix", xlabel="Cluster Center", ylabel="Galaxy", color=:cool, size=(1200, 600), bottom_margin=50px, left_margin=100px, right_margin=100px, top_margin=10px, titlefont=font("Courier New", 16, weight=:bold, italic=true), guidefont=font("Courier New", 14), tickfont=font("Courier New", 12), colorbar_titlefont=font("Courier New", 10))
Plots.savefig(weight_matrix_plot, "/home/eddieberman/research/mcclearygroup/AstroCorr/assets/weight_matrix_heatmap.png")

