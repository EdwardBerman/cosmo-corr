module fuzzy
include("metrics.jl")

using .metrics

using Zygote
using ForwardDiff
using UnicodePlots
using Random
using LinearAlgebra
using Statistics
using Distributions

export fuzzy_c_means, fuzzy_shear_estimator, fuzzy_correlator, fuzzy_shear, weighted_average, calculate_direction, calculate_weights, calculate_centers, kmeans_plusplus_weighted_initialization_vincenty

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

struct fuzzy_shear
    shear::Vector{Float64}
end

function calculate_weights(current_weights, data, centers, fuzziness, dist_metric=Vincenty_Formula)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(current_weights)
    ϵ = 1e-10
    dists = [max(dist_metric(data[:,i], centers[:,j]), ϵ) for i in 1:size(data,2), j in 1:size(centers,2)]
    weights = [1.0 / sum(( (dists[i,j] + ϵ) /(dists[i,k] + ϵ))^pow for k in 1:ncols) for i in 1:nrows, j in 1:ncols]
    return weights
end

function calculate_centers(current_centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    centers = hcat([sum(weights[i,j]^fuzziness * data[:,i] for i in 1:nrows) / (1e-10 + sum(weights[i,j]^fuzziness for i in 1:nrows)) for j in 1:ncols]...)
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
    weighted_average = weighted_sum ./ (sum_weights .+ 1e-6)
    return weighted_average
end

function calculate_direction(x_1, x_2, y_1, y_2, z_1, z_2)
    euclidean_distance_squared = (x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2
    cosA = (z_1 - z_2) + 0.5 * z_2 * euclidean_distance_squared
    sinA = y_1 * x_2 - x_1 * y_2
    r = Complex(sinA, -cosA) 
    return r
end

function fuzzy_shear_estimator(fuzzy_distance)
    ϵ = 1e-10
    ra1, dec1, ra2, dec2 = fuzzy_distance[1][1], fuzzy_distance[1][2], fuzzy_distance[2][1], fuzzy_distance[2][2]
    x1, y1, z1 = cos(ra1 * π / 180) * cos(dec1 * π / 180), sin(ra1 * π / 180) * cos(dec1 * π / 180), sin(dec1 * π / 180)
    x2, y2, z2 = cos(ra2 * π / 180) * cos(dec2 * π / 180), sin(ra2 * π / 180) * cos(dec2 * π / 180), sin(dec2 * π / 180)

    r21 = calculate_direction(x2, x1, y2, y1, z2, z1)
    ϕ21 = real(conj(r21) * r21 / (ϵ + norm(r21)^2)  ) # rotating 2 in the direction of 1

    r12 = calculate_direction(x1, x2, y1, y2, z1, z2)
    ϕ12 = real(conj(r12) * r12 / (ϵ + norm(r12)^2)) # rotating 1 in the direction of 2

    object_one_shear_one = fuzzy_distance[1][3]
    object_one_shear_two = fuzzy_distance[1][4]
    object_two_shear_one = fuzzy_distance[2][3]
    object_two_shear_two = fuzzy_distance[2][4]

    object_one_shear_one_rotated_factor = -exp(-2im * ϕ12) * (object_one_shear_one[1] + (object_one_shear_one[2] * 1im))
    object_one_shear_one_rotated = [real(object_one_shear_one_rotated_factor), imag(object_one_shear_one_rotated_factor)]
    
    object_two_shear_two_rotated_factor = -exp(-2im * ϕ21) * (object_two_shear_two[1] + (object_two_shear_two[2] * 1im))
    object_two_shear_two_rotated = [real(object_two_shear_two_rotated_factor), imag(object_two_shear_two_rotated_factor)]

    object_one_shear_two_rotated_factor = -exp(-2im * ϕ12) * (object_one_shear_two[1] + (object_one_shear_two[2] * 1im))
    object_one_shear_two_rotated = [real(object_one_shear_two_rotated_factor), imag(object_one_shear_two_rotated_factor)]

    object_two_shear_one_rotated_factor = -exp(-2im * ϕ21) * (object_two_shear_one[1] + (object_two_shear_one[2] * 1im))
    object_two_shear_one_rotated = [real(object_two_shear_one_rotated_factor), imag(object_two_shear_one_rotated_factor)]

    return dot(object_one_shear_one_rotated, object_two_shear_two_rotated) + dot(object_one_shear_two_rotated, object_two_shear_one_rotated)
end

function sigmoid_bump_function(fuzzy_dist, a, b; sharpness=10) # assume a < x < b
    ϵ = 1e-10
    return (1 / (ϵ +  (1 + exp(-sharpness * (fuzzy_dist - a)))) ) * (1 / (ϵ +  (1 + exp(sharpness * (fuzzy_dist - b))) ))
end


function fuzzy_correlator(ra::Vector{<:Real}, 
        dec::Vector{<:Real},
        quantity_one::Vector{fuzzy_shear},
        quantity_two::Vector{fuzzy_shear},
        initial_centers, 
        initial_weights,
        nclusters,
        θ_min,
        number_bins,
        θ_max;
        spacing = "linear",
        fuzziness=2.0, 
        dist_metric=Vincenty_Formula, 
        tol=1e-6, 
        verbose=false,
        max_iter=1000)

    data = hcat([[ra[i], dec[i]] for i in 1:length(ra)]...)

    centers, weights, iterations = fuzzy_c_means(data, nclusters, initial_centers, initial_weights, fuzziness, dist_metric, tol, max_iter)

    if verbose == true
        println("Fuzzy C Means Converged in $iterations iterations")
        println("Size centers: ", size(centers))
        println("Size weights: ", size(weights))
        println("Size new_weights: ", size(new_weights))
    end
    
    quantity_one_shear_one = [quantity_one[i].shear[1] for i in 1:length(quantity_one)]
    quantity_one_shear_two = [quantity_one[i].shear[2] for i in 1:length(quantity_one)]
    quantity_two_shear_one = [quantity_two[i].shear[1] for i in 1:length(quantity_two)]
    quantity_two_shear_two = [quantity_two[i].shear[2] for i in 1:length(quantity_two)]

    weighted_shear_one = [[weighted_average(quantity_one_shear_one, weights)[i], weighted_average(quantity_one_shear_two, weights)[i]] for i in 1:nclusters]
    weighted_shear_two = [[weighted_average(quantity_two_shear_one, weights)[i], weighted_average(quantity_two_shear_two, weights)[i]] for i in 1:nclusters]

    fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:nclusters]

    fuzzy_distances = [[fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])] for i in 1:nclusters, j in 1:nclusters if i < j] # check this
    ϵ = 1e-10
    if spacing == "linear"
        bins = range(θ_min, θ_max, length=number_bins)
    elseif spacing == "log"
        bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
    end
    
    filter_weights = [sigmoid_bump_function(fuzzy_distance[3], bins[i], bins[i+1]) 
                  for i in 1:length(bins)-1, fuzzy_distance in fuzzy_distances]

    fuzzy_estimates = [fuzzy_shear_estimator(fuzzy_distances[j]) * filter_weights[i, j]
                   for i in 1:size(filter_weights, 1), j in 1:size(filter_weights, 2)]

    fuzzy_correlations = [ sum(fuzzy_estimates[i, :]) / (ϵ + sum(filter_weights[i, :]))
                          for i in 1:size(fuzzy_estimates, 1)] 
    mean_weighted_distances = [sum(filter_weights[i, :]) for i in 1:size(filter_weights, 1)]
    return fuzzy_correlations, mean_weighted_distances, bins
end

data = rand(100, 2)
initial_centers = kmeans_plusplus_weighted_initialization_vincenty(data, 3, rand(100), 0.5)
initial_weights = rand(100, 3)
fuzzy_shear_one = [fuzzy_shear([rand(), rand()]) for i in 1:100]
fuzzy_shear_two = [fuzzy_shear([rand(), rand()]) for i in 1:100]

function finite_diff(f, x, h = 1e-6)
    (f(x + h) - f(x)) / h
end

ra = rand(100)
dec = rand(100)
fuzzy_shear_vector = [[rand(), rand()] for i in 1:100]

println(minimum([Vincenty_Formula(collect(ra), collect(center)) for center in eachrow(initial_centers)]))
println(maximum([Vincenty_Formula(collect(ra), collect(center)) for center in eachrow(initial_centers)]))

@time begin
    intermediate = ForwardDiff.derivative(x -> fuzzy_correlator(ra .* x .^2, x*dec, fuzzy_shear_one, fuzzy_shear_two, initial_centers, initial_weights, 3, 10.0, 10, 100.0, spacing="linear", fuzziness=1.5, dist_metric=Vincenty_Formula, tol=1e-6, verbose=false, max_iter=1000)[1], 90.0)
end

println(intermediate)

end
