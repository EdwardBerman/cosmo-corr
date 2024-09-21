module probabilistic_fuzzy
include("metrics.jl")
include("fuzzy_c_means.jl")

using .metrics
using .fuzzy

using Zygote
using UnicodePlots
using Random
using LinearAlgebra
using Statistics
using Distributions

export fuzzy_c_means, fuzzy_shear_estimator, fuzzy_correlator, fuzzy_shear, weighted_average, calculate_direction, calculate_weights, calculate_centers, kmeans_plusplus_weighted_initialization_vincenty

function probabilistic_correlator(ra::Vector{Float64}, 
        dec::Vector{Float64},
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

    weighted_shear_one = [[weighted_average(quantity_one_shear_one, new_weights)[i], weighted_average(quantity_one_shear_two, new_weights)[i]] for i in 1:nclusters]
    weighted_shear_two = [[weighted_average(quantity_two_shear_one, new_weights)[i], weighted_average(quantity_two_shear_two, new_weights)[i]] for i in 1:nclusters]

    fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:nclusters]

    fuzzy_distances = [(fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:nclusters, j in 1:nclusters if i < j] # check this
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
end
