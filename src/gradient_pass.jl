module skip_fuzzy
include("metrics.jl")
include("fuzzy_c_means.jl")

using .metrics
using .fuzzy

using ForwardDiff
using DiffRules
using UnicodePlots
using Random
using LinearAlgebra
using Statistics
using Distributions

export skip_correlator

function indicator_function(x, a, b)
    if x < a
        return 0
    elseif x > b
        return 0
    else
        return 1
    end
end

function skip_correlator(ra::Vector{Float64}, 
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
    end
    
    quantity_one_shear_one = [quantity_one[i].shear[1] for i in 1:length(quantity_one)]
    quantity_one_shear_two = [quantity_one[i].shear[2] for i in 1:length(quantity_one)]
    quantity_two_shear_one = [quantity_two[i].shear[1] for i in 1:length(quantity_two)]
    quantity_two_shear_two = [quantity_two[i].shear[2] for i in 1:length(quantity_two)]
    
    assignment_matrix = zeros(size(weights))
    for i in 1:size(weights, 1)
        sample = argmax(weights[i, :])
        for j in 1:size(weights, 2)
            if j == sample
                assignment_matrix[i, j] = 1
            else
                assignment_matrix[i, j] = 0
            end
        end
    end

    weighted_shear_one = [[weighted_average(quantity_one_shear_one, assignment_matrix)[i], weighted_average(quantity_one_shear_two, assignment_matrix)[i]] for i in 1:nclusters]
    weighted_shear_two = [[weighted_average(quantity_two_shear_one, assignment_matrix)[i], weighted_average(quantity_two_shear_two, assignment_matrix)[i]] for i in 1:nclusters]

    fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:nclusters]

    fuzzy_distances = [(fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:nclusters, j in 1:nclusters if i < j] 
    ϵ = 1e-10
    if spacing == "linear"
        bins = range(θ_min, θ_max, length=number_bins)
    elseif spacing == "log"
        bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
    end

    if verbose == true
        println(histogram([fuzzy_distance[3] for fuzzy_distance in fuzzy_distances], nbins=10))
    end
    
    filter_weights = [indicator_function(fuzzy_distance[3], bins[i], bins[i+1]) 
                  for i in 1:length(bins)-1, fuzzy_distance in fuzzy_distances]

    fuzzy_estimates = [fuzzy_shear_estimator(fuzzy_distances[j]) * filter_weights[i, j]
                   for i in 1:size(filter_weights, 1), j in 1:size(filter_weights, 2)]

    fuzzy_correlations = [ sum(fuzzy_estimates[i, :]) / (ϵ + sum(filter_weights[i, :]))
                          for i in 1:size(fuzzy_estimates, 1)] 
    
    mean_weighted_distances = [mean([fuzzy_distances[j][3] for j in 1:size(fuzzy_distances, 1) if filter_weights[i, j] == 1]) 
                               for i in 1:size(filter_weights, 1)]

    mean_weighted_distances = [if sum(filter_weights[i, :]) > 0
                               mean([fuzzy_distances[j][3] for j in 1:size(fuzzy_distances, 1) if filter_weights[i, j] == 1])
                           else
                               NaN
                           end for i in 1:size(filter_weights, 1)]

    fuzzy_correlations = [if sum(filter_weights[i, :]) > 0
                         fuzzy_correlations[i]
                     else
                         NaN
                     end for i in 1:size(filter_weights, 1)]

    return fuzzy_correlations, mean_weighted_distances
end

function fuzzy_galaxies_correlate(ra,
        dec, 
        weighted_quantity_one,
        weighted_quantity_two,
        θ_min,
        number_bins,
        θ_max;
        spacing = "linear",
        dist_metric=Vincenty_Formula)

    fuzzy_distances = [(fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:nclusters, j in 1:nclusters if i < j] 
    ϵ = 1e-10
    if spacing == "linear"
        bins = range(θ_min, θ_max, length=number_bins)
    elseif spacing == "log"
        bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
    end

    if verbose == true
        println(histogram([fuzzy_distance[3] for fuzzy_distance in fuzzy_distances], nbins=10))
    end
    
    filter_weights = [indicator_function(fuzzy_distance[3], bins[i], bins[i+1]) 
                  for i in 1:length(bins)-1, fuzzy_distance in fuzzy_distances]

    fuzzy_estimates = [fuzzy_shear_estimator(fuzzy_distances[j]) * filter_weights[i, j]
                   for i in 1:size(filter_weights, 1), j in 1:size(filter_weights, 2)]

    fuzzy_correlations = [ sum(fuzzy_estimates[i, :]) / (ϵ + sum(filter_weights[i, :]))
                          for i in 1:size(fuzzy_estimates, 1)] 
    
    mean_weighted_distances = [mean([fuzzy_distances[j][3] for j in 1:size(fuzzy_distances, 1) if filter_weights[i, j] == 1]) 
                               for i in 1:size(filter_weights, 1)]

    mean_weighted_distances = [if sum(filter_weights[i, :]) > 0
                               mean([fuzzy_distances[j][3] for j in 1:size(fuzzy_distances, 1) if filter_weights[i, j] == 1])
                           else
                               NaN
                           end for i in 1:size(filter_weights, 1)]

    fuzzy_correlations = [if sum(filter_weights[i, :]) > 0
                         fuzzy_correlations[i]
                     else
                         NaN
                     end for i in 1:size(filter_weights, 1)]

    return fuzzy_correlations, mean_weighted_distances
end

DiffRules.@define_diffrule Main.skip_correlator(ra, dec, quantity_one, quantity_two, initial_centers, initial_weights, nclusters, θ_min, number_bins, θ_max) = custom_rule

function custom_rule(f::typeof(skip_correlator), args::NTuple{10})
    ra, dec, quantity_one, quantity_two, initial_centers, initial_weights, nclusters, θ_min, number_bins, θ_max = args
    
    function fuzzy_correlation_func(ra, dec, quantity_one, quantity_two)
        skip_correlator(ra, dec, quantity_one, quantity_two, initial_centers, initial_weights, nclusters, θ_min, number_bins, θ_max)
    end
    ForwardDiff.gradient(x -> fuzzy_correlation_func(x...), args)
end

num_galaxies = 100
ra = rand(Uniform(0, 360), num_galaxies)  # Right Ascension values between 0 and 360 degrees
dec = rand(Uniform(-90, 90), num_galaxies)  # Declination values between -90 and 90 degrees
quantity_one = [fuzzy_shear([randn(), randn()]) for _ in 1:num_galaxies]
quantity_two = [fuzzy_shear([randn(), randn()]) for _ in 1:num_galaxies]
nclusters = 10
initial_centers = [rand(2) for _ in 1:nclusters]  # Random centers
initial_weights = rand(nclusters, num_galaxies)  # Random weights
initial_weights .= initial_weights ./ sum(initial_weights, dims=1)  # Normalize weights
θ_min = 0.01
θ_max = 5.0
number_bins = 20

gradient = ForwardDiff.gradient(skip_correlator, (ra, dec, quantity_one, quantity_two, initial_centers, initial_weights, nclusters, θ_min, number_bins, θ_max))
print(gradient)

end

