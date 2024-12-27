using Random
using LinearAlgebra
using Distributions
using Plots

function distance_squared(x, center)
    return sum((x .- center).^2)
end

function kmeans_plusplus_weighted_initialization(data, k, random_weights, weight_factor=0.5)
    rng = MersenneTwister(1234)
    n, d = size(data)
    centers = zeros(k, d)
    centers[1, :] = data[rand(1:n), :]
    for i in 2:k
        distances = map(x -> minimum([distance_squared(x, center) for center in eachrow(centers[1:i-1, :])]), eachrow(data))
        combined_weights = weight_factor .* distances .+ (1 .- weight_factor) .* random_weights
        probs = combined_weights / sum(combined_weights)
        centers[i, :] = data[rand(rng, Categorical(probs)), :]
    end

    return centers
end

