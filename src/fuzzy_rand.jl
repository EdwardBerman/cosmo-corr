using Random
using LinearAlgebra
using Distributions
using Statistics
using Plots

# Helper function to calculate squared distances
function distance_squared(x, center)
    return sum((x .- center).^2)
end

# Gumbel-Softmax function for differentiable sampling
function gumbel_softmax(probs, temperature=0.5)
    gumbel_noise = -log.(-log.(rand(length(probs))))
    soft_sample = softmax((log.(probs) .+ gumbel_noise) / temperature)
    return soft_sample
end

# Softmax function
function softmax(x)
    exp_x = exp.(x .- maximum(x))  # For numerical stability
    return exp_x / sum(exp_x)
end

# Function to initialize centers using weighted distances and Gumbel-Softmax sampling
function kmeans_plusplus_differentiable(data, k, random_weights, first_center_idx, weight_factor=0.5, temperature=0.5)
    n, d = size(data)
    centers = zeros(k, d)

    # Step 1: Use the precomputed first center index
    centers[1, :] = data[first_center_idx, :]

    # Step 2-4: Select the remaining centers using a weighted combination of distances and random weights
    for i in 2:k
        # Calculate squared distances to the nearest center
        distances = map(x -> minimum([distance_squared(x, center) for center in eachrow(centers[1:i-1, :])]), eachrow(data))

        # Combine distances with random weights
        combined_weights = weight_factor .* distances .+ (1 .- weight_factor) .* random_weights

        # Normalize combined weights to create a probability distribution
        probs = combined_weights / sum(combined_weights)

        # Use Gumbel-Softmax to get differentiable "sampling"
        soft_sample = gumbel_softmax(probs, temperature)

        # Calculate the next center as a weighted sum (instead of sampling directly)
        centers[i, :] = sum(data .* soft_sample, dims=1)
    end

    return centers
end

# Example usage for 500 points and 100 clusters
data = randn(500, 2)  # 500 points in 2 dimensions
k = 100  # Number of clusters
random_weights = rand(500)  # Precompute random weights for all points

# Precompute the first center index
first_center_idx = rand(1:500)

centers = kmeans_plusplus_differentiable(data, k, random_weights, first_center_idx)

# Plot the data and centers
scatter(data[:, 1], data[:, 2], label="Data", title="K-means++ Initialization (Gumbel-Softmax)", xlabel="X", ylabel="Y")
scatter!(centers[:, 1], centers[:, 2], marker=:star5, markersize=10, label="Centers", color=:red)
