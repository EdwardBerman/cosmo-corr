using Random
using LinearAlgebra
using Distributions
using Plots

# Helper function to calculate squared distances
function distance_squared(x, center)
    return sum((x .- center).^2)
end

# Function to initialize centers using weighted distances and random sampling
function kmeans_plusplus_weighted_initialization(data, k, random_weights, weight_factor=0.5)
    n, d = size(data)
    centers = zeros(k, d)

    # Step 1: Select the first center randomly
    centers[1, :] = data[rand(1:n), :]

    # Step 2-4: Select the remaining centers using a weighted combination of distances and random weights
    for i in 2:k
        # Calculate squared distances to the nearest center
        distances = map(x -> minimum([distance_squared(x, center) for center in eachrow(centers[1:i-1, :])]), eachrow(data))

        # Combine distances with random weights
        combined_weights = weight_factor .* distances .+ (1 .- weight_factor) .* random_weights

        # Normalize combined weights to create a probability distribution
        probs = combined_weights / sum(combined_weights)
        
        # Select the next center based on the new probabilities
        centers[i, :] = data[rand(Categorical(probs)), :]
    end

    return centers
end

# Example usage
data = randn(100, 2)  # 100 points in 2 dimensions
k = 5  # Number of clusters
random_weights = rand(100)  # Precompute random weights for all points

centers = kmeans_plusplus_weighted_initialization(data, k, random_weights)

# Plot the data and centers
scatter(data[:, 1], data[:, 2], label="Data", title="K-means++ Initialization", xlabel="X", ylabel="Y")
scatter!(centers[:, 1], centers[:, 2], marker=:star5, markersize=10, label="Centers", color=:red)

