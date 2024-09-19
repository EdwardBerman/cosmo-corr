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
    centers[1, :] = data[rand(1:n), :]
    for i in 2:k
        distances = map(x -> minimum([distance_squared(x, center) for center in eachrow(centers[1:i-1, :])]), eachrow(data))
        combined_weights = weight_factor .* distances .+ (1 .- weight_factor) .* random_weights
        probs = combined_weights / sum(combined_weights)
        centers[i, :] = data[rand(Categorical(probs)), :]
    end

    return centers
end

data = randn(100, 2)  
k = 5  
random_weights = rand(100)  

centers = kmeans_plusplus_weighted_initialization(data, k, random_weights)

scatter(data[:, 1], data[:, 2], label="Data", title="K-means++ Initialization", xlabel="X", ylabel="Y")
scatter!(centers[:, 1], centers[:, 2], marker=:star5, markersize=10, label="Centers", color=:red)

