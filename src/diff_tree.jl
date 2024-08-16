module kdtree 

include("metrics.jl")
using .metrics

export Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_galaxy_cells!, populate, get_leaves, collect_leaves

using AbstractTrees
using Statistics
using Base.Threads

using Zygote

# Define the kd_tree function
function kd_tree(data::Matrix{T}, hyperparameters::Dict{Symbol, Any}) where T
    # Extract hyperparameters
    bin_size = hyperparameters[:bin_size]
    max_depth = hyperparameters[:max_depth]

    # Initialize the kd-tree structure
    function build_tree(data, depth)
        n, d = size(data)
        
        if n <= bin_size || depth == max_depth
            return data  # Return the data as a leaf
        end

        # Determine splitting axis (cycle through axes)
        axis = depth % d + 1

        # Sort data along the chosen axis
        sorted_data = sort(data, by=row -> row[axis])

        # Choose the median as the pivot
        median_idx = div(n, 2)
        median_value = sorted_data[median_idx, axis]

        # Recursively build left and right subtrees
        left_tree = build_tree(sorted_data[1:median_idx-1, :], depth + 1)
        right_tree = build_tree(sorted_data[median_idx:end, :], depth + 1)

        return (left_tree, median_value, right_tree)
    end

    # Build the tree and return the leaves
    tree = build_tree(data, 0)

    return tree
end

# Example usage with automatic differentiation
data = rand(100, 2)  # 100 points in 2D space
hyperparameters = Dict(:bin_size => 5, :max_depth => 10)

# Create the kd-tree and differentiate the tree with respect to the data
tree, back = Zygote.pullback(kd_tree, data, hyperparameters)
grads = back(ones(size(tree)))

