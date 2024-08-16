module kdtree 

include("metrics.jl")
using .metrics

export Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_galaxy_cells!, populate, get_leaves, collect_leaves

using AbstractTrees
using Statistics
using Base.Threads

using Zygote

function kd_tree(data::Matrix{T}, hyperparameters::Dict{Symbol, Any}) where T
    bin_size = hyperparameters[:bin_size]
    max_depth = hyperparameters[:max_depth]

    function build_tree(data, depth)
        n, d = size(data)
        
        if n <= bin_size || depth == max_depth
            return data  # Return the data as a leaf
        end

        axis = argmax([maximum(data[:, i]) - minimum(data[:, i]) for i in 1:d])

        sorted_data = sort(data, by=row -> row[axis])
        median_idx = div(n, 2)
        median_value = sorted_data[median_idx, axis]

        left_tree = build_tree(sorted_data[1:median_idx-1, :], depth + 1)
        right_tree = build_tree(sorted_data[median_idx:end, :], depth + 1)

        return (left_tree, median_value, right_tree)
    end

    tree = build_tree(data, 0)

    return tree
end

data = rand(100, 2)  # 100 points in 2D space
hyperparameters = Dict{Symbol, Any}(:bin_size => 5, :max_depth => 10)
output = kd_tree(data, hyperparameters)


tree, back = Zygote.pullback(kd_tree, data, hyperparameters)
grads = back(ones(size(tree)))

