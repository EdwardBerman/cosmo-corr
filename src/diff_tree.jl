module kdtree 

include("metrics.jl")
using .metrics

export Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_galaxy_cells!, populate, get_leaves, collect_leaves

using AbstractTrees
using Statistics
using Base.Threads

using Zygote

function kd_tree(positions::Matrix{T}, quantities::Vector{S}, hyperparameters::Dict{Symbol, Any}) where {T, S}
    bin_size = hyperparameters[:bin_size]
    max_depth = hyperparameters[:max_depth]
    cell_minimum_count = hyperparameters[:cell_minimum_count]

    function build_tree(data, quantities, depth)
        n, d = size(data)
        
        if n <= bin_size || depth == max_depth || n <= cell_minimum_count
            return positions, quantities
        end
    
        ra_list = data[:, 1]
        dec_list = data[:, 2]
        if !isempty(ra_list) 
            ra_extent = maximum(ra_list) - minimum(ra_list)
        else
            ra_extent = NaN
        end
        if !isempty(dec_list)
            dec_extent = maximum(dec_list) - minimum(dec_list)
        else
            dec_extent = NaN
        end
        if ra_extent > dec_extent && ra_extent !== NaN && dec_extent !== NaN
            axis = 1
        else ra_extent !== NaN && dec_extent !== NaN
            axis = 2
        end

        sorted_data = sort(data, by=row -> row[axis])
        median_idx = div(n, 2)
        median_value = sorted_data[median_idx, axis]

        left_tree = build_tree(sorted_data[1:median_idx-1, :], depth + 1)
        right_tree = build_tree(sorted_data[median_idx:end, :], depth + 1)

        return (left_tree, median_value, right_tree)
    end

    tree = build_tree(data, quantities, 0)

    return tree
end

positions = rand(100, 2)  # 100 points in 2D space
quantities = rand(100)  # 100 random quantities
hyperparameters = Dict{Symbol, Any}(:bin_size => 5, :max_depth => 10, :cell_minimum_count => 1)
output = kd_tree(positions, quantities, hyperparameters)


tree, back = Zygote.pullback(kd_tree, data, hyperparameters)
grads = back(ones(size(tree)))

