module kdtree 

include("metrics.jl")
using .metrics

export Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_galaxy_cells!, populate, get_leaves, collect_leaves

using AbstractTrees
using Statistics
using Base.Threads

using Zygote
using Statistics

struct diff_Galaxy 
    ra::Float64
    dec::Float64
    corr1::Any
    corr2::Any
end

function kd_tree(galaxies::Vector{T}, hyperparameters::Dict{Symbol, Any}) where T
    bin_size = hyperparameters[:bin_size]
    max_depth = hyperparameters[:max_depth]
    cell_minimum_count = hyperparameters[:cell_minimum_count]
    spatial_dimensions = hyperparameters[:spatial_dimensions]
    leaves = []

    function build_tree(galaxies, depth)
        number_galaxies = length(galaxies)
        
        if number_galaxies <= bin_size || depth == max_depth || number_galaxies <= cell_minimum_count
            push!(leaves, galaxies)
            return galaxies
        end
    
        ra_list = [galaxy.ra for galaxy in galaxies]
        dec_list = [galaxy.dec for galaxy in galaxies]
        ra_extent = maximum(ra_list) - minimum(ra_list)
        dec_extent = maximum(dec_list) - minimum(dec_list)
        
        if ra_extent > dec_extent
            axis = 1
            median_value = median(ra_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.ra <= median_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.ra > median_value]
        else 
            axis = 2
            median_value = median(dec_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.dec <= median_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.dec > median_value]
        end
            

        left_tree = build_tree(galaxies_left, depth + 1)
        right_tree = build_tree(galaxies_right, depth + 1)
    end

    tree = build_tree(galaxies, 0)

    return leaves
end

positions = rand(100, 2)  # 100 points in 2D space
quantities_one = rand(100)  # 100 random quantities
quantities_two = rand(100)  # 100 random quantities
galaxies = [diff_Galaxy(positions[i, 1], positions[i, 2], quantities_one[i], quantities_two[i]) for i in 1:size(positions, 1)]
hyperparameters = Dict{Symbol, Any}(:bin_size => 5, :max_depth => 10, :cell_minimum_count => 1, :spatial_dimensions => 2)
output = kd_tree(galaxies, hyperparameters)

function estimator(leaves) 
    c1 = [sum([galaxy.corr1 for galaxy in leaf]) for leaf in leaves]
    c2 = [sum([galaxy.corr2 for galaxy in leaf]) for leaf in leaves]
    return sum(c1 .* c2) / length(c1)
end

grads = Zygote.gradient(estimator, output)

end
