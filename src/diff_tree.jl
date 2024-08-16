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

struct hyperparameters
    bin_size::Int
    max_depth::Int
    cell_minimum_count::Int
end

function diff_kd_tree(galaxies::Vector{T}, hyperparameters::hyperparameters) where T
    bin_size = hyperparameters.bin_size
    max_depth = hyperparameters.max_depth
    cell_minimum_count = hyperparameters.cell_minimum_count

    function calculate_radius(galaxies)
        ra_list = [galaxy.ra for galaxy in galaxies]
        dec_list = [galaxy.dec for galaxy in galaxies]
        average_position_ra = mean(ra_list)
        average_position_dec = mean(dec_list)
        radius = maximum([Vincenty_Formula([average_position_ra, average_position_dec], [galaxy.ra, galaxy.dec]) for galaxy in galaxies])
        return radius
    end

    function can_merge(leaves)
        for i in 1:length(leaves)
            for j in i+1:length(leaves)
                radius_i = calculate_radius(leaves[i])
                radius_j = calculate_radius(leaves[j])
                distance_ij = Vincenty_Formula(
                    [mean([galaxy.ra for galaxy in leaves[i]]), mean([galaxy.dec for galaxy in leaves[i]])],
                    [mean([galaxy.ra for galaxy in leaves[j]]), mean([galaxy.dec for galaxy in leaves[j]])]
                )
                if (radius_i + radius_j) / distance_ij >= bin_size
                    return false
                end
            end
        end
        return true
    end

    function build_tree(galaxies, depth)
        number_galaxies = length(galaxies)

         if number_galaxies <= 1
            radius = 0
        else
            radius = calculate_radius(galaxies)
        end

        if depth == max_depth || number_galaxies <= cell_minimum_count || can_merge([galaxies])
            if depth == max_depth
                println("Max depth reached")
            end
            return [galaxies]
        end
    
        ra_list = [galaxy.ra for galaxy in galaxies]
        dec_list = [galaxy.dec for galaxy in galaxies]
        ra_extent = maximum(ra_list) - minimum(ra_list)
        dec_extent = maximum(dec_list) - minimum(dec_list)
        
        if ra_extent > dec_extent
            median_value = median(ra_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.ra <= median_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.ra > median_value]
        else 
            median_value = median(dec_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.dec <= median_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.dec > median_value]
        end
            

        left_leaves = build_tree(galaxies_left, depth + 1)
        right_leaves = build_tree(galaxies_right, depth + 1)

        return vcat(left_leaves, right_leaves)
    end

    leaves = build_tree(galaxies, 0)

    return leaves
end

positions = rand(100, 2)  # 100 points in 2D space
quantities_one = rand(100)  # 100 random quantities
quantities_two = rand(100)  # 100 random quantities
galaxies = [diff_Galaxy(positions[i, 1], positions[i, 2], quantities_one[i], quantities_two[i]) for i in 1:size(positions, 1)]
hyperparams = hyperparameters(5, 10, 1)
output = diff_kd_tree(galaxies, hyperparams)

function estimator(leaves) 
    c1 = [sum([galaxy.corr1 for galaxy in leaf]) for leaf in leaves]
    c2 = [sum([galaxy.corr2 for galaxy in leaf]) for leaf in leaves]
    return sum(c1 .* c2) / length(c1)
end

function generate_output(galaxies::Vector{diff_Galaxy}, hyperparameters::hyperparameters)
    output = kd_tree(galaxies, hyperparameters)
    return output
end

function combined_function(galaxies::Vector{diff_Galaxy}, hyperparameters::hyperparameters)
    output = generate_output(galaxies, hyperparameters)
    return estimator(output)
end

grads = Zygote.gradient(estimator, output)
grads_galaxies, grads_hyperparameters = Zygote.gradient((g, h) -> combined_function(g, h), galaxies, hyperparameters)

end
