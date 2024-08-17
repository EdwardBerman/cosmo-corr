module diff_tree

include("metrics.jl")
using .metrics

export diff_Galaxy, hyperparameters, diff_kd_tree, estimator, generate_output, combined_function

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
    bin_size::Float64
    max_depth::Float64
    cell_minimum_count::Float64
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

        println("Current Depth: ", depth, " Number of Galaxies: ", number_galaxies, " Radius: ", radius)

        if depth == max_depth || number_galaxies <= cell_minimum_count || (can_merge([galaxies]) && depth > 1)
            if depth == max_depth
                println("Max depth reached")
            end
            if number_galaxies <= cell_minimum_count
                println("Minimum count reached in a cell")
            end
            if can_merge([galaxies]) && depth > 1
                println("Merging Condition Satisfied at depth: ", depth)
            end
            return [galaxies]
        end
    
        ra_list = [galaxy.ra for galaxy in galaxies]
        dec_list = [galaxy.dec for galaxy in galaxies]
        ra_extent = maximum(ra_list) - minimum(ra_list)
        dec_extent = maximum(dec_list) - minimum(dec_list)
        
        if ra_extent > dec_extent
            mean_value = mean(ra_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.ra <= mean_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.ra > mean_value]
        else 
            mean_value = mean(dec_list)
            galaxies_left = [galaxy for galaxy in galaxies if galaxy.dec <= mean_value]
            galaxies_right = [galaxy for galaxy in galaxies if galaxy.dec > mean_value]
        end
            

        left_leaves = build_tree(galaxies_left, depth + 1)
        right_leaves = build_tree(galaxies_right, depth + 1)

        return vcat(left_leaves, right_leaves)
    end

    leaves = build_tree(galaxies, 1)

    return leaves
end

positions = rand(100, 2)  # 100 points in 2D space
quantities_one = rand(100)  # 100 random quantities
quantities_two = rand(100)  # 100 random quantities
galaxies = [diff_Galaxy(positions[i, 1], positions[i, 2], quantities_one[i], quantities_two[i]) for i in 1:size(positions, 1)]
hyperparams = hyperparameters(5.0, 20000.0, 1.0)
output = diff_kd_tree(galaxies, hyperparams)

function estimator(leaves) 
    c1 = [sum([galaxy.corr1 for galaxy in leaf]) for leaf in leaves]
    c2 = [sum([galaxy.corr2 for galaxy in leaf]) for leaf in leaves]
    return sum(c1 .* c2) / length(c1)
end

function generate_output(galaxies::Vector{diff_Galaxy}, hyperparameters::hyperparameters)
    output = diff_kd_tree(galaxies, hyperparameters)
    return output
end

function combined_function(galaxies::Vector{diff_Galaxy}, hyperparameters::hyperparameters)
    output = generate_output(galaxies, hyperparameters)
    return estimator(output)
end

#grads = Zygote.gradient(estimator, output)
grads_galaxies, grads_hyperparameters = Zygote.gradient((g, h) -> combined_function(g, h), galaxies, hyperparams)

end
