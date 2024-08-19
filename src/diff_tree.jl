module diff_tree

include("metrics.jl")
using .metrics

export diff_Galaxy, hyperparameters, diff_kd_tree, estimator, generate_output, combined_function

using AbstractTrees
using Statistics
using Base.Threads

using Zygote
using Statistics
using PyCall
using Distributions

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
    #should_stop = Ref(false)  # Boolean flag to stop recursion

    function calculate_radius(galaxies)
        ra_list = [galaxy.ra for galaxy in galaxies]
        dec_list = [galaxy.dec for galaxy in galaxies]
        average_position_ra = mean(ra_list)
        average_position_dec = mean(dec_list)
        radius = maximum([Vincenty_Formula([galaxy.ra, galaxy.dec], [average_position_ra, average_position_dec]) for galaxy in galaxies])
        #radius = maximum([sqrt((galaxy.ra - average_position_ra)^2 + (galaxy.dec - average_position_dec)^2) for galaxy in galaxies])
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
                #distance_ij = sqrt((mean([galaxy.ra for galaxy in leaves[i]]) - mean([galaxy.ra for galaxy in leaves[j]]))^2 + (mean([galaxy.dec for galaxy in leaves[i]]) - mean([galaxy.dec for galaxy in leaves[j]]))^2)
                if ((radius_i + radius_j) / distance_ij) >= bin_size
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

        #println("Current Depth: ", depth, " Number of Galaxies: ", number_galaxies, " Radius: ", radius)

        if depth == max_depth || number_galaxies <= cell_minimum_count 
            #=
            if depth == max_depth
                println("Max depth reached")
            end
            if number_galaxies <= cell_minimum_count
                println("Minimum count reached in a cell")
            end
            =#
            return [galaxies]
        end

        #if should_stop[]  # Check the flag to stop recursion
         #   return [galaxies]  # Return the galaxies as they are
        #end
    
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

        all_leaves = vcat(left_leaves, right_leaves)
        if can_merge(all_leaves)
            #should_stop[] = true
            return [vcat(all_leaves...)]
        else
            return all_leaves
        end
    end

    leaves = build_tree(galaxies, 1)

    return leaves
end

fits = pyimport("astropy.io.fits")
f = fits.open("../benchmarks/Aardvark.fit")
print(f[2].data)

ra = f[2].data["RA"]
dec = f[2].data["DEC"]
ra = convert(Vector{Float64}, ra) 
dec = convert(Vector{Float64}, dec) 

ra = ra .* π / 180
dec = dec .* π / 180

ra_min = minimum(ra)
ra_max = maximum(ra)
dec_min = minimum(dec)
dec_max = maximum(dec)

rand_ra = rand(Uniform(ra_min, ra_max), length(ra)) 
rand_sin_dec = rand(Uniform(sin(dec_min), sin(dec_max)), length(dec))
rand_dec = asin.(rand_sin_dec) 

rand_ra_pi = rand_ra ./ π 
rand_cos_dec = cos.(rand_dec)
mask = (rand_cos_dec .< 0.1 .*(1 .+ 2 .*rand_ra_pi .+ 8 .*(rand_ra_pi).^2)) .& (rand_cos_dec .< 0.1 .*(1 .+ 2 .*(0.5 .- rand_ra_pi) .+ 8 .*(0.5 .-rand_ra_pi) .^2)) 
rand_ra, rand_dec = rand_ra[mask], rand_dec[mask]
rand_ra .*= 180 / π
rand_dec .*= 180 / π

positions = hcat(rand_ra, rand_dec)
quantities_one = rand(length(rand_ra))  # 100 random quantities
quantities_two = rand(length(rand_ra))  # 100 random quantities
galaxies = [diff_Galaxy(positions[i, 1], positions[i, 2], quantities_one[i], quantities_two[i]) for i in 1:size(positions, 1)]
hyperparams = hyperparameters(2.0, 2000000.0, 1.0)
@time begin
    output = diff_kd_tree(galaxies, hyperparams)
end
println(length(output))
#println(output)
println(mean([length(leaf) for leaf in output]))
#println(maximum([length(leaf) for leaf in output]))
#println(output[1])
#println(length(output[1]))

θ_bins = 10 .^ range(log10(1), log10(400), length=100)
# Check the merging condition
function calculate_radius(galaxies)
    ra_list = [galaxy.ra for galaxy in galaxies]
    dec_list = [galaxy.dec for galaxy in galaxies]
    average_position_ra = mean(ra_list)
    average_position_dec = mean(dec_list)
    radius = maximum([sqrt((galaxy.ra - average_position_ra)^2 + (galaxy.dec - average_position_dec)^2) for galaxy in galaxies])
    return radius
end

for i in 1:length(output)
    for j in i+1:length(output)
        max_ratio = 0
        radius_i = calculate_radius(output[i])
        radius_j = calculate_radius(output[j])
        mean_position_i = [mean([galaxy.ra for galaxy in output[i]]), mean([galaxy.dec for galaxy in output[i]])]
        mean_position_j = [mean([galaxy.ra for galaxy in output[j]]), mean([galaxy.dec for galaxy in output[j]])]
        distance_ij = Vincenty_Formula(mean_position_i, mean_position_j)
        # find which θ bin the distance_ij falls into
        bin_index = searchsortedfirst(θ_bins, distance_ij)
        ratio = (radius_i + radius_j) / distance_ij
        if ratio > max_ratio
            max_ratio = ratio
        end
        if i == length(output) && j == length(output)
            println("Max ratio: ", max_ratio)
            println("Distance: ", distance_ij)
            println("Radius i: ", radius_i)
            println("Radius j: ", radius_j)
        end
    end
end




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

#grads_galaxies, grads_hyperparameters = Zygote.gradient((g, h) -> combined_function(g, h), galaxies, hyperparams)
#println(grads_galaxies)
#println(grads_hyperparameters)


end
