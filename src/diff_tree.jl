module diff_tree

include("metrics.jl")

using .metrics

export diff_Galaxy, hyperparameters, diff_kd_tree, estimator, generate_output, combined_function, diff_Galaxy_Circle, calculate_radius

using AbstractTrees
using Statistics
using Base.Threads
using UnicodePlots

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

mutable struct diff_Galaxy_Circle{T, r, g}
    center::Vector{T}
    radius::r
    galaxies::Vector{g}
end

struct hyperparameters
    bin_size::Float64
    max_depth::Float64
    cell_minimum_count::Float64
end

function calculate_radius(galaxies)
    ra_list = [galaxy.ra for galaxy in galaxies]
    dec_list = [galaxy.dec for galaxy in galaxies]
    average_position_ra = mean(ra_list)
    average_position_dec = mean(dec_list)
    radius = maximum([sqrt((galaxy.ra - average_position_ra)^2 + (galaxy.dec - average_position_dec)^2) for galaxy in galaxies])
    return radius
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

        if depth == max_depth || number_galaxies <= cell_minimum_count 
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

        all_leaves = vcat(left_leaves, right_leaves)
        if can_merge(all_leaves)
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
hyperparams = hyperparameters(0.06, 2000000.0, 1.0)
@time begin
    output = diff_kd_tree(galaxies, hyperparams)
end


ra_leaves = [mean([galaxy.ra for galaxy in leaf]) for leaf in output]
dec_leaves = [mean([galaxy.dec for galaxy in leaf]) for leaf in output]
scatterplot_galaxies = scatterplot(ra_leaves, dec_leaves, title="Object Positions", xlabel="RA", ylabel="DEC")
println(scatterplot_galaxies)
println(length(output))
println(maximum([length(leaf) for leaf in output]))

galaxy_circles = [diff_Galaxy_Circle([mean([galaxy.ra for galaxy in leaf]), mean([galaxy.dec for galaxy in leaf])], calculate_radius(leaf), leaf) for leaf in output]

num_blocks = 100
for i in 1:num_blocks:length(galaxy_circles)
    for j in 1:num_blocks:length(galaxy_circles)
        galaxy_list_i = galaxy_circles[i:min(i+num_blocks-1, length(galaxy_circles))]
        galaxy_list_j = galaxy_circles[j:min(j+num_blocks-1, length(galaxy_circles))]
        indices_ij = [(k,l) for k in [i:min(i+num_blocks-1, length(galaxy_circles))] for l in [j:min(j+num_blocks-1, length(galaxy_circles))]]
        above_diagonal = [k >= l for (k,l) in indices_ij]
        if !all(above_diagonal)
            subblock = build_distance_subblock(galaxy_list_i, galaxy_list_j)
            for ii in 1:size(subblock, 1)
                for jj in 1:size(subblock, 2)
                    global_i = i + ii - 1
                    global_j = j + jj - 1
                    if global_i >= global_j
                        subblock[ii, jj] = NaN
                    end
                end
            end
        end
        subblock = subblock[.!isnan.(subblock)]
    end
end


println(mean([length(leaf) for leaf in output]))
#println(output)
#println(maximum([length(leaf) for leaf in output]))
#println(output[1])
#println(length(output[1]))

θ_bins = 10 .^ range(log10(1), log10(400), length=100)
# Check the merging condition


# Note: Do this with 100 x 100 blocks instead!
# psuedo code:
# ra = ...
# dec = ...
# ra_bins = 100
# dec_bins = 100
# ra_edges = range(minimum(ra), stop=maximum(ra), length=ra_bins)
# dec_edges = range(minimum(dec), stop=maximum(dec), length=dec_bins)
# for each bin 
# build_distance_matrix
#
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
