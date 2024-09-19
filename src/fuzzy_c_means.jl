using Zygote
using UnicodePlots
using Random
using LinearAlgebra
using Statistics

function Vincenty_Formula(coord1::Vector{Float64}, coord2::Vector{Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Vector{Any}, coord2::Vector{Any})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Tuple{Float64, Float64}, coord2::Tuple{Float64, Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60 
end

Zygote.@nograd Vincenty_Formula

struct fuzzy_shear
    shear::Vector{Float64}
end

function calculate_weights(current_weights, data, centers, fuzziness, dist_metric=Vincenty_Formula)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(current_weights)
    ϵ = 1e-10
    dists = [dist_metric(data[:,i], centers[:,j]) for i in 1:size(data,2), j in 1:size(centers,2)]
    weights = [1.0 / sum(( (dists[i,j] + ϵ) /(dists[i,k] + ϵ))^pow for k in 1:ncols) for i in 1:nrows, j in 1:ncols]
    return weights
end

function calculate_centers(current_centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    centers = hcat([sum(weights[i,j]^fuzziness * data[:,i] for i in 1:nrows) / sum(weights[i,j]^fuzziness for i in 1:nrows) for j in 1:ncols]...)
    return centers
end

function fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, fuzziness, dist_metric=Vincenty_Formula, tol=1e-6, max_iter=15)
    centers = initial_centers
    weights = initial_weights
    current_iteration = 0
    while current_iteration < max_iter
        old_centers = copy(centers)
        old_weights = copy(weights)
        centers = calculate_centers(centers, data, weights, fuzziness)
        weights = calculate_weights(weights, data, centers, fuzziness, dist_metric)
        current_iteration += 1
        if sum(abs2, weights - old_weights) < tol
            break
        end
    end
    return centers, weights, current_iteration
end

function weighted_average(quantity, weights)
    weighted_sum = quantity' * weights  
    sum_weights = sum(weights, dims=1)
    weighted_average = weighted_sum ./ sum_weights
    return weighted_average
end

data = hcat([90 .* rand(2) for i in 1:500]...)  
shear_one = rand(500)
shear_two = rand(500)
n_clusters = 100
nrows, ncols = size(data)
initial_centers = rand(nrows, n_clusters)
initial_weights = rand(ncols, n_clusters)

function calculate_direction(x_1, x_2, y_1, y_2, z_1, z_2)
    euclidean_distance_squared = (x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2
    cosA = (z_1 - z_2) + 0.5 * z_2 * euclidean_distance_squared
    sinA = y_1 * x_2 - x_1 * y_2
    r = Complex(sinA, -cosA) 
    return r 
end
    
function fuzzy_shear_estimator(fuzzy_distance)
    ra1, dec1, ra2, dec2 = fuzzy_distance[1][1], fuzzy_distance[1][2], fuzzy_distance[2][1], fuzzy_distance[2][2]
    x1, y1, z1 = cos(ra1 * π / 180) * cos(dec1 * π / 180), sin(ra1 * π / 180) * cos(dec1 * π / 180), sin(dec1 * π / 180)
    x2, y2, z2 = cos(ra2 * π / 180) * cos(dec2 * π / 180), sin(ra2 * π / 180) * cos(dec2 * π / 180), sin(dec2 * π / 180)

    r21 = calculate_direction(x2, x1, y2, y1, z2, z1)
    ϕ21 = real(conj(r21) * r21 / norm(r21)^2) # rotating 2 in the direction of 1

    r12 = calculate_direction(x1, x2, y1, y2, z1, z2)
    ϕ12 = real(conj(r12) * r12 / norm(r12)^2) # rotating 1 in the direction of 2

    object_one_shear_one = fuzzy_distance[1][3]
    object_one_shear_two = fuzzy_distance[1][4]
    object_two_shear_one = fuzzy_distance[2][3]
    object_two_shear_two = fuzzy_distance[2][4]

    object_one_shear_one_rotated_factor = -exp(-2im * ϕ12) * (object_one_shear_one[1] + (object_one_shear_one[2] * 1im))
    object_one_shear_one_rotated = [real(object_one_shear_one_rotated_factor), imag(object_one_shear_one_rotated_factor)]
    
    object_two_shear_two_rotated_factor = -exp(-2im * ϕ21) * (object_two_shear_two[1] + (object_two_shear_two[2] * 1im))
    object_two_shear_two_rotated = [real(object_two_shear_two_rotated_factor), imag(object_two_shear_two_rotated_factor)]

    object_one_shear_two_rotated_factor = -exp(-2im * ϕ12) * (object_one_shear_two[1] + (object_one_shear_two[2] * 1im))
    object_one_shear_two_rotated = [real(object_one_shear_two_rotated_factor), imag(object_one_shear_two_rotated_factor)]

    object_two_shear_one_rotated_factor = -exp(-2im * ϕ21) * (object_two_shear_one[1] + (object_two_shear_one[2] * 1im))
    object_two_shear_one_rotated = [real(object_two_shear_one_rotated_factor), imag(object_two_shear_one_rotated_factor)]

    return dot(object_one_shear_one_rotated, object_two_shear_two_rotated) + dot(object_one_shear_two_rotated, object_two_shear_one_rotated)
end

function sigmoid_bump_function(fuzzy_dist, a, b; sharpness=10) # assume a < x < b
    ϵ = 1e-10
    return 1 / (1 + exp(-sharpness * (x - a))) * 1 / (1 + exp(sharpness * (x - b)))
end


function fuzzy_correlator(ra::Vector{Float64}, 
        dec::Vector{Float64},
        quantity_one::Vector{fuzzy_shear},
        quantity_two::Vector{fuzzy_shear},
        initial_centers, 
        initial_weights,
        nclusters,
        θ_min,
        number_bins,
        θ_max;
        spacing = "linear",
        fuzziness=2.0, 
        dist_metric=Vincenty_Formula, 
        tol=1e-6, 
        verbose=false,
        max_iter=1000)

    data = hcat([[ra[i], dec[i]] for i in 1:500]...)

    centers, weights, iterations = fuzzy_c_means(data, nclusters, initial_centers, initial_weights, fuzziness, dist_metric, tol, max_iter)
    
    if verbose == true
        println("Fuzzy C Means Converged in $iterations iterations")
    end
    
    quantity_one_shear_one = [quantity_one[i].shear[1] for i in 1:length(quantity_one)]
    quantity_one_shear_two = [quantity_one[i].shear[2] for i in 1:length(quantity_one)]
    quantity_two_shear_one = [quantity_two[i].shear[1] for i in 1:length(quantity_two)]
    quantity_two_shear_two = [quantity_two[i].shear[2] for i in 1:length(quantity_two)]

    weighted_shear_one = [[weighted_average(quantity_one_shear_one, weights)[i], weighted_average(quantity_one_shear_two, weights)[i]] for i in 1:nclusters]
    weighted_shear_two = [[weighted_average(quantity_two_shear_one, weights)[i], weighted_average(quantity_two_shear_two, weights)[i]] for i in 1:nclusters]

    fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:nclusters]

    fuzzy_distances = [(fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:nclusters, j in 1:nclusters if i < j] # check this

    if spacing == "linear"
        bins = range(θ_min, θ_max, length=number_bins)
    elseif spacing == "log"
        bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
    end
    
    filter_weights = [sigmoid_bump_function(fuzzy_distance, bins[i], bins[i+1]) 
                  for i in 1:length(bins)-1, fuzzy_distance in fuzzy_distances]

    fuzzy_estimates = [fuzzy_shear_estimator(fuzzy_distances[j]) * filter_weights[i, j]
                   for i in 1:size(filter_weights, 1), j in 1:size(filter_weights, 2)]

    fuzzy_correlations = [ sum(fuzzy_estimates[i, :]) / sum(filter_weights[i, :])
                          for i in 1:size(fuzzy_estimates, 1)] 
    return fuzzy_correlations
end

function scale_data(rand_ra, rand_dec, rand_shear_one, rand_shear_two, b)
    scaled_ra = b .* rand_ra
    scaled_dec = b .* rand_dec
    scaled_shear_one = [fuzzy_shear(b .* s.shear) for s in rand_shear_one]
    scaled_shear_two = [fuzzy_shear(b .* s.shear) for s in rand_shear_two]
    return scaled_ra, scaled_dec, scaled_shear_one, scaled_shear_two
end

function scaled_fuzzy_correlator(b, rand_ra, rand_dec, rand_shear_one, rand_shear_two, initial_centers, initial_weights)
    scaled_ra, scaled_dec, scaled_shear_one, scaled_shear_two = scale_data(rand_ra, rand_dec, rand_shear_one, rand_shear_two, b)
    return fuzzy_correlator(scaled_ra, scaled_dec, scaled_shear_one, scaled_shear_two, initial_centers, initial_weights, 100, 0.1, 10, 1.0, verbose=false)[1]
end

rand_ra = 90 .* rand(500)
rand_dec = 90 .* rand(500)
rand_shear_one = [fuzzy_shear(rand(2)) for i in 1:500]
rand_shear_two = [fuzzy_shear(rand(2)) for i in 1:500]


b = 1.0
grad_b = Zygote.gradient(b -> scaled_fuzzy_correlator(b, rand_ra, rand_dec, rand_shear_one, rand_shear_two, initial_centers, initial_weights), b)

println("Gradient w.r.t. b: ", grad_b)

#output = fuzzy_correlator(rand_ra, rand_dec, rand_shear_one, rand_shear_two, initial_centers, initial_weights, 100, 0.1, 10, 1.0, verbose=true)[1]

println(output)
