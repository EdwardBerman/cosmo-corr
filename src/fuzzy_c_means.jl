using Zygote
using UnicodePlots
using Random

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

struct shear
    tan_cross::Vector{Float64}
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

function fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, fuzziness, dist_metric=Vincenty_Formula, tol=1e-6, max_iter=1000)
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

data = hcat([90 .* rand(2) for i in 1:100]...)  
shear_one = rand(100)
shear_two = rand(100)
n_clusters = 3
nrows, ncols = size(data)
initial_centers = rand(nrows, n_clusters)
initial_weights = rand(ncols, n_clusters)
centers, weights, iterations = fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, 2.0)
@info "Converged in $iterations iterations"

weighted_shear_one = weighted_average(shear_one, weights)
weighted_shear_two = weighted_average(shear_two, weights)

fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:n_clusters]
fuzzy_distances = [(fuzzy_galaxies[i], 
                    fuzzy_galaxies[j], 
                    Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:n_clusters, j in 1:n_clusters if i < j] 

function combine_vectors_to_matrix(vec1, datapoint, vec2)
    result = vcat(vec1, datapoint, vec2)
    return result
end

function filter_in_range(matrix::AbstractMatrix{T}, min_val::T, max_val::T) where T
    return map(x -> (x > min_val && x <= max_val) ? x : zero(T), matrix)
end
    
function fuzzy_shear_rotator(fuzzy_distance)
    ra1, dec1, ra2, dec2 = fuzzy_distance[1][1], fuzzy_distance[1][2], fuzzy_distance[2][1], fuzzy_distance[2][2]
    x1, y1, z1 = cos(ra1) * cos(dec1), sin(ra1) * cos(dec1), sin(dec1)
    x2, y2, z2 = cos(ra2) * cos(dec2), sin(ra2) * cos(dec2), sin(dec2)

    euclidean_distance_squared = (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2
    cosA = (z1 - z2) + 0.5 * z2 * euclidean_distance_squared
    sinA = y1 * x2 - x1 * y2
    r2 = Complex(sinA, -cosA) # because we are computing the direction of 2 in the frame of 1, we do r2 
    ϕ2 = Real(conj(r2) * r2 / norm(r2)^2) # imaginary component is zero, but we cast as real regardless
    
    cosA = (z2 - z1) + 0.5 * z1 * euclidean_distance_squared
    sinA = y2 * x1 - x2 * y1
    r1 = Complex(sinA, -cosA) 
    ϕ1 = Real(conj(r1) * r1 / norm(r1)^2)

    shear_one = fuzzy_distance[1][3:4]
    shear_two = fuzzy_distance[2][3:4]

    shear_one_weight_one = shear_one[1]
    shear_two_weight_two = shear_two[2]
    shear_one_weight_two = shear_one[2]
    shear_two_weight_one = shear_two[1]

    shear_one_rotated = @. -exp(-2im * ϕ1) * (shear_one_weight_one + (shear_one_weight_two * 1im))
    shear_two_rotated = @. -exp(-2im * ϕ2) * (shear_two_weight_one + (shear_two_weight_two * 1im))

    return dot(shear_one, shear_two_rotated) + dot(shear_two, shear_one_rotated)
end

function fuzzy_correlator(ra, 
        dec, 
        quantity_one, 
        quantity_two, 
        nclusters, 
        initial_centers, 
        initial_weights, 
        number_bins,
        fuzziness=2.0, 
        dist_metric=Vincenty_Formula, 
        tol=1e-6, 
        max_iter=1000)

    data = hcat([ra, dec]...)

    centers, weights, iterations = fuzzy_c_means(data, nclusters, initial_centers, initial_weights, fuzziness, dist_metric, tol, max_iter)
    
    quantity_one_shear_one = [quantity_one.tan_cross[1] for i in 1:length(quantity_one)]
    quantity_one_shear_two = [quantity_one.tan_cross[2] for i in 1:length(quantity_one)]
    quantity_two_shear_one = [quantity_two.tan_cross[1] for i in 1:length(quantity_two)]
    quantity_two_shear_two = [quantity_two.tan_cross[2] for i in 1:length(quantity_two)]

    weighted_shear_one = [weighted_average(quantity_one_shear_one, weights), weighted_average(quantity_one_shear_two, weights)]
    weighted_shear_two = [weighted_average(quantity_two_shear_one, weights), weighted_average(quantity_two_shear_two, weights)]
    fuzzy_galaxies = [[centers[1,i], centers[2,i], weighted_shear_one[i], weighted_shear_two[i]] for i in 1:nclusters]
    fuzzy_distances = [(fuzzy_galaxies[i], 
                        fuzzy_galaxies[j], 
                        Vincenty_Formula(fuzzy_galaxies[i][1:2], fuzzy_galaxies[j][1:2])) for i in 1:nclusters, j in 1:nclusters if i < j]
    #binned_fuzzy_distances = ...
    correlations = [mean([fuzzy_shear_rotator(fuzzy_distance) for fuzzy_distance in binned_fuzzy_distance[i]]) for i in 1:number_bins]
end

@time begin
    grad_data = Zygote.gradient(data -> sum(fuzzy_c_means(data, n_clusters, initial_centers, initial_weights, 2.0)[1]), data)
    println("Gradient with respect to data", grad_data)
    println(size(grad_data[1]))
    println(heatmap(grad_data[1]', title="Gradient with respect to data", xlabel="Longitude", ylabel="Latitude", colormap=:coolwarm))
end

println(scatterplot(data[1,:], data[2,:], title="Data Points", xlabel="Longitude", ylabel="Latitude"))
println(scatterplot(centers[1,:], centers[2,:], title="Cluster Centers", xlabel="Longitude", ylabel="Latitude"))
println(heatmap(weights', title="Weights", xlabel="Data Point", ylabel="Cluster", colormap=:coolwarm))
println(heatmap(weights', title="Weights", xlabel="Data Point", ylabel="Cluster", colormap=:cool))

