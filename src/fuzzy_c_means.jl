using Zygote

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

function calculate_weights(current_weights, data, centers, fuzziness, dist_metric=Vincenty_Formula)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(current_weights)
    dists = [dist_metric(data[:,i], centers[:,j]) for i in 1:size(data,2), j in 1:size(centers,2)]
    weights = [1.0 / sum((dists[i,j]/dists[i,k])^pow for k in 1:ncols) for i in 1:nrows, j in 1:ncols]
    return weights
end

function calculate_centers(current_centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    T = eltype(centers)
    for j in 1:ncols
        num = zeros(T, size(data,1))
        den = zero(T)
        for i in 1:nrows
            δm = weights[i,j]^fuzziness
            num += δm * data[:,i]
            den += δm
        end
        centers[:,j] = num/den
    end
end

function fuzzy_c_means(data, n_clusters, fuzziness, dist_metric=Vincenty_Formula, tol=1e-6, max_iter=1000)
    nrows, ncols = size(data)
    centers = rand(nrows, n_clusters)
    weights = rand(nrows, n_clusters)
    current_iteration = 0
    while current_iteration < max_iter
        centers = calculate_centers(centers, data, weights, fuzziness)
        weights = calculate_weights(weights, data, centers, fuzziness, dist_metric)
        if sum(abs2, new_centers - centers) < tol
            break
        end
    return centers, weights
end


#=
# Define the range filtering function
function filter_in_range(matrix::AbstractMatrix{T}, min_val::T, max_val::T) where T
    return map(x -> (x > min_val && x <= max_val) ? x : zero(T), matrix)
end

# Define the function that outputs sums for different increments
function sum_intervals(matrix::AbstractMatrix{T}) where T
    return [sum(filter_in_range(matrix, min_val, min_val + 0.1)) for min_val in 0:0.1:0.9]
end

# Example usage
matrix = rand(3, 3)  # A 3x3 random matrix

# Compute the sum over all intervals
interval_sums = sum_intervals(matrix)
println(interval_sums)

# Compute the Jacobian with respect to the input matrix
jac = Zygote.jacobian(sum_intervals, matrix)

# Output
println("Original Matrix:")
println(matrix)
println("\nSums for Each Interval:")
println(interval_sums)
println("\nJacobian of the Sums w.r.t. the Input Matrix:")
println(jac[1])
=#
