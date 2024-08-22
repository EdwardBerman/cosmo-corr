using Zygote

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
