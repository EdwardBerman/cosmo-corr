using Zygote

struct Galaxy 
    ra::Float64
    dec::Float64
    corr1::Any
    corr2::Any
end

struct TreeNode
    left::Union{Nothing, TreeNode}
    right::Union{Nothing, TreeNode}
    data::Vector{Galaxy}
end

struct hyperparameters
    bin_size::Float64
    max_depth::Float64
    cell_minimum_count::Float64
end

function build_tree(data::Vector{Galaxy}, hyperparams::hyperparameters; depth::Int=0)::TreeNode
    if length(data) <= hyperparams.cell_minimum_count || depth >= hyperparams.max_depth
        return TreeNode(nothing, nothing, data)
    end
    
    # Determine the splitting point
    mid = length(data) ÷ 2
    
    # Recursively build the left and right subtrees
    left_subtree = build_tree(data[1:mid], hyperparams, depth=depth + 1)
    right_subtree = build_tree(data[mid+1:end], hyperparams, depth=depth + 1)
    
    return TreeNode(left_subtree, right_subtree, data)
end

data = [Galaxy(0.1, 0.2, 0.3, 0.4), Galaxy(0.2, 0.3, 0.4, 0.5), Galaxy(0.3, 0.4, 0.5, 0.6), Galaxy(0.4, 0.5, 0.6, 0.7), Galaxy(0.5, 0.6, 0.7, 0.8)]

tree = build_tree(data, hyperparameters(0.1, 10.0, 1.0))
println("gradient: ", gradient(x -> build_tree(x, hyperparameters(0.1, 3, 1)).data[1].corr1, data))

f = max_depth -> length(build_tree(data, hyperparameters(0.1, max_depth, 1.0)).data)

# Compute the gradient with respect to bin_size
grad = gradient(f, 1.0)
println("gradient with respect to bin_size: ", grad[1])
