using Zygote

# Define an immutable struct for the tree nodes
struct TreeNode
    left::Union{Nothing, TreeNode}
    right::Union{Nothing, TreeNode}
    data::Float64
end

# Recursive function to build the tree
function build_tree(data::Vector{Float64}, depth::Int=0)::TreeNode
    # Base case: if the data is empty, return a leaf node
    if length(data) == 1
        return TreeNode(nothing, nothing, data[1])
    end
    
    # Determine the splitting point
    mid = length(data) ÷ 2
    
    # Recursively build the left and right subtrees
    left_subtree = build_tree(data[1:mid], depth + 1)
    right_subtree = build_tree(data[mid+1:end], depth + 1)
    
    # Return the current node with left and right children
    return TreeNode(left_subtree, right_subtree, data[mid])
end

# Example data
data = sort(rand(10))  # Sorted data for tree building

# Build the tree
tree = build_tree(data)

# Test differentiability with Zygote
gradient(x -> build_tree(x).data, data)

data = [0.3, 0.1, 0.2, 0.5, 0.4]
println("Data: ", data)
gradient(x -> build_tree(x).data, data)
println("Gradient: ", gradient(x -> build_tree(x).data, data))
