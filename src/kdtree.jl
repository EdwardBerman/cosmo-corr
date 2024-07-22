struct kdtree{T}
    root::Node{T}
end

function append_left!(tree::kdtree{T}, node::Node{T}) where T
    if tree.root == nothing
        tree.root = node
    else
        append_left!(tree.root, node)
    end
end

function append_right!(tree::kdtree{T}, node::Node{T}) where T
    if tree.root == nothing
        tree.root = node
    else
        append_right!(tree.root, node)
    end
end

function initialize_circles()
    return centers, radii
end

function populate!(tree::kdtree{T}, galaxies::Vector{T}) where T
    centers, radii = initialize_circles()
    while condition
        @threads for i in 1:size(galaxies, 1)
            node = Node{T}(galaxies[i, :])
            append_left!(tree, node)
        end
    end
end
