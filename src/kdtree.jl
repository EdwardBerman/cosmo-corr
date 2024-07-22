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
