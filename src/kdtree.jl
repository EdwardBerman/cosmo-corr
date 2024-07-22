using AbstractTrees

mutable struct KD_Galaxy_Tree
    root::Galaxy
    left::Union{KD_Galaxy_Tree, Nothing}
    right::Union{KD_Galaxy_Tree, Nothing}
end

mutable struct Galaxy_Circle{T, r, g}
    center::Vector{T}
    radius::r
    galaxies::Vector{g}
end

function append_left!(tree::KD_Galaxy_Tree, node::Galaxy)
    if tree.root == KD_Galaxy_Tree(node, nothing, nothing)
        tree.root = node
    else
        append_left!(tree.root, node)
    end
end

function append_right!(tree::KD_Galaxy_Tree, node::Galaxy)
    if tree.root == nothing
        tree.root = KD_Galaxy_Tree(node, nothing, nothing)
    else
        append_right!(tree.root, node)
    end
end

function initialize_circles()
    return centers, radii
end

function split_cirlces(galaxy_circles::Vector{galaxy_circle})
    return left_circles, right_circles
end

function populate!(tree::KD_Galaxy_Tree, galaxies::Vector{Galaxy})
    centers, radii = initialize_circles()
    for galaxy in galaxies
        insert!(tree, galaxy)
    end
end

function insert!(tree::KD_Galaxy_Tree, galaxy::Galaxy)
    if galaxy.ra < tree.root.ra
        if tree.left == nothing
            tree.left = KD_Galaxy_Tree(galaxy, nothing, nothing)
        else
            insert!(tree.left, galaxy)
        end
    else
        if tree.right == nothing
            tree.right = KD_Galaxy_Tree(galaxy, nothing, nothing)
        else
            insert!(tree.right, galaxy)
        end
    end
end
