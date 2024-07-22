module kdtree 

include("metrics.jl")
using .metrics

export KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_circles, populate!, insert!

using AbstractTrees

struct Galaxy_Circle{T, r, g}
    center::Vector{T}
    radius::r
    galaxies::Vector{g}
end

mutable struct KD_Galaxy_Tree
    root::Galaxy_Circle
    left::Union{KD_Galaxy_Tree, Nothing}
    right::Union{KD_Galaxy_Tree, Nothing}
end

function append_left!(tree::KD_Galaxy_Tree, node::Galaxy_Circle)
    if tree.root == KD_Galaxy_Tree(node, nothing, nothing)
        tree.root = node
    else
        append_left!(tree.root, node)
    end
end

function append_right!(tree::KD_Galaxy_Tree, node::Galaxy_Circle)
    if tree.root == nothing
        tree.root = KD_Galaxy_Tree(node, nothing, nothing)
    else
        append_right!(tree.root, node)
    end
end

function initialize_circles()
    return Galaxy_Circle{Float64, Float64, Galaxy}[], Galaxy_Circle{Float64, Float64, Galaxy}[]
end

function split_cirlces(galaxy_circles::Vector{Galaxy_Circle})
    return left_circles, right_circles
end

function populate!(tree::KD_Galaxy_Tree, galaxies::Vector{Galaxy})
    initial_circles = initialize_circles()
    for Galaxy_Circle in initial_circles 
        insert!(tree, Galaxy_Circle)
    end
end

function insert!(tree::KD_Galaxy_Tree, galaxy_circle::Galaxy_Circle)
    distance_matrix = build_distance_matrix(ra, dec, metric=sky_metric)
    distance_matrix = spacing.(distance_matrix)
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

end
