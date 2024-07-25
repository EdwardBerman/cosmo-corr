module kdtree 

include("metrics.jl")
using .metrics

export Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_circles, populate!, insert!

using AbstractTrees

struct Galaxy 
    ra::Float64
    dec::Float64
    corr1::Any
    corr2::Any
end

struct Galaxy_Circle{T, r, g}
    center::Vector{T}
    radius::r
    galaxies::Vector{g}
    index::Int
    split::Bool
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

function initialize_circles(galaxies::Vector{Galaxy})
    # root is a circle with all galaxies, then split into left and right groups
    # divide either RA or DEC into 2 groups, then enclose them in non overlapping circles
    ra_list = [galaxy.ra for galaxy in galaxies]
    dec_list = [galaxy.dec for galaxy in galaxies]
    # return a vector of galaxy circles 
    return Galaxy_Circle{Vector{Float64}, Float64, Galaxy, 1, false}[], Galaxy_Circle{Vector{Float64}, Float64, Galaxy, 2, false}[]
end

function split_cirlces!(tree::KD_Galaxy_Tree, galaxy_circles::Vector{Galaxy_Circle})
    circle_ra = [circle.center[1] for circle in galaxy_circles]
    circle_dec = [circle.center[2] for circle in galaxy_circles]
    distance_matrix = build_distance_matrix(circle_ra, circle_dec, metric=euclidean_metric) # place holder metric
    distance_matrix = spacing.(distance_matrix)
    for i in 1:length(galaxy_circles)
        for j in 1:length(galaxy_circles)
            if (galaxy_circles[i].radius + galaxy_circles[j].radius)/ distance_matrix[i, j] < b # b = Δ ln d
                galaxy_circles[i].split = true
                galaxy_circles[j].split = true
                # split the circle, append left and right to tree
            end
        end
    end
end

function populate!(tree::KD_Galaxy_Tree, galaxies::Vector{Galaxy})
    initial_circles = initialize_circles(galaxies)
    i = 3
    for Galaxy_Circle in initial_circles 
        insert!(tree, Galaxy_Circle)
    end
    count = 0
    while # splitting condition (function to check the splitting condition)
        if count == 0
            split_circles!(tree, initial_circles)
            count += 1
        else 
            circles_to_split = # get circles to split
            split_circles!(tree, initial_circles)
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
