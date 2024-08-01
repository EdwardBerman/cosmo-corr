module hcc

include("metrics.jl")
include("kdtree.jl")

using .metrics
using .kdtree

using Distances
using Clustering

function hcc(galaxies::Vector{Galaxy}, clusters::Int64, sky_metric=Vincenty_Formula, verbose=false)
    distance_matrix = pairwise(sky_metric, [galaxy.ra for galaxy in galaxies], [galaxy.dec for galaxy in galaxies])
    hc = hclust(distance_matrix)
    clusters = cutree(hc, k=clusters)

    galaxy_clusters = Dict{Int, Vector{Tuple{Int, Galaxy}}}()
    for (i, cluster) in enumerate(clusters)
        if !haskey(galaxy_clusters, cluster)
            galaxy_clusters[cluster] = []
        end
        push!(galaxy_clusters[cluster], (i, galaxies[i]))
    end

    if verbose
        for (cluster, galaxy_info) in galaxy_clusters
            println("Cluster $cluster:")
            for (index, galaxy) in galaxy_info
                println("  Galaxy $index: ra=$(galaxy.ra), dec=$(galaxy.dec), quantity1=$(galaxy.quantity1), quantity2=$(galaxy.quantity2)")
            end
        end
    end

    galaxy_circles = []
    for (cluster, galaxy_info) in galaxy_clusters
        ra_list = [galaxy.ra for (index, galaxy) in galaxy_info]
        dec_list = [galaxy.dec for (index, galaxy) in galaxy_info]
        ra_circle = mean(ra_list)
        dec_circle = mean(dec_list)
        radius_circle = maximum([sky_metric([initial_ra, initial_dec], [galaxy.ra, galaxy.dec]) for (index, galaxy) in galaxy_info])
        galaxy_circle = Galaxy_Circle(
                                      [initial_ra, initial_dec],
                                      initial_radius,
                                      [galaxy for (index, galaxy) in galaxy_info],
                                      cluster,
                                      false
                                     )
        push!(galaxy_circles, initial_circle)
    end

    for circle in galaxy_circles
        println("Cluster index $(circle.index):")
        println("  Center: $(circle.center)")
        println("  Radius: $(circle.radius)")
        println("  Galaxies: $(circle.galaxies)")
    end
    return galaxy_circles
end


end

