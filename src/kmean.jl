module hcc

include("metrics.jl")
include("kdtree.jl")

using .metrics
using .kdtree

using Distances
using Clustering

function hcc(galaxies::Vector{Galaxy}, clusters::Int64, sky_metric=Vincenty_Formula, kmeans_metric=Vincenty, verbose=false)
    positions = hcat([galaxy.ra for galaxy in galaxies], [galaxy.dec for galaxy in galaxies])'
    result = kmeans(positions, clusters, distance=kmeans_metric)
    labels = result.assignments

    galaxy_clusters = Dict{Int, Vector{Tuple{Int, Galaxy}}}()
    for (i, cluster) in enumerate(labels)
        if !haskey(galaxy_clusters, cluster)
            galaxy_clusters[cluster] = []
        end
        push!(galaxy_clusters[cluster], (i, galaxies[i]))
    end
    
    galaxy_circles = []
    for (cluster, galaxy_info) in galaxy_clusters
        ra_list = [galaxy.ra for (index, galaxy) in galaxy_info]
        dec_list = [galaxy.dec for (index, galaxy) in galaxy_info]
        ra_circle = mean(ra_list)
        dec_circle = mean(dec_list)
        radius_circle = maximum([sky_metric([ra_circle, dec_circle], [galaxy.ra, galaxy.dec]) for (index, galaxy) in galaxy_info])
        galaxy_circle = Galaxy_Circle(
                                      [ra_circle, dec_circle],
                                      radius_circle,
                                      [galaxy for (index, galaxy) in galaxy_info],
                                      cluster,
                                      false
                                     )
        push!(galaxy_circles, galaxy_circle)
    end
    
    if nclusters(result) != clusters
        if verbose
            println("Warning: Number of clusters found is less than the number of clusters requested...")
        end
    end

    if verbose
        for (cluster, galaxy_info) in galaxy_clusters
            println("Cluster $cluster:")
            for (index, galaxy) in galaxy_info
                println("  Galaxy $index: ra=$(galaxy.ra), dec=$(galaxy.dec), quantity1=$(galaxy.corr1), quantity2=$(galaxy.corr2)")
            end
        end
    end

    if verbose
        for circle in galaxy_circles
            println("Cluster index $(circle.index):")
            println("  Center: $(circle.center)")
            println("  Radius: $(circle.radius)")
            println("  Galaxies: $(circle.galaxies)")
        end
    end
    return galaxy_circles
end


end

