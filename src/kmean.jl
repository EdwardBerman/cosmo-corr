module hcc

include("metrics.jl")
include("kdtree.jl")

using .metrics
using .kdtree

using Distances
using Clustering

function hcc(galaxies::Vector{Galaxy}, clusters::Int64, sky_metric=Vincenty_Formula, verbose=false)
    positions = hcat([galaxy.ra for galaxy in galaxies], [galaxy.dec for galaxy in galaxies])
    result = kmeans(positions, clusters; maxiter=300, tol=1e-6)
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
            println("Warning: Number of clusters found is different from the number of clusters requested. Using Heirarchical Clustering to merge.")
        end
        distance_matrix = build_distance_matrix([galaxy_circle.ra for galaxy_circle in galaxy_circles], [galaxy_circle.dec for galaxy_circle in galaxy_circles], sky_metric=sky_metric)
        distance_matrix = distance_matrix + distance_matrix'
        hc = hclust(distance_matrix)
        cluster_labels = cutree(hc, k=clusters)

        galaxy_clusters = Dict{Int, Vector{Tuple{Int, Galaxy}}}()
        for (i, galaxy_circle) in enumerate(galaxy_circles)
            cluster = cluster_labels[i]
            if !haskey(galaxy_clusters, cluster)
                galaxy_clusters[cluster] = []
            end
            for galaxy in galaxy_circle.galaxies
                append!(galaxy_clusters[cluster], galaxy)
            end
        end

        galaxy_circles = []
        for (cluster, galaxies_in_cluster) in galaxy_clusters
            ra_list = [galaxy.ra for galaxy in galaxies_in_cluster]
            dec_list = [galaxy.dec for galaxy in galaxies_in_cluster]
            ra_circle = mean(ra_list)
            dec_circle = mean(dec_list)
            radius_circle = maximum([sky_metric([ra_circle, dec_circle], [galaxy.ra, galaxy.dec]) for galaxy in galaxies_in_cluster])
            galaxy_circle = Galaxy_Circle([ra_circle, dec_circle],
                                          radius_circle,
                                          galaxies_in_cluster,
                                          cluster,
                                          false
                                         )
            push!(galaxy_circles, galaxy_circle)
        end
        
        for (i, cluster) in enumerate(clusters)
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

