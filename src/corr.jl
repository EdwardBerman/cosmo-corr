module astrocorr
    include("metrics.jl")
    include("kdtree.jl")
    include("kmean.jl")
    include("estimators.jl")

    using .metrics
    using .kdtree
    using .kmc
    using .estimators

    export corr, naivecorr, clustercorr, treecorr, corr_metric_default_point_point, corr_metric_default_position_position, Position_RA_DEC, landy_szalay_estimator, DD, DR, RR, interpolate_to_common_bins_spline, deg2rad_custom, Galaxy_Catalog, Galaxy, KD_Galaxy_Tree, Galaxy_Circle, append_left!, append_right!, initialize_circles, split_galaxy_cells!, populate, get_leaves, collect_leaves, kmeans_clustering, build_distance_matrix, metric_dict, Vincenty_Formula, Vincenty

    using LinearAlgebra
    using Base.Threads
    using Statistics
    using Clustering
    using Distances
    using DataFrames
    using Interpolations
    using Base.Iterators: product
    using Base.Iterators: partition
    using UnicodePlots

    struct Galaxy_Catalog
        ra::Vector{Float64}
        dec::Vector{Float64}
        corr1::Vector{Any}
        corr2::Vector{Any}
    end

    struct Position_RA_DEC
        ra::Float64
        dec::Float64
        value::String
        Position_RA_DEC(ra::Float64, dec::Float64, value::String) = begin
            if value == "DATA" || value == "RANDOM"
                new(ra, dec, value)
            else
                error("value must be either 'DATA' or 'RANDOM'")
            end
        end
    end

    function treecorr(ra, 
            dec, 
            corr1, 
            corr2, 
            θ_min, 
            number_bins, 
            θ_max; 
            cluster_factor=0.25, 
            spacing=log, 
            sky_metric=Vincenty_Formula, 
            kmeans_metric=Vincenty,
            corr_metric=corr_metric_default, 
            splitter=split_galaxy_cells!,
            verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        
        if verbose
            scatterplot_galaxies = scatterplot(ra, dec, title="Object Positions", xlabel="RA", ylabel="DEC")
            densityplot_galaxies = densityplot(ra, dec, title="Object Density", xlabel="RA", ylabel="DEC")
            println("Tree Correlation")
        end

        sky_metric = sky_metric
        galaxies = [Galaxy(ra[i], dec[i], corr1[i], corr2[i]) for i in 1:length(ra)]
        
        b = Δ_ln_d = (log(θ_max) - log(θ_min) )/ number_bins
        bin_size = b

        if verbose
            println("Bin size: ", bin_size)
            println("Populating KDTree")
        end

        tree = populate(galaxies, bin_size, sky_metric=sky_metric, splitter=splitter) # b = Δ (ln d) 
        
        if verbose
            println("Populated KDTree")
        end
        
        leafs = get_leaves(tree)
        ra_circles = [leaf.root.center[1] for leaf in leafs]
        dec_circles = [leaf.root.center[2] for leaf in leafs]
        n = length(ra_circles)

        if verbose
            scatterplot_clusters = scatterplot(ra_circles, dec_circles, title="Cluster Positions", xlabel="RA", ylabel="DEC")
            densityplot_clusters = densityplot(ra_circles, dec_circles, title="Cluster Density", xlabel="RA", ylabel="DEC")
            println("Number of circles: ", n)
            println("Computing Distance Matrix")
        end

        distance_matrix = build_distance_matrix(ra_circles, dec_circles, metric=sky_metric)

        if verbose
            println("Distance Matrix complete")
        end

        indices = [(i, j) for i in 1:n, j in 1:n if j < i]
        distance_vector = [distance_matrix[i, j] for (i, j) in indices]
        
        θ_bins = range(θ_min, stop=θ_max, length=number_bins)
        
        if spacing == log
            θ_bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
        end

        θ_bin_assignments = zeros(length(distance_vector))
        for i in 1:length(distance_vector)
            for j in 1:length(θ_bins)
                if distance_vector[i] < θ_bins[j] && distance_vector[i] > θ_min
                    θ_bin_assignments[i] = j
                    break
                end
            end
        end

        lock = ReentrantLock()
        df = DataFrame(bin_number=Int[], 
                       min_distance=Float64[], 
                       max_distance=Float64[], 
                       mean_distance=Float64[], 
                       ψ=Any[])
        
        if verbose
            println("Assigning data points to θ bins")
        end

        Threads.@threads for i in 1:number_bins
            bin = findall(θ_bin_assignments .== i)
            if !isempty(bin)
                min_distance = minimum(distance_vector[bin])
                max_distance = maximum(distance_vector[bin])
                mean_distance = mean(distance_vector[bin])
                bin_indices = [indices[k] for k in bin]

                corr1_values = []
                corr2_values = []

                corr1_reverse_values = []
                corr2_reverse_values = []

                for (i, j) in bin_indices
                    for galaxy_i in leafs[i].root.galaxies
                        for galaxy_j in leafs[j].root.galaxies
                            append!(corr1_values, [galaxies[galaxy_i].corr1])
                            append!(corr2_values, [galaxies[galaxy_j].corr2])
                            append!(corr1_reverse_values, [galaxies[galaxy_j].corr1])
                            append!(corr2_reverse_values, [galaxies[galaxy_i].corr2])
                        end
                    end
                end

                Threads.lock(lock) do
                    push!(df, (bin_number=i, 
                               min_distance=min_distance, 
                               max_distance=max_distance, 
                               mean_distance=mean_distance, 
                               ψ=corr_metric(corr1_values, corr2_values, corr1_reverse_values, corr2_reverse_values)))
                end
            end
        end
        
        if verbose
            println("Assigned data points to θ bins")
            println(df)
        end
        
        ψ_θ = zeros(2, number_bins) 
        @threads for i in 1:nrow(df)
            ψ_θ[1,i] = df[i, :mean_distance]
            ψ_θ[2,i] = df[i, :ψ]
            ψ_θ = ψ_θ[:, sortperm(ψ_θ[1,:])]
        end
        return ψ_θ
    end
    
    
    function clustercorr(ra,
            dec, 
            corr1, 
            corr2, 
            θ_min, 
            number_bins, 
            θ_max; 
            cluster_factor=0.25, 
            spacing=log, 
            sky_metric=Vincenty_Formula, 
            kmeans_metric=Vincenty, 
            corr_metric=corr_metric, 
            verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        
        if verbose
            scatterplot_galaxies = scatterplot(ra, dec, title="Object Positions", xlabel="RA", ylabel="DEC")
            densityplot_galaxies = densityplot(ra, dec, title="Object Density", xlabel="RA", ylabel="DEC")
            println("K means clustering")
        end
        
        sky_metric = sky_metric
        galaxies = [Galaxy(ra[i], dec[i], corr1[i], corr2[i]) for i in 1:length(ra)]
        clusters = round(Int, length(galaxies) * cluster_factor)
        
        if verbose
            println("Number of clusters: ", clusters)
            println("Clustering...")
        end
        
        circles = kmeans_clustering(galaxies, clusters, sky_metric=sky_metric, kmeans_metric=kmeans_metric, verbose=verbose)
        ra_circles = [circle.center[1] for circle in circles]
        dec_circles = [circle.center[2] for circle in circles]
        n = length(ra_circles)
        
        if verbose
            println("Clustering complete")
            println("Number of circles: ", n)
            scatterplot_clusters = scatterplot(ra_circles, dec_circles, title="Cluster Positions", xlabel="RA", ylabel="DEC")
            densityplot_clusters = densityplot(ra_circles, dec_circles, title="Cluster Density", xlabel="RA", ylabel="DEC")
            println("Computing Distance Matrix")
        end
        
        distance_matrix = build_distance_matrix(ra_circles, dec_circles, metric=sky_metric)
        
        if verbose
            println("Distance Matrix complete")
        end

        indices = [(i, j) for i in 1:n, j in 1:n if j < i]
        distance_vector = [distance_matrix[i, j] for (i, j) in indices]
        
        θ_bins = range(θ_min, stop=θ_max, length=number_bins)
        
        if spacing == log
            θ_bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
        end

        θ_bin_assignments = zeros(length(distance_vector))
        for i in 1:length(distance_vector)
            for j in 1:length(θ_bins)
                if distance_vector[i] < θ_bins[j] && distance_vector[i] > θ_min
                    θ_bin_assignments[i] = j
                    break
                end
            end
        end

        lock = ReentrantLock()
        df = DataFrame(bin_number=Int[], 
                       min_distance=Float64[], 
                       max_distance=Float64[], 
                       count=Int[], 
                       mean_distance=Float64[], 
                       corr1=Vector{Any}[], 
                       corr2=Vector{Any}[], 
                       corr1_reverse=Vector{Any}[], 
                       corr2_reverse=Vector{Any}[])
        
        if verbose
            println("Assigning data points to θ bins")
        end

        Threads.@threads for i in 1:number_bins
            bin = findall(θ_bin_assignments .== i)
            if !isempty(bin)
                min_distance = minimum(distance_vector[bin])
                max_distance = maximum(distance_vector[bin])
                mean_distance = mean(distance_vector[bin])
                bin_indices = [indices[k] for k in bin]

                corr1_values = []
                corr2_values = []

                corr1_reverse_values = []
                corr2_reverse_values = []
                
                for (i, j) in bin_indices
                    for galaxy_i in circles[i].galaxies
                        for galaxy_j in circles[j].galaxies
                            append!(corr1_values, [galaxies[galaxy_i].corr1])
                            append!(corr2_values, [galaxies[galaxy_j].corr2])
                            append!(corr1_reverse_values, [galaxies[galaxy_j].corr1])
                            append!(corr2_reverse_values, [galaxies[galaxy_i].corr2])
                        end
                    end
                end
                count = length(bin_indices)

                Threads.lock(lock) do
                    push!(df, (bin_number=i, 
                               min_distance=min_distance, 
                               max_distance=max_distance, 
                               count=count, 
                               mean_distance=mean_distance, 
                               corr1=corr1_values, 
                               corr2=corr2_values, 
                               corr1_reverse=corr1_reverse_values, 
                               corr2_reverse=corr2_reverse_values))
                end
            end
        end
        
        if verbose
            println("Assigned data points to θ bins")
            println(df)
        end
        
        ψ_θ = zeros(2, number_bins) 
        @threads for i in 1:nrow(df)
            ψ_θ[1,i] = df[i, :mean_distance]
            c1 = df[i, :corr1]
            c2 = df[i, :corr2]
            c3 = df[i, :corr1_reverse]
            c4 = df[i, :corr2_reverse]
            ψ_θ[2,i] = corr_metric(c1, c2, c3, c4)
            ψ_θ = ψ_θ[:, sortperm(ψ_θ[1,:])]
        end
        return ψ_θ
    end

    function naivecorr(ra, 
            dec, 
            corr1, 
            corr2, 
            θ_min, 
            number_bins, 
            θ_max; 
            cluster_factor=0.25, 
            spacing=log, 
            sky_metric=Vincenty_Formula, 
            kmeans_metric=Vincenty,
            corr_metric=corr_metric, 
            verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        if verbose
            scatterplot_galaxies = scatterplot(ra, dec, title="Object Positions", xlabel="RA", ylabel="DEC")
            densityplot_galaxies = densityplot(ra, dec, title="Object Density", xlabel="RA", ylabel="DEC")
            println("Naive Correlation")
            println("Computing Distance Matrix")
        end
        distance_matrix = build_distance_matrix(ra, dec, metric=sky_metric)

        n = length(ra)
        indices = [(i, j) for i in 1:n, j in 1:n if j < i]  
        distance_vector = [distance_matrix[i, j] for (i, j) in indices]
        
        θ_bins = range(θ_min, stop=θ_max, length=number_bins)
        
        if spacing == log
            θ_bins = 10 .^ range(log10(θ_min), log10(θ_max), length=number_bins)
        end

        θ_bin_assignments = zeros(length(distance_vector))
        for i in 1:length(distance_vector)
            for j in 1:length(θ_bins)
                if distance_vector[i] < θ_bins[j]
                    θ_bin_assignments[i] = j
                    break
                end
            end
        end

        lock = ReentrantLock()
        df = DataFrame(bin_number=Int[], 
                       min_distance=Float64[], 
                       max_distance=Float64[], 
                       count=Int[], 
                       mean_distance=Float64[], 
                       corr1=Vector{Any}[], 
                       corr2=Vector{Any}[], 
                       corr1_reverse=Vector{Any}[], 
                       corr2_reverse=Vector{Any}[])
        
        if verbose
            println("Assigning data points to θ bins")
        end

        Threads.@threads for i in 1:number_bins
            bin = findall(θ_bin_assignments .== i)
            if !isempty(bin)
                min_distance = minimum(distance_vector[bin])
                max_distance = maximum(distance_vector[bin])
                mean_distance = mean(distance_vector[bin])
                bin_indices = [indices[k] for k in bin]
                corr1_values, corr2_values = [], []
                corr1_reverse_values, corr2_reverse_values = [], []

                for (i, j) in bin_indices
                    append!(corr1_values, corr1[i])
                    append!(corr2_values, corr2[j])
                    append!(corr1_reverse_values, corr1[j])
                    append!(corr2_reverse_values, corr2[i])
                end

                count = length(bin_indices)

                Threads.lock(lock) do
                    push!(df, (bin_number=i, 
                               min_distance=min_distance, 
                               max_distance=max_distance, 
                               count=count, 
                               mean_distance=mean_distance, 
                               corr1=corr1_values, 
                               corr2=corr2_values, 
                               corr1_reverse=corr1_reverse_values, 
                               corr2_reverse=corr2_reverse_values))
                end
            end
        end

        if verbose
            println("Assigned data points to θ bins")
            println(df)
        end
        
        ψ_θ = zeros(2, number_bins) 
        @threads for i in 1:nrow(df)
            ψ_θ[1,i] = df[i, :mean_distance]
            c1 = df[i, :corr1]
            c2 = df[i, :corr2]
            c3 = df[i, :corr1_reverse]
            c4 = df[i, :corr2_reverse]
            ψ_θ[2,i] = corr_metric(c1, c2, c3, c4)
            ψ_θ = ψ_θ[:, sortperm(ψ_θ[1,:])]
        end
        return ψ_θ
    end

    corr_metric_default_point_point(c1,c2,c3,c4) = (sum(c1 .* c2) + sum(c3 .* c4)) / (length(c1) + length(c3))
    corr_metric_default_position_position(c1,c2,c3,c4) = length(c1)

    function corr(ra::Vector{Float64},
            dec::Vector{Float64}, 
            x::Vector{Vector{Float64}}, 
            y::Vector{Float64}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            kmeans_metric=Vincenty,
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, 
                          dec, 
                          x, 
                          y, 
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric, 
                          kmeans_metric=kmeans_metric, 
                          corr_metric=corr_metric, 
                          verbose=verbose)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Float64}, 
            y::Vector{Vector{Float64}}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            kmeans_metric=Vincenty,
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, 
                          dec, 
                          x, 
                          y, 
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric, 
                          kmeans_metric=kmeans_metric, 
                          corr_metric=corr_metric, 
                          verbose=verbose)
    end

    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Vector{Complex{Float64}}}, 
            y::Vector{Float64}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            kmeans_metric=Vincenty, 
            corr_metric=corr_metric_default,
            correlator=treecorr, 
            verbose=false)
        return correlator(ra, 
                          dec, 
                          x, 
                          y, 
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric,
                          kmeans_metric=kmeans_metric,
                          corr_metric=corr_metric,
                          verbose=verbose)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Float64}, 
            y::Vector{Vector{Float64}}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            kmeans_metric=Vincenty,
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, 
                          dec, 
                          x, 
                          y, 
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric,
                          kmeans_metric=kmeans_metric,
                          corr_metric=corr_metric, 
                          verbose=verbose)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Float64}, 
            y::Vector{Float64}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            kmeans_metric=Vincenty,
            corr_metric=corr_metric_default_point_point,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, 
                          dec, 
                          x, 
                          y, 
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric,
                          kmeans_metric=kmeans_metric,
                          corr_metric=corr_metric, 
                          verbose=verbose)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Position_RA_DEC},
            y::Vector{Position_RA_DEC},
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula,
            kmeans_metric=Vincenty,
            DD=DD,
            DR=DR,
            RR=RR,
            estimator=landy_szalay_estimator,
            correlator=treecorr,
            splitter=split_galaxy_cells!,
            verbose=false)
        DD_cat = Galaxy_Catalog([pos.ra for pos in x if pos.value == "DATA"], 
                                [pos.dec for pos in x if pos.value == "DATA"], 
                                [pos.value for pos in x if pos.value == "DATA"],
                                [pos.value for pos in x if pos.value == "DATA"])
        DR_cat = Galaxy_Catalog([pos.ra for pos in x],
                                [pos.dec for pos in x],
                                [pos.value for pos in x],
                                [pos.value for pos in x])
        RR_cat = Galaxy_Catalog([pos.ra for pos in x if pos.value == "RANDOM"], 
                                [pos.dec for pos in x if pos.value == "RANDOM"], 
                                [pos.value for pos in x if pos.value == "RANDOM"],
                                [pos.value for pos in x if pos.value == "RANDOM"])

        θ_common = 10 .^(range(log10(θ_min), log10(θ_max), length=number_bins))
        
        if verbose
            println("Computing DD")
        end
        
        DD_θ = correlator(DD_cat.ra, 
                          DD_cat.dec, 
                          DD_cat.corr1,
                          DD_cat.corr2,
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric,
                          kmeans_metric=kmeans_metric,
                          corr_metric=DD, 
                          splitter=splitter,
                          verbose=verbose)
        
        if verbose
            println("DD complete")
            println("Computing DR")
        end
        
        DR_θ = correlator(DR_cat.ra,
                          DR_cat.dec,
                          DR_cat.corr1,
                          DR_cat.corr2,
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric, 
                          kmeans_metric=kmeans_metric,
                          corr_metric=DR, 
                          splitter=splitter,
                          verbose=verbose)
        
        if verbose
            println("DR complete")
            println("Computing RR")
        end
        
        RR_θ = correlator(RR_cat.ra,
                          RR_cat.dec,
                          RR_cat.corr1,
                          RR_cat.corr2,
                          θ_min, 
                          number_bins, 
                          θ_max, 
                          cluster_factor=cluster_factor, 
                          spacing=spacing, 
                          sky_metric=sky_metric,
                          kmeans_metric=kmeans_metric,
                          corr_metric=RR, 
                          splitter=splitter,
                          verbose=verbose)
        
        if verbose
            println("RR complete")
        end

        DD_interp = interpolate_to_common_bins_spline(DD_θ, θ_common)
        DR_interp = interpolate_to_common_bins_spline(DR_θ, θ_common)
        RR_interp = interpolate_to_common_bins_spline(RR_θ, θ_common)
        if verbose
            println("Interpolated to common bins")
            plt = lineplot(log10.(θ_common), log10.(estimator(DD_interp, DR_interp, RR_interp)), title="Correlation Function", name="Correlation Function", xlabel="log10(θ)", ylabel="log10(ξ(θ))")
            lineplot!(plt, log10.(θ_common), log10.(DD_interp), color=:red, name="DD")
            lineplot!(plt, log10.(θ_common), log10.(DR_interp), color=:yellow, name="DR")
            lineplot!(plt, log10.(θ_common), log10.(RR_interp), color=:cyan, name="RR")
        end
        return estimator(DD_interp, DR_interp, RR_interp)
    end
    #=
    Rest of corr functions here, multiple dispatch!
    =#
    #=
    #example
    #gal_cal = galaxy_catalog([1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5])
    #cor(θ) = corr(gal_cal.ra, gal_cal.dec, gal_cal.corr1, gal_cal.corr2)
    =#

end
