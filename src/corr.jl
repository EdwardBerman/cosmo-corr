module astrocorr
    include("metrics.jl")
    include("kdtree.jl")
    include("hcc.jl")

    using .metrics
    using .kdtree
    using .hcc

    export corr
    using LinearAlgebra
    using Base.Threads
    using Statistics
    using Clustering
    using Distances
    using DataFrames
    using Base.Iterators: product
    using Base.Iterators: partition

    struct Galaxy_Catalog
        ra::Vector{Float64}
        dec::Vector{Float64}
        corr1::Vector{Any}
        corr2::Vector{Any}
    end

    struct Position
        ra::Float64
        dec::Float64
    end

    function treecorr(ra, dec, corr1, corr2, θ_min, number_bins, θ_max; cluster_factor=0.25, spacing=log, sky_metric=Vincenty_Formula, corr_metric=corr_metric_default, verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        sky_metric = sky_metric
        galaxies = [Galaxy(ra[i], dec[i], corr1[i], corr2[i]) for i in 1:length(ra)]
        
        b = Δ_ln_d = (log(θ_max) - log(θ_min) )/ number_bins
        bin_size = b

        tree = populate(galaxies, bin_size, metric=sky_metric) # b = Δ (ln d) 
        leafs = get_leaves(tree)
        ra_circles = [leaf.root.center[1] for leaf in leafs]
        dec_circles = [leaf.root.center[2] for leaf in leafs]
        distance_matrix = build_distance_matrix(ra_circles, dec_circles, metric=sky_metric)

        n = length(ra_circles)
        if verbose
            println("Number of circles: ", n)
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
                    append!(corr1_values, [galaxies[k].corr1 for k in leafs[i].root.galaxies])
                    append!(corr2_values, [galaxies[k].corr2 for k in leafs[j].root.galaxies])
                    append!(corr1_reverse_values, [galaxies[k].corr1 for k in leafs[j].root.galaxies])
                    append!(corr2_reverse_values, [galaxies[k].corr2 for k in leafs[i].root.galaxies])
                end
                count = length(cor1_values)

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
    
    
    function clustercorr(ra, dec, corr1, corr2, θ_min, number_bins, θ_max; cluster_factor=0.25, spacing=log, sky_metric=Vincenty_Formula, corr_metric=corr_metric, verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        sky_metric = sky_metric
        galaxies = [Galaxy(ra[i], dec[i], corr1[i], corr2[i]) for i in 1:length(ra)]
        clusters = round(Int, length(galaxies) * cluster_factor)
        circles = hcc(galaxies, clusters, sky_metric=sky_metric, verbose=verbose)
        ra_circles = [circle.center[1] for circle in circles]
        dec_circles = [circle.center[2] for circle in circles]
        distance_matrix = build_distance_matrix(ra_circles, dec_circles, metric=sky_metric)

        n = length(ra_circles)
        if verbose
            println("Number of circles: ", n)
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
                    append!(corr1_values, [galaxies[k].corr1 for k in leafs[i].root.galaxies])
                    append!(corr2_values, [galaxies[k].corr2 for k in leafs[j].root.galaxies])
                    append!(corr1_reverse_values, [galaxies[k].corr1 for k in leafs[j].root.galaxies])
                    append!(corr2_reverse_values, [galaxies[k].corr2 for k in leafs[i].root.galaxies])
                end
                count = length(cor1_values)

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

    function naivecorr(ra, dec, corr1, corr2, θ_min, number_bins, θ_max; cluster_factor=0.25, spacing=log, sky_metric=Vincenty_Formula, corr_metric=corr_metric, verbose=false)
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
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
        
        Threads.@threads for i in 1:number_bins
            bin = findall(θ_bin_assignments .== i)
            if !isempty(bin)
                min_distance = minimum(distance_vector[bin])
                max_distance = maximum(distance_vector[bin])
                mean_distance = mean(distance_vector[bin])
                bin_indices = [indices[k] for k in bin]
                corr1_values = [corr1[i] for (i, j) in bin_indices]
                corr2_values = [corr2[j] for (i, j) in bin_indices]
                corr1_reverse_values = [corr1[j] for (i, j) in bin_indices]
                corr2_reverse_values = [corr2[i] for (i, j) in bin_indices]
                count = length(cor1_values)

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

    corr_metric_default_point_point(c1,c2,c3,c4) = sum(c1 * c2') + sum(c3 * c4') / (length(c1 * c2') + length(c3 * c4'))
    corr_metric_default_position_position(c1,c2,c3,c4) = landy_szalay_estimator(DD(c1, c2, c3, c4), DR(c1, c2, c3, c4), RR(c1, c2, c3, c4))

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
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
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
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
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
            corr_metric=corr_metric_default,
            correlator=treecorr, 
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
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
            corr_metric=corr_metric_default,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
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
            corr_metric=corr_metric_default_point_point,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Position},
            y::Vector{Position},
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            cluster_factor=0.25,
            spacing=log, 
            sky_metric=Vincenty_Formula(),
            corr_metric=corr_metric_default_position_position,
            correlator=treecorr,
            verbose=false)
        return correlator(ra, dec, x, y, θ_min, number_bins, θ_max, cluster_factor=cluster_factor, spacing=spacing, sky_metric=sky_metric, corr_metric=corr_metric, verbose=true)
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
