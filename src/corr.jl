module astrocorr
    export corr
    using LinearAlgebra
    using Base.Threads
    using Statistics
    using Clustering
    using Distances
    using DataFrames
    using Base.Iterators: product
    using Base.Iterators: partition

    include("metrics.jl")
    # Vector of vectors, vector of floats,
    
    struct galaxy_catalog
        ra::Vector{Float64}
        dec::Vector{Float64}
        corr1::Any
        corr2::Any
    end

    #Heirarchical Clustering from Julia
    
    function treecorr()
        #initialize partition sizes and positions of 2 cells
        # Naively separate into line above some y axis and below some y axis, if the split is lopsided, rotate the axis
        return 0
    end
    
    
    #Advantage of cluster corr is that you can specify the exact number of bins you want, where as in treecorr, you can't. Bin at granularity that loses info. The disadvantage is that you don't have a bound on the binning error.
    function clustercorr(ra, dec, corr1, corr2; spacing=log, metric=Euclidean())
        distance_matrix = build_distance_matrix(x, y, metric=metric)
        distance_matrix = spacing.(distance_matrix)
        distance_vector = reshape(distance_matrix, :, 1) # probably don't do this actually. 2 x ? matrix
        hclusters = hclust(distance_matrix)
        θ_bin_assigments = cutree(hclusters, k=number_bins)
        #df = |min theta in bin _i | max theta in bin _i | index of bin _i | number of points in bin _i
        df = DataFrame()
        for i in 1:number_bins
            bin = findall(θ_bin_assigments .== i)
            push!(df, [minimum(distance_vector[bin]), maximum(distance_vector[bin]), i, length(bin)])
        end
        if verbose
            println(df)
        end
    end

    function naivecorr(ra, dec, corr1, corr2, θ_min, number_bins, θ_max; spacing=log, metric=Euclidean())
        @assert length(ra) == length(dec) == length(corr1) == length(corr2) "ra, dec, corr1, and corr2 must be the same length"
        distance_matrix = build_distance_matrix(ra, dec, metric=metric)
        distance_matrix = spacing.(distance_matrix)

        n = length(ra)
        indices = [(i, j) for i in 1:n, j in 1:n if j < i]  
        distance_vector = [distance_matrix[i, j] for (i, j) in indices]
        distance_vector = filter(!=(0), distance_vector)
        
        θ_bins = range(θ_min, stop=θ_max, length=number_bins)

        θ_bin_assignments = zeros(length(distance_vector))
        for i in 1:length(distance_vector)
            for j in 1:length(θ_bins)
                if distance_vector[i] < θ_bins[j]
                    θ_bin_assignments[i] = j
                    break
                end
            end
        end

        df = DataFrame(bin_number=Int[], min_distance=Float64[], max_distance=Float64[], count=Int[], mean_distance=Float64[], corr1=Vector{Float64}[], corr2=Vector{Float64}[])
        
        for i in 1:number_bins
            bin = findall(θ_bin_assignments .== i)
            if !isempty(bin)
                min_distance = minimum(distance_vector[bin])
                max_distance = maximum(distance_vector[bin])
                count = length(bin)
                mean_distance = mean(distance_vector[bin])
                bin_indices = [indices[k] for k in bin]
                corr1_values = [corr1[i] for (i, j) in bin_indices]
                corr2_values = [corr2[j] for (i, j) in bin_indices]

                push!(df, (bin_number=i, min_distance=min_distance, max_distance=max_distance, count=count, mean_distance=mean_distance, corr1=corr1_values, corr2=corr2_values))
            end
        end

        if verbose
            println(df)
        end
        
        ψ_θ = zeros(2, number_bins) 
        @threads for i in 1:nrow(df)
            ψ_θ[1,i] = df[i, :mean_distance]
            ψ_θ[2,i] = mean(metric(df[i, :corr1], df[i, :corr2]))
        end

        return ψ_θ
    end
    # parallelize ξ (θ) for different θ bins 

    #=
    Correlators: TreeCorr, Heirarchical Clustering, etc.
    =#

    function corr(ra::Vector{Float64},
            dec::Vector{Float64}, 
            x::Vector{Vector{Float64}}, 
            y::Vector{Float64}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            spacing=log, 
            metric=Euclidean(), 
            correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Float64}, 
            y::Vector{Vector{Float64}}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            spacing=log, 
            metric=Euclidean(), 
            correlator=treecorr)
        return correlator(x, y)
    end

    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Vector{Complex{Float64}}}, 
            y::Vector{Float64}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            spacing=log, 
            metric=Euclidean(), 
            correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, 
            dec::Vector{Float64}, 
            x::Vector{Float64}, 
            y::Vector{Vector{Float64}}, 
            θ_min::Float64, 
            number_bins::Int64, 
            θ_max::Float64; 
            spacing=log, 
            metric=Euclidean(), 
            correlator=treecorr)
        return correlator(x, y)
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
