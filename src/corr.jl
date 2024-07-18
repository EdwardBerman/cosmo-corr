module astro
    export corr
    using LinearAlgebra
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
    function clustercorr(x, y; spacing=log, metric=Euclidean())
        distance_matrix = build_distance_matrix(x, y, metric=metric)
        distance_matrix = spacing.(distance_matrix)
        distance_vector = reshape(distance_matrix, :, 1)
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

    function naivecorr(x, y; metric=euclidean)
        return mean([metric(x[i], y[j]) for i in 1:length(x), j in 1:length(y)])
    end

    #=
    Correlators: TreeCorr, Heirarchical Clustering, etc.
    =#

    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Vector{Float64}}, y::Vector{Float64}, number_bins::Int64; spacing=log, metric=Euclidean(), correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Float64}, y::Vector{Vector{Float64}}, number_bins::Int64 ;spacing=log, metric=Euclidean(), correlator=treecorr)
        return correlator(x, y)
    end

    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Vector{Complex{Float64}}}, y::Vector{Float64}, number_bins::Int64 ;spacing=log, metric=Euclidean(), correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Float64}, y::Vector{Vector{Float64}}, number_bins::Int64 ;spacing=log, metric=Euclidean(), correlator=treecorr)
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
