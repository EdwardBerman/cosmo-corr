module astro
    export corr
    using LinearAlgebra
    using Statistics
    using Clustering
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
    function clustercorr(x, y; spacing=log, metric=euclidean)
        distance_matrix = build_distance_matrix(x, y, metric=metric)
        # log scale all entries
        hclusters = hclust(distance_matrix)
    end

    function naivecorr(x, y; metric=euclidean)
        return mean([metric(x[i], y[j]) for i in 1:length(x), j in 1:length(y)])
    end

    #=
    Correlators: TreeCorr, Heirarchical Clustering, etc.
    =#

    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Vector{Float64}}, y::Vector{Float64} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Float64}, y::Vector{Vector{Float64}} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end

    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Vector{Complex{Float64}}}, y::Vector{Float64} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(ra::Vector{Float64}, dec::Vector{Float64}, x::Vector{Float64}, y::Vector{Vector{Float64}} ;spacing=log, metric=euclidean, correlator=treecorr)
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
