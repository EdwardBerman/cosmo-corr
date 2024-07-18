module astro
    export corr
    using LinearAlgebra
    using Statistics
    using Base.Iterators: product
    using Base.Iterators: partition

    include("metrics.jl")
    # Vector of vectors, vector of floats,
    
    function treecorr()
        #initialize partition sizes and positions of 2 cells
        return 0
    end

    function corr(x::Vector{Vector{Float64}}, y::Vector{Float64} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(x::Vector{Float64}, y::Vector{Vector{Float64}} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end

    function corr(x::Vector{Vector{Complex{Float64}}}, y::Vector{Float64} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end
    
    function corr(x::Vector{Float64}, y::Vector{Vector{Float64}} ;spacing=log, metric=euclidean, correlator=treecorr)
        return correlator(x, y)
    end
    #=
    Rest of corr functions here, multiple dispatch!
    =#



end
