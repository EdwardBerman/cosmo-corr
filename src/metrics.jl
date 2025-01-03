#euclidean
module metrics

export build_distance_matrix, metric_dict, Vincenty_Formula, Vincenty, build_distance_matrix_subblock

using Base.Threads
using Distances


function Vincenty_Formula(coord1::Vector{Float64}, coord2::Vector{Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Vector{<:Real}, coord2::Vector{<:Real})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Vector{Any}, coord2::Vector{Any})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Tuple{Float64, Float64}, coord2::Tuple{Float64, Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60 
end


struct Vincenty <: SemiMetric end

function (d::Vincenty)(point1, point2)
    
    lat1, lon1 = point1 .*(π / 180.0)
    lat2, lon2 = point2 .*(π / 180.0)

    Δλ = abs(lon2 - lon1)

    c1 = (cos(lat2) * sin(Δλ))^2
    c2 = (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(Δλ))^2
    c3 = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(Δλ)

    y = sqrt(c1 + c2)
    x = c3

    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60 
end


metric_dict = Dict(
    "angular_separation" => Vincenty_Formula,
    "log" => log,
)

function build_distance_matrix(ra, dec; metric=Vincenty_Formula)
    n = length(ra)
    coords = [(ra[i], dec[i]) for i in 1:n]
    distance_matrix = zeros(n, n)
    @threads for i in 1:n
        for j in 1:i-1
            distance_matrix[i,j] = metric(coords[i], coords[j])
        end
    end
    return distance_matrix
end

function build_distance_matrix_subblock(galaxy_circles_a, galaxy_circles_b; metric=Vincenty_Formula)
    n = length(galaxy_circles_a)
    m = length(galaxy_circles_b)
    distance_matrix = Matrix{Any}(undef, n, m)
    for i in 1:n
        for j in 1:m
            coord1 = (galaxy_circles_a[i].center[1], galaxy_circles_a[i].center[2])
            coord2 = (galaxy_circles_b[j].center[1], galaxy_circles_b[j].center[2])
            distance_matrix[i,j] = (metric(coord1, coord2), galaxy_circles_a[i], galaxy_circles_b[j])
        end
    end
    return distance_matrix
end

end
