#euclidean
module metrics
export build_distance_matrix, metric_dict, Vincenty_Formula

using Base.Threads


function build_distance_matrix(ra, dec; metric=Euclidean())
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

function Vincenty_Formula(coord1::Vector{Float64}, coord2::Vector{Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    Δλ = abs(λ2 - λ1)
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    numerator = sqrt(c1 + c2)
    denominator = c3
    Δσ = atan(numerator, denominator)
    return Δσ
end

metric_dict = Dict(
    "angular_separation" => Vincenty_Formula(),
    "log" => Log(),
)

# go from radians to arcmins?

end
