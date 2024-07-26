#euclidean
module metrics
export build_distance_matrix, metric_dict

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

metric_dict = Dict(
    "euclidean" => Euclidean(),
    "log" => Log(),
)

end
