#euclidean


function build_distance_matrix(x,y, metric=euclidean)
    distance_matrix = zeros(length(x), length(y))
    for i in 1:length(x)
        for j in 1:length(y)
            distance_matrix[i,j] = metric(x[i], y[j])
        end
    end
    return distance_matrix
end

