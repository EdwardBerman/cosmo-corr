#euclidean


function build_distance_matrix(ra, dec, metric=Euclidean())
    distance_matrix = zeros(length(ra), length(dec))
    @threads for i in 1:length(ra)
        for j in 1:length(dec)
            if j ≤ i
                distance_matrix[i,j] = metric(ra[i], dec[j])
            end
        end
    end
    return distance_matrix
end

