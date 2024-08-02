module estimators

export landy_szalay_estimator, DD, DR, RR

using Distributions

DD(c1,c2,c3,c4) = length(c1) 
RR(c1,c2,c3,c4) = length(c1) 

function DR(c1,c2,c3,c4)
    count = 0
    for a in c1
        for b in c2
            if a.value == "DATA" && b.value == "RANDOM"
                count += 1
            end
        end
    end
    return count
end

landy_szalay_estimator(DD, DR, RR) = (DD - 2*DR + RR) / RR


end
