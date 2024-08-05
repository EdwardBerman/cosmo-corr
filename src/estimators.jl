module estimators

export landy_szalay_estimator, DD, DR, RR, interpolate_to_common_bins_spline

using Distributions

DD(c1,c2,c3,c4) = length(c1) 
RR(c1,c2,c3,c4) = length(c1) 

function DR(c1,c2,c3,c4)
    count = 0
    for i in 1:length(c1)
        if c1[i] == "DATA" && c2[i] == "RANDOM"
            count += 1
        end
    end
    return count
end

landy_szalay_estimator(DD, DR, RR) = (DD .- 2 .*DR .+ RR) ./ RR

function interpolate_to_common_bins_spline(corr_func, θ_common)
    θ_vals, values = corr_func[1,:], corr_func[2,:]
    spline_interpolator = CubicSplineInterpolation(θ_vals, values, extrapolation_bc=Line())
    return spline_interpolator(θ_common)
end

end
