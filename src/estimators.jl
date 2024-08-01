module estimators

export landy_szalay_estimator

using Distributions


DD(c1,c2,c3,c4) = sum(c1 * c2') + sum(c3 * c4') / (length(c1 * c2') + length(c3 * c4'))
function corr_metric_default_point_point(c1,c2,c3,c4)
    DD = sum(c1 * c2') + sum(c3 * c4') / (length(c1 * c2') + length(c3 * c4'))
    number_random_galaxies = length(c1)



end
