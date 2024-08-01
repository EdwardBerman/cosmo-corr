module estimators

export landy_szalay_estimator

using Distributions

landy_szalay_estimator(DD, DR, RR) = (DD - 2*DR + RR) / RR


end
