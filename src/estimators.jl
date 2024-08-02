module estimators

export landy_szalay_estimator, DD, DR, RR

using Distributions

DD(c1,c2,c3,c4) = length(c1) 
RR(c1,c2,c3,c4) = length(c1)
DR(c1,c2,c3,c4) = length(c1)

landy_szalay_estimator(DD, DR, RR) = (DD - 2*DR + RR) / RR


end
