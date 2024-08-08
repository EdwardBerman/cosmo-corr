include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
using UnicodePlots
using Base.Threads
using Statistics
using Distributions

println(nthreads())

fits = pyimport("astropy.io.fits")
f = fits.open("Aardvark.fit")
print(f[2].data)

ra = f[2].data["RA"]
dec = f[2].data["DEC"]
ra = convert(Vector{Float64}, ra) 
dec = convert(Vector{Float64}, dec) 
positions = [Position_RA_DEC(ra, dec, "DATA") for (ra, dec) in zip(ra, dec)]

ra = ra .* π / 180
dec = dec .* π / 180

ra_min = minimum(ra)
ra_max = maximum(ra)
dec_min = minimum(dec)
dec_max = maximum(dec)

rand_ra = rand(Uniform(ra_min, ra_max), length(ra)) 
rand_sin_dec = rand(Uniform(sin(dec_min), sin(dec_max)), length(dec))
rand_dec = asin.(rand_sin_dec) 

rand_ra_pi = rand_ra ./ π 
rand_cos_dec = cos.(rand_dec)
mask = (rand_cos_dec .< 0.1 .*(1 .+ 2 .*rand_ra_pi .+ 8 .*(rand_ra_pi).^2)) .& (rand_cos_dec .< 0.1 .*(1 .+ 2 .*(0.5 .- rand_ra_pi) .+ 8 .*(0.5 .-rand_ra_pi) .^2)) 
rand_ra, rand_dec = rand_ra[mask], rand_dec[mask]
rand_ra .*= 180 / π
rand_dec .*= 180 / π

rand_positions = [Position_RA_DEC(ra, dec, "RANDOM") for (ra, dec) in zip(rand_ra, rand_dec)]
all_positions = vcat(positions, rand_positions)

ψ = corr(ra, dec, all_positions, all_positions, 0.6, 15, 600.0; verbose=true)
