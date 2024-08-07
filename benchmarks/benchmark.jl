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

ra_min = minimum(ra)
ra_max = maximum(ra)
dec_min = minimum(dec)
dec_max = maximum(dec)

rand_ra = rand(Uniform(ra_min, ra_max), length(ra))
rand_sin_dec = rand(Uniform(sin(dec_min * π / 180), sin(dec_max * π / 180)), length(dec))
rand_dec = asin.(rand_sin_dec) * 180 / π

rand_positions = [Position_RA_DEC(ra, dec, "RANDOM") for (ra, dec) in zip(rand_ra, rand_dec)]
positions = [Position_RA_DEC(ra, dec, "DATA") for (ra, dec) in zip(ra, dec)]
all_positions = vcat(positions, rand_positions)

corr(ra, dec, all_positions, all_positions, 0.6, 15, 600.0; verbose=true)
