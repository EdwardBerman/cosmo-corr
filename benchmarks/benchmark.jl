include("../src/corr.jl")
using .astrocorr

using PyCall
using Statistics
using UnicodePlots
using Base.Threads
using Statistics

println(nthreads())

fits = pyimport("astropy.io.fits")
f = fits.open("Aardvark.fit")
print(f[2].data)

ra = f[2].data["RA"]
dec = f[2].data["DEC"]
ra = convert(Vector{Float64}, ra)
dec = convert(Vector{Float64}, dec)

rand_inds = rand(1:length(ra), 1000)
ra, dec = ra[rand_inds], dec[rand_inds]
positions = [Position_RA_DEC(ra, dec, "DATA") for (ra, dec) in zip(ra, dec)]

corr(ra, dec, positions, positions, 0.6, 15, 600.0; max_depth=5, verbose=true)
