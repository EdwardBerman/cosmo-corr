import FreeType, FileIO
using UnicodePlots
using Crayons

# Define colors using RGB values
purple = Crayon(foreground=(149, 88, 178))  # RGB for #9558B2
green = Crayon(foreground=(56, 142, 60))    # RGB for #388E3C
red = Crayon(foreground=(211, 60, 50))      # RGB for #D33C32
blue = Crayon(foreground=(31, 112, 193))    # RGB for #1F70C1

# Sample data
x = 1:10
y1 = rand(10)
y2 = rand(10)
y3 = rand(10)
y4 = rand(10)

# Create the initial line plot with the first color
#
println("\n\n")
plt = lineplot(x, y1, title="CosmoCorr.jl", xlabel="θ", ylabel="ξ(θ)", color=purple)

# Add the other lines with the specified colors
lineplot!(plt, x, y2, color=green)
lineplot!(plt, x, y3, color=red)
lineplot!(plt, x, y4, color=blue)

# Display the plot
println(plt)
println("\n\n")
savefig(plt, "CC_logo.png")
