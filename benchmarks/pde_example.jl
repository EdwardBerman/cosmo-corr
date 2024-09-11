using SciMLSensitivity
using DelimitedFiles, Plots
using OrdinaryDiffEq, Zygote

# Problem setup parameters:
Lx = 10.0
x = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

# Initial condition for 500 particles, each starting at random positions
num_particles = 500
particle_positions = rand(num_particles) .* Lx
u0 = zeros(length(x))
for pos in particle_positions
    idx = findfirst(x -> x > pos, x)
    if !isnothing(idx)
        u0[idx] += 1.0 # Adding particle at the nearest grid point
    end
end

## Problem Parameters
p = [1.0]  # Only diffusion coefficient is left as parameter
const xtrs = [dx, Nx]      # Extra parameters
dt = 0.40 * dx^2    # CFL condition
t0, tMax = 0.0, 1000 * dt
tspan = (t0, tMax)
t = t0:dt:tMax;

function d2dx(u, dx)
    return [zero(eltype(u));
            (@view(u[3:end]) .- 2.0 .* @view(u[2:(end - 1)]) .+ @view(u[1:(end - 2)])) ./
            (dx^2)
            zero(eltype(u))]
end

function heat(u, p, t, xtrs)
    a1 = p[1]  
    dx, Nx = xtrs
    return a1 .* d2dx(u, dx) 
end
heat_closure(u, p, t) = heat(u, p, t, xtrs)

prob = ODEProblem(heat_closure, u0, tspan, p)
sol = solve(prob, Tsit5(), dt = dt, saveat = t);
arr_sol = Array(sol)

function final_particle_distribution(θ)
    sol = solve(prob, Tsit5(), p = θ, dt = dt, saveat = t)
    return sol.u[end]  
end

final_state_gradient = gradient(θ -> sum(final_particle_distribution(θ)), p)
println("Gradient of the final state with respect to diffusion constant: ", final_state_gradient)

plot(x, sol.u[1], lw = 3, label = "Initial", size = (800, 500))
plot!(x, sol.u[end], lw = 3, ls = :dash, label = "Final")
