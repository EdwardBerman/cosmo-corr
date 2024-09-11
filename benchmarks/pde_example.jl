using SciMLSensitivity
using DelimitedFiles, Plots
using OrdinaryDiffEq, Zygote
using SciMLSensitivity
using DelimitedFiles, Plots
using OrdinaryDiffEq, Zygote

# Problem setup parameters for 3D:
Lx, Ly, Lz = 10.0, 10.0, 10.0
x = 0.0:0.1:Lx  # Grid for x
y = 0.0:0.1:Ly  # Grid for y
z = 0.0:0.1:Lz  # Grid for z
dx = x[2] - x[1]
dy = y[2] - y[1]
dz = z[2] - z[1]
Nx = size(x)
Ny = size(y)
Nz = size(z)

# Initial condition for 100 walkers, each starting at random x, y, and z positions
num_walkers = 100
walker_positions_x = rand(num_walkers) .* Lx
walker_positions_y = rand(num_walkers) .* Ly
walker_positions_z = rand(num_walkers) .* Lz

u0_x = zeros(length(x))
u0_y = zeros(length(y))
u0_z = zeros(length(z))

for pos_x in walker_positions_x
    idx_x = findfirst(x -> x > pos_x, x)
    if !isnothing(idx_x)
        u0_x[idx_x] += 1.0 # Adding walker at the nearest grid point in x
    end
end

for pos_y in walker_positions_y
    idx_y = findfirst(y -> y > pos_y, y)
    if !isnothing(idx_y)
        u0_y[idx_y] += 1.0 # Adding walker at the nearest grid point in y
    end
end

for pos_z in walker_positions_z
    idx_z = findfirst(z -> z > pos_z, z)
    if !isnothing(idx_z)
        u0_z[idx_z] += 1.0 # Adding walker at the nearest grid point in z
    end
end

## Problem Parameters
p = [1.0, 1.0, 1.0]  # Diffusion coefficients for x, y, and z
const xtrs = [dx, dy, dz, Nx, Ny, Nz]  # Extra parameters
dt = 0.40 * min(dx^2, dy^2, dz^2)  # CFL condition
t0, tMax = 0.0, 1000 * dt
tspan = (t0, tMax)
t = t0:dt:tMax;

## Definition of Auxiliary functions for 3D diffusion
function d2dx(u, dx)
    """
    2nd order Central difference for 2nd degree derivative in x
    """
    return [zero(eltype(u));
            (@view(u[3:end]) .- 2.0 .* @view(u[2:(end - 1)]) .+ @view(u[1:(end - 2)])) ./
            (dx^2)
            zero(eltype(u))]
end

function d2dy(u, dy)
    """
    2nd order Central difference for 2nd degree derivative in y
    """
    return [zero(eltype(u));
            (@view(u[3:end]) .- 2.0 .* @view(u[2:(end - 1)]) .+ @view(u[1:(end - 2)])) ./
            (dy^2)
            zero(eltype(u))]
end

function d2dz(u, dz)
    """
    2nd order Central difference for 2nd degree derivative in z
    """
    return [zero(eltype(u));
            (@view(u[3:end]) .- 2.0 .* @view(u[2:(end - 1)]) .+ @view(u[1:(end - 2)])) ./
            (dz^2)
            zero(eltype(u))]
end

## ODE description of the Physics (No drift term, 3D diffusion):
function heat3d(u_x, u_y, u_z, p, t, xtrs)
    # Model parameters
    a1_x, a1_y, a1_z = p  # Diffusion coefficients for x, y, and z
    dx, dy, dz, Nx, Ny, Nz = xtrs
    
    # Diffusion equations in x, y, and z
    dudx2 = a1_x .* d2dx(u_x, dx)
    dudy2 = a1_y .* d2dy(u_y, dy)
    dudz2 = a1_z .* d2dz(u_z, dz)
    
    return dudx2, dudy2, dudz2
end

heat_closure(u_x, u_y, u_z, p, t) = heat3d(u_x, u_y, u_z, p, t, xtrs)

function sensitivity_heat_eq(u0_x, u0_y, u0_z, p, tspan, xtrs)
    prob_x = ODEProblem((u,p,t)->heat_closure(u,u,u,p,t)[1], u0_x, tspan, p)
    prob_y = ODEProblem((u,p,t)->heat_closure(u,u,u,p,t)[2], u0_y, tspan, p)
    prob_z = ODEProblem((u,p,t)->heat_closure(u,u,u,p,t)[3], u0_z, tspan, p)

    sol_x = solve(prob_x, Tsit5(), dt = dt, saveat = t)
    sol_y = solve(prob_y, Tsit5(), dt = dt, saveat = t)
    sol_z = solve(prob_z, Tsit5(), dt = dt, saveat = t)

    final_positions_x = sol_x[end]
    final_positions_y = sol_y[end]
    final_positions_z = sol_z[end]
    
    return final_positions_x, final_positions_y, final_positions_z
end

# Define the loss function for automatic differentiation
function loss(p)
    final_positions_x, final_positions_y, final_positions_z = sensitivity_heat_eq(u0_x, u0_y, u0_z, p, tspan, xtrs)
    
    # Sum the individual components of final walker positions
    return sum(final_positions_x) + sum(final_positions_y) + sum(final_positions_z)
end

gradients = Zygote.gradient(loss, p)


