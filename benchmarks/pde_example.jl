using SciMLSensitivity
using DelimitedFiles, Plots
using OrdinaryDiffEq, Zygote

# Problem setup parameters:
Lx = 10.0
x = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x .- 3.0) .^ 2) # Initial condition

## Problem Parameters
p = [1.0, 1.0]    # True solution parameters
const xtrs = [dx, Nx]      # Extra parameters
dt = 0.40 * dx^2    # CFL condition
t0, tMax = 0.0, 1000 * dt
tspan = (t0, tMax)
t = t0:dt:tMax;

## Definition of Auxiliary functions
function ddx(u, dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    return [[zero(eltype(u))]; (u[3:end] - u[1:(end - 2)]) ./ (2.0 * dx); [zero(eltype(u))]]
end

function d2dx(u, dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    return [zero(eltype(u));
            (@view(u[3:end]) .- 2.0 .* @view(u[2:(end - 1)]) .+ @view(u[1:(end - 2)])) ./
            (dx^2)
            zero(eltype(u))]
end

## ODE description of the Physics:
function heat(u, p, t, xtrs)
    # Model parameters
    a0, a1 = p
    dx, Nx = xtrs 
    return 2.0 * a0 .* u + a1 .* d2dx(u, dx)
end
heat_closure(u, p, t) = heat(u, p, t, xtrs)

# Solver for the heat equation
prob = ODEProblem(heat_closure, u0, tspan, p)
sol = solve(prob, Tsit5(), dt = dt, saveat = t);
arr_sol = Array(sol)

## Loss function definition
function loss(θ)
    pred = Array(solve(prob, Tsit5(), p = θ, dt = dt, saveat = t))
    return sum(abs2.(pred .- arr_sol)) # Mean squared error
end

# Compute the gradient of the loss function with respect to parameters p
loss_gradient = gradient(θ -> loss(θ), p)
println("Gradient of the loss with respect to parameters: ", loss_gradient)

