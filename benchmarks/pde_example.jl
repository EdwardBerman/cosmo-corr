using SciMLSensitivity
using DelimitedFiles, Plots
using OrdinaryDiffEq, Optimization, OptimizationPolyalgorithms, Zygote

# Problem setup parameters:
Lx = 10.0
x = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x .- 3.0) .^ 2) # I.C

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
    dx, Nx = xtrs #[1.0,3.0,0.125,100]
    return 2.0 * a0 .* u + a1 .* d2dx(u, dx)
end
heat_closure(u, p, t) = heat(u, p, t, xtrs)

# Testing Solver on linear PDE
prob = ODEProblem(heat_closure, u0, tspan, p)
sol = solve(prob, Tsit5(), dt = dt, saveat = t);
arr_sol = Array(sol)

plot(x, sol.u[1], lw = 3, label = "t0", size = (800, 500))
plot!(x, sol.u[end], lw = 3, ls = :dash, label = "tMax")

ps = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob, Tsit5(), p = θ, dt = dt, saveat = t))
end

## Defining Loss function
function loss(θ)
    pred = predict(θ)
    return sum(abs2.(predict(θ) .- arr_sol)), pred # Mean squared error
end

l, pred = loss(ps)
size(pred), size(sol), size(t) # Checking sizes

LOSS = []                              # Loss accumulator
PRED = []                              # prediction accumulator
PARS = []                              # parameters accumulator

cb = function (θ, l, pred) #callback function to observe training
    display(l)
    append!(PRED, [pred])
    append!(LOSS, l)
    append!(PARS, [θ])
    false
end

cb(ps, loss(ps)...) # Testing callback function

# Let see prediction vs. Truth
scatter(sol[:, end], label = "Truth", size = (800, 500))
plot!(PRED[end][:, end], lw = 2, label = "Prediction")

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

optprob = Optimization.OptimizationProblem(optf, ps)
res = Optimization.solve(optprob, PolyOpt(), callback = cb)
@show res.u # returns [0.999999999613485, 0.9999999991343996]
