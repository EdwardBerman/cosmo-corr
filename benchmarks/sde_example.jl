using DifferentialEquations
using Zygote
using SciMLSensitivity
using Statistics

# Define the drift and diffusion functions
function f!(du, u, p, t)
    du[1] = 0.0  # No drift term for Brownian motion
end

# The diffusion function now depends on the parameter p (as a vector)
function g!(du, u, p, t)
    du[1] = p[1]  # Constant diffusion term with parameter p[1]
end

u0 = [0.0]  # Start at the origin
tspan = (0.0, 10.0)  # From time 0 to time 10

function solve_sde(p)
    prob = SDEProblem(f!, g!, u0, tspan, p)
    sol = solve(prob)
    return sol.u[end]  # Return the final value of the solution
end

p = [1.0]  # Parameter p as a vector with one element

sample(p) = [solve_sde(p) for _ in 1:500]  # Sample the solution 100 times

grads = jacobian(p -> mean(sample(p)), p)  # Compute the gradient of the mean
println(grads)  # Print the gradient
