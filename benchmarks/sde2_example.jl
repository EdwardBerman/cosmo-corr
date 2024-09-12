using Zygote
using Plots
using SciMLSensitivity
using DifferentialEquations
using Statistics

function sde_function(du, u, p, t)
    μ = p[1]
    du .= μ * u[1]
end

function sde_noise(du, u, p, t)
    σ = p[2]
    du .= σ * u[1]
end

u0 = [1.0]

tspan = (0.0, 1.0)

function sde_solve(p)
    prob = SDEProblem(sde_function, sde_noise, u0, tspan, p)
    sol = solve(prob, SOSRI(), tstops=range(tspan[1], tspan[2], length=50))  # Set consistent time stops
    return sol
end

p = [0.05, 0.2]  

@time begin
    sample(p) = [sde_solve(p) for _ in 1:500]  # Sample the solution 500 times
    grads = jacobian(p -> mean([mean(sol.u[end]) for sol in sample(p)]), p)  # Compute the gradient of the mean at the final time
end
println("Gradient with respect to μ and σ: ", grads)
