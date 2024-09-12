using Zygote
using Plots
using SciMLSensitivity
using DifferentialEquations

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
    sol = solve(prob, SOSRI())
    return sol
end

p = [0.05, 0.2]  

sol = sde_solve(p)

plot(sol, title="Geometric Brownian Motion", xlabel="Time", ylabel="X(t)")

function loss_function(p)
    sol = sde_solve(p)
    return sum(sol.u)  # Example loss function: sum of solution values
end

grads = jacobian(loss_function, p)

println("Gradient with respect to μ and σ: ", grads)
