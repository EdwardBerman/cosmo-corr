using Zygote
using Plots
using SciMLSensitivity
using DifferentialEquations
using Statistics
using UnicodePlots

function sde_function(du, u, p, t)
    μ_ra, μ_dec, μ_g1, μ_g2 = p[1], p[2], p[3], p[4]
    du .= μ_ra * u[1], μ_dec * u[2], μ_g1 * u[3], μ_g2 * u[4]
end

function sde_noise(du, u, p, t)
    σ_ra, σ_dec, σ_g1, σ_g2 = p[5], p[6], p[7], p[8]
    du .= σ_ra * u[1], σ_dec * u[2], σ_g1 * u[3], σ_g2 * u[4]
end

u0 = [0.01, 0.01, 0.01, 0.01]

tspan = (0.0, 1.0)

function sde_solve(p)
    prob = SDEProblem(sde_function, sde_noise, u0, tspan, p)
    sol = solve(prob, SOSRI(), tstops=range(tspan[1], tspan[2], length=50))  # Set consistent time stops
    return sol
end

# p = [μ_ra, μ_dec, μ_g1, μ_g2, σ_ra, σ_dec, σ_g1, σ_g2]
p = [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]  

#sol = sde_solve(p)

#sol_ra = [sol.u[i][1] for i in 1:length(sol.u)]
#sol_dec = [sol.u[i][2] for i in 1:length(sol.u)]
#sol_g1 = [sol.u[i][3] for i in 1:length(sol.u)]
#sol_g2 = [sol.u[i][4] for i in 1:length(sol.u)]

@time begin
    sample(p) = [sde_solve(p) for _ in 1:500]  # Sample the solution 500 times
    grads = jacobian(p -> mean([mean(sol.u[:][1] + sol.u[:][2] + sol.u[:][3] + sol.u[:][4]) for sol in sample(p)]), p)
end
println("Gradient with respect to μ and σ: ", grads)
