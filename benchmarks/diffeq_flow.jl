using DifferentialEquations
using Zygote
using SciMLSensitivity

struct MyODEParams 
    α::Float64  # Define α as a tunable parameter
end

N = 500        
Δx = 1.0       
tspan = (0.0, 10.0)  


function heat_eq(u, p, t)
    α = p[1]
    du_start = α * (u[2] - u[1]) / Δx^2
    du_end = α * (u[N-1] - u[N]) / Δx^2
    du_middle = [α * (u[i+1] - 2*u[i] + u[i-1]) / Δx^2 for i in 2:N-1]
    return [du_start; du_middle; du_end]
end

function solve_ode(α)
    u0 = [i == Int(N/2) ? 1.0 : 0.0 for i in 1:N]
    p = MyODEParams(α)
    prob = ODEProblem(heat_eq, u0, tspan, p)
    sol = solve(prob, Tsit5())
    return sol[end]  #
end

cost(α) = sum(solve_ode(α))  

dcost_dalpha = Zygote.gradient(cost, 0.01)  # Example α = 0.01
println(dcost_dalpha)

