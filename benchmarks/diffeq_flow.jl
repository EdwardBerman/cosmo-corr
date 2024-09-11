using DifferentialEquations
using Zygote

# Number of particles and initial parameters
N = 500        # Number of particles
Δx = 1.0       # Spatial step size
tspan = (0.0, 10.0)  # Time span for the simulation

# ODE system for the heat equation with Neumann boundary conditions
function heat_eq!(du, u, α, t)
    # Loop over all particles except the first and last (Neumann boundary conditions)
    for i in 2:N-1
        du[i] = α * (u[i+1] - 2*u[i] + u[i-1]) / Δx^2
    end
    # Neumann boundary conditions (no flux at boundaries)
    du[1] = α * (u[2] - u[1]) / Δx^2
    du[N] = α * (u[N-1] - u[N]) / Δx^2
end

# Define a function that solves the ODE given an initial condition and parameter α
function solve_ode(α)
    u0 = zeros(N)    # Initial condition: all zero temperatures
    u0[Int(N/2)] = 1.0  # Set the center particle to a high temperature to model diffusion from the center

    prob = ODEProblem(heat_eq!, u0, tspan, α)
    sol = solve(prob, Tsit5())
    return sol[end]  # Return the solution at the final time
end

# Define a cost function based on the solution to the ODE
cost(α) = sum(solve_ode(α))  # For example, sum of final temperatures

# Compute the gradient of the cost function with respect to α using Zygote
dcost_dalpha = Zygote.gradient(cost, 0.01)  # Example α = 0.01
println(dcost_dalpha)

