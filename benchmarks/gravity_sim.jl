using DifferentialEquations

# Define the ODE function for gravitational motion
function gravitational_ode!(du, u, p, t)
    # State vector u: [x, y, vx, vy]
    x, y, vx, vy = u
    # Parameters: G (gravitational constant), M (mass of the central body)
    G = p[1]
    M = p[2]  # assume mass M is 1 for simplicity, you can adjust it if needed

    # Distance from the center
    r = sqrt(x^2 + y^2)

    # Gravitational acceleration
    ax = -G * M * x / r^3
    ay = -G * M * y / r^3

    # Differential equations
    du[1] = vx            # dx/dt = vx
    du[2] = vy            # dy/dt = vy
    du[3] = ax            # dvx/dt = ax
    du[4] = ay            # dvy/dt = ay
end

# Initial conditions: position (x0, y0) and velocity (vx0, vy0)
u0 = [1.0, 0.0, 0.0, 1.0]  # [x0, y0, vx0, vy0]

# Time span for the simulation
tspan = (0.0, 10.0)

# Gravitational constant G and mass M
G = [1.0]
M = 1.0
p = [G[1], M]  # Parameters [G, M]

# Define the ODE problem
prob = ODEProblem(gravitational_ode!, u0, tspan, p)

# Solve the ODE problem
sol = solve(prob)

# Plot the result
using Plots
plot(sol, vars=(1, 2), label="Trajectory", xlabel="x", ylabel="y")


# Time evolve 500 particles according to the save gravitational field, say, around a blackhole
