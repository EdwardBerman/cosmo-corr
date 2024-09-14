using DifferentialEquations, Plots
using Zygote, SciMLSensitivity
using Statistics
using LinearAlgebra

function gravitational_ode_3d!(du, u, p, t)
    x, y, z, vx, vy, vz = u
    G = p[1]
    M = p[2]  

    r = sqrt(x^2 + y^2 + z^2)

    ax = -G * M * x / r^3
    ay = -G * M * y / r^3
    az = -G * M * z / r^3

    du[1] = vx            # dx/dt = vx
    du[2] = vy            # dy/dt = vy
    du[3] = vz            # dz/dt = vz
    du[4] = ax            # dvx/dt = ax
    du[5] = ay            # dvy/dt = ay
    du[6] = az            # dvz/dt = az
end

u0_3d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.5]  # [x0, y0, z0, vx0, vy0, vz0]

tspan = (0.0, 10.0)

G = [1.0]
M = 1.0
p = [G[1], M]  

prob_3d = ODEProblem(gravitational_ode_3d!, u0_3d, tspan, p)
sol_3d = solve(prob_3d)

function gravity_sim(p; nbodies = 500)
    G = p[1]
    M = p[2]
    p = [G, M]
    prob_3d = ODEProblem(gravitational_ode_3d!, u0_3d, tspan, p)
    solulations_3d = [solve(prob_3d) for i in 1:nbodies]
    end_states = [sol_3d.u[end] for sol_3d in solulations_3d]
    mean_end_state = mean(end_states)
    norm_mean_end_state = norm(mean_end_state)
    return norm_mean_end_state
end

@time begin
    G_gradients = jacobian(gravity_sim, p)
    println(G_gradients)
end

# Time evolve 500 particles according to the save gravitational field, say, around a blackhole
