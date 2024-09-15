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

#u0_3d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.5]  # [x0, y0, z0, vx0, vy0, vz0]

function generate_initial_conditions(nbodies)
    return [[randn(), randn(), randn(), randn(), randn(), randn()] for i in 1:nbodies]
end

initial_conditions = generate_initial_conditions(500)
tspan = (0.0, 10.0)

G = [1.0]
M = 1.0
p = [G[1], M]  

function xyz_to_ra_dec(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    dec = acos(z / r) * (180 / π )
    ra = atan(y/x) * (180 / π)
    return [ra, dec]
end

function gravity_sim(p; u0_3d = initial_conditions, nbodies = 500)
    G = p[1]
    M = p[2]
    p = [G, M]
    prob_3d_list = [ODEProblem(gravitational_ode_3d!, u0_3d, tspan, p) for u0_3d in u0_3d]
    solulations_3d = [solve(prob_3d) for prob_3d in prob_3d_list]
    end_states = [sol_3d.u[end] for sol_3d in solulations_3d]
    end_states_ra_dec = [xyz_to_ra_dec(end_state[1], end_state[2], end_state[3]) for end_state in end_states]
    return end_states_ra_dec
end


# Time evolve 500 particles according to the save gravitational field, say, around a blackhole
