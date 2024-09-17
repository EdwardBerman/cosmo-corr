using DifferentialEquations, Plots
using Zygote, SciMLSensitivity
using Statistics
using LinearAlgebra
using UnicodePlots

function Vincenty_Formula(coord1::Vector{Float64}, coord2::Vector{Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60
end

function Vincenty_Formula(coord1::Tuple{Float64, Float64}, coord2::Tuple{Float64, Float64})
    ϕ1, λ1 = coord1
    ϕ2, λ2 = coord2
    ϕ1 *= π / 180
    λ1 *= π / 180
    ϕ2 *= π / 180
    λ2 *= π / 180
    
    Δλ = λ2 - λ1
    c1 = (cos(ϕ2)*sin(Δλ))^2
    c2 = (cos(ϕ1)*sin(ϕ2) - sin(ϕ1)*cos(ϕ2)*cos(Δλ))^2
    c3 = sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(Δλ)
    y = sqrt(c1 + c2)
    x = c3
    Δσ = atan(y, x)
    return Δσ * (180 / π) * 60 
end

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

#u0_3d = [x0, y0, z0, vx0, vy0, vz0]

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
    prob_3d_list = [ODEProblem(gravitational_ode_3d!, u0, tspan, p) for u0 in u0_3d]
    solulations_3d = [solve(prob_3d) for prob_3d in prob_3d_list]
    end_states = [sol_3d.u[end] for sol_3d in solulations_3d]
    end_states_ra_dec = [xyz_to_ra_dec(end_state[1], end_state[2], end_state[3]) for end_state in end_states]
    return end_states_ra_dec
end

end_states_ra_dec = gravity_sim(p)

distances = [Vincenty_Formula(end_states_ra_dec[i], end_states_ra_dec[j]) for i in 1:500, j in 1:500 if i < j]
println(UnicodePlots.histogram(distances, nbins = 100))

