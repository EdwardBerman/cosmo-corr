module shear_ellipticity
    export shear

    struct shear
        g1::Float64
        g2::Float64
        gtan::Float64
        gcross::Float64
        function shear(g1::Float64, g2::Float64)
            ϕ = π / 4
            shear_tan_cross = -exp(-2im * ϕ) * (g1 + (g2 * 1im))
            gtan, gcross = real(shear_tan_cross), imag(shear_tan_cross)
            return new(g1, g2, gtan, gcross)
        end
    end

end
