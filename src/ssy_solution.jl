#=

Code for solving the Schorfheide, Song and Yaron mode.  We will compute the
fixed point g = w^θ of A by iteration, where

    Ag(x) = [(1-β) + (Kg(x))^(1/θ)]^θ

=#

include("ssy_model.jl")

"""
Apply the operator A to scm.g

Writes Ag to scm.g_next.

"""
function A!(scm::SSYComputableModel)

    β, θ = scm.ez.β, scm.θ

    # Update scm.g_next to Kg
    s = K_interp!(scm)  
    # And then update it to Ag
    copy!(scm.g_next, ((1 - β) .+ scm.g_next.^(1/θ)).^θ)
end


"""
Compute the fixed point g = w^θ of A by iteration.

Returns a pointer to an SSYComputableModel instance scm such that

   scm.w_star

is an estimate of the fixed point of T

"""
function compute_solution(ez::EpsteinZin, 
                          ssy::SSYconsumption;
                          g=nothing,            # Initial condition
                          verbose=false, 
                          tol=1e-6,     
                          max_iter=10000) 

    scm = SSYComputableModel(ez, ssy)

    # By default, scm.g is constant at 1 --- this is an opportunity to
    # pick a better initial condition
    if g != nothing
        copy!(scm.g, g)
    end

    error = tol + 1
    i = 1
    while error > tol && i < max_iter
        A!(scm)
        error = maximum(abs, scm.g - scm.g_next)
        i += 1
        copy!(scm.g, scm.g_next)
    end

    if i == max_iter
        warn("Hit iteration upper bound when computing fixed point of T!")
    end

    if verbose == true
        println("Converged $i iterations using tolerance $tol.")
    end

    copy!(scm.w_star, scm.g_next.^(1/scm.θ))
    return scm

end



