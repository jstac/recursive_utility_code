include("by_stability.jl")
include("ssy_stability.jl")

using PyPlot
plt = PyPlot


function stability_plot(model_type::String,
                        param1::Symbol,    # parameter on x axis
                        p1_min::Float64,   # min value for param1
                        p1_max::Float64,   # min value for param1
                        param2::Symbol,    # parameter on y axis
                        p2_min::Float64,   # min value for param2
                        p2_max::Float64;   # min value for param2
                        coords=(-225, 30),
                        method="spec_rad", # method
                        save=true,
                        G=20)              # grid size for x and y axes

    if model_type == "by"
        ez = EpsteinZinBY()
        cp = BYconsumption()
        text = "Bansal and Yaron "
    else
        ez = EpsteinZinSSY()
        cp = SSYconsumption()
        text = "Schorfheide, Song and Yaron "
    end

    # First extract default parameter values
    if param1 in fieldnames(cp)
        param1_value = getfield(cp, param1)
    else
        param1_value = getfield(ez, param1)
    end

    if param2 in fieldnames(cp)
        param2_value = getfield(cp, param2)
    else
        param2_value = getfield(ez, param2)
    end


    R = Array{Float64}(G, G);

    x_vals = linspace(p1_min, p1_max, G)   # values for param1 
    y_vals = linspace(p2_min, p2_max, G)   # values for param2


    # Loop through parameters computing test coefficient
    for (i, x) in enumerate(x_vals)

        if param1 in fieldnames(cp)
            setfield!(cp, param1, x)
        else
            setfield!(ez, param1, x)
        end


        for (j, y) in enumerate(y_vals)

            if param2 in fieldnames(cp)
                setfield!(cp, param2, y)
            else
                setfield!(ez, param2, y)
            end

            if method == "spec_rad"

                r = compute_spec_rad_coeff(ez, cp)
                R[i, j] = r
            else

                R[i, j] = compute_mm_coef(ez, cp)
            end

        end
    end

    fig, ax = plt.subplots(figsize=(10, 5.7))

    cs1 = ax[:contourf](x_vals, 
                        y_vals, 
                        R', # cmap=plt.cm[:jet],
                        alpha=0.5)

    ctr1 = ax[:contour](x_vals, 
                        y_vals, 
                        R', 
                        levels=[1.0])

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax)


    ax[:annotate](text, 
             xy=(param1_value, param2_value),  
             xycoords="data",
             xytext=coords,
             textcoords="offset points",
             fontsize=12,
             arrowprops=Dict("arrowstyle" => "->"))

    ax[:plot]([param1_value], [param2_value],  "ko", alpha=0.6)

    #ax[:set_title]("Spectral radius")
    ax[:set_xlabel](String(param1), fontsize=16)
    ax[:set_ylabel](String(param2), fontsize=16)

    if save == true
        filename = String(param1) * String(param2) * model_type 
        filename = filename * "_" * method * ".pdf"
        plt.savefig(filename)
    end


    plt.show()

end
