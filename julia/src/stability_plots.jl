
include("stability_coefficients.jl")

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
                        save=false,
                        G=20)              # grid size for x and y axes


    # Allocate arrays, set up parameter grid
    R = Array{Float64}(G, G);
    x_vals = linspace(p1_min, p1_max, G)   # values for param1 
    y_vals = linspace(p2_min, p2_max, G)   # values for param2


    # Set text, pick constructors
    if model_type == "by"
        EpsteinZin = EpsteinZinBY
        Consumption = BYConsumption
        text = "Bansal and Yaron "
    else
        EpsteinZin = EpsteinZinSSY
        Consumption = SSYConsumption
        text = "Schorfheide, Song and Yaron "
    end

    # Build default instances and extract default parameter values
    default_ez = EpsteinZin()
    default_cp = Consumption()

    cp_names = fieldnames(default_cp)
    ez_names = fieldnames(default_ez)

    if param1 in cp_names
        param1_value = getfield(default_cp, param1)
    else
        param1_value = getfield(default_ez, param1)
    end

    if param2 in cp_names
        param2_value = getfield(default_cp, param2)
    else
        param2_value = getfield(default_ez, param2)
    end

    # Loop through parameters computing test coefficient
    for (i, x) in enumerate(x_vals)
        for (j, y) in enumerate(y_vals)

            # Construct dictionaries used to build instances
            if param1 in cp_names && param2 in cp_names
                cp_d = Dict(param1 => x, param2 => y)
                ez_d = Dict()
            elseif param1 in cp_names && param2 in ez_names
                cp_d = Dict(param1 => x)
                ez_d = Dict(param2 => y)
            elseif param1 in ez_names && param2 in cp_names
                ez_d = Dict(param1 => x)
                cp_d = Dict(param2 => y)
            else
                ez_d = Dict(param1 => x, param2 => y)
                cp_d = Dict()
            end

            # Construct instances
            cp = Consumption(; cp_d...)
            ez = EpsteinZin(; ez_d...)

            if method == "spec_rad"

                r = compute_spec_rad_coef(ez, cp)
                R[i, j] = r
            else

                R[i, j] = compute_ez_coef(ez, cp)
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
