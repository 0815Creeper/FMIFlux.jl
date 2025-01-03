### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
using PlutoUI # Notebook UI

# ╔═╡ 72604eef-5951-4934-844d-d2eb7eb0292c
using FMI # load and simulate FMUs

# ╔═╡ 21104cd1-9fe8-45db-9c21-b733258ff155
using FMIFlux # machine learning with FMUs

# ╔═╡ 9d9e5139-d27e-48c8-a62e-33b2ae5b0086
using FMIZoo # a collection of demo FMUs

# ╔═╡ eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
using FMIFlux.Flux # default Julia Machine Learning library

# ╔═╡ 98c608d9-c60e-4eb6-b611-69d2ae7054c9
using FMIFlux.DifferentialEquations # the mighty (O)DE solver suite

# ╔═╡ ddc9ce37-5f93-4851-a74f-8739b38ab092
using ProgressLogging: @withprogress, @logprogress, @progressid, uuid4

# ╔═╡ de7a4639-e3b8-4439-924d-7d801b4b3eeb
using BenchmarkTools # default benchmarking library

# ╔═╡ 45c4b9dd-0b04-43ae-a715-cd120c571424
using Plots

# ╔═╡ 1470df0f-40e1-45d5-a4cc-519cc3b28fb8
md"""
# Scientific Machine Learning $br using Functional Mock-Up Units
(former *Hybrid Modeling using FMI*)

Workshop $br
@ JuliaCon 2024 (Eindhoven, Netherlands) $br
@ MODPROD 2024 (Linköping University, Sweden)

by Tobias Thummerer (University of Augsburg)

*#hybridmodeling, #sciml, #neuralode, #neuralfmu, #penode*

# Abstract
If there is something YOU know about a physical system, AI shouldn’t need to learn it. How to integrate YOUR system knowledge into a ML development process is the core topic of this hands-on workshop. The entire workshop evolves around a challenging use case from robotics: Modeling a robot that is able to write arbitrary messages with a pen. After introducing the topic and the considered use case, participants can experiment with their very own hybrid model topology. 

# Introduction
This workshop focuses on the integration of Functional Mock-Up Units (FMUs) into a machine learning topology. FMUs are simulation models that can be generated within a variety of modeling tools, see the [FMI homepage](https://fmi-standard.org/). Together with deep neural networks that complement and improve the FMU prediction, so called *neural FMUs* can be created. 
The workshop itself evolves around the hybrid modeling of a *Selective Compliance Assembly Robot Arm* (SCARA), that is able to write user defined words on a sheet of paper. A ready to use physical simulation model (FMU) for the SCARA is given and shortly highlighted in this workshop. However, this model – as any simulation model – shows some deviations if compared to measurements from the real system. These deviations results from not modeled slip-stick-friction: The pen sticks to the paper until a force limit is reached, but then moves jerkily. A hard to model physical effect – but not for a neural FMU.

More advanced code snippets are hidden by default and marked with a ghost `👻`. Computations, that are disabled for performance reasons, are marked with `ℹ️`. They offer a hint how to enable the idled computation by activating the corresponding checkbox marked with `🎬`. 

## Example Video
If you haven't seen such a SCARA system yet, you can watch the following video. There are many more similar videos out there.
"""

# ╔═╡ 7d694be0-cd3f-46ae-96a3-49d07d7cf65a
html"""
<iframe width="560" height="315" src="https://www.youtube.com/embed/ryIwLLr6yRA?si=ncr1IXlnuNhWPWgl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
"""

# ╔═╡ 10cb63ad-03d7-47e9-bc33-16c7786b9f6a
md"""
This video is by *Alexandru Babaian* on YouTube.

## Workshop Video
"""

# ╔═╡ 1e0fa041-a592-42fb-bafd-c7272e346e46
html"""
<iframe width="560" height="315" src="https://www.youtube.com/embed/sQ2MXSswrSo?si=XcEoe1Ai7U6hqnp5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ 6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
md"""
This video is from JuliaCon 2024 (Eindhoven, Netherlands).

## Requirements
To follow this workshop, you should ...
- ... have a rough idea what the *Functional Mock-Up Interface* is and how the standard-conform models - the *Functional Mock-Up Units* - work. If not, a good source is the homepage of the standard, see the [FMI Homepage](https://fmi-standard.org/).
- ... know the *Julia Programming Language* or at least have some programming skills in another high-level programming language like *Python* or *Matlab*. An introduction to Julia can be found on the [Julia Homepage](https://julialang.org/), but there are many more introductions in different formats available on the internet.
- ... have an idea of how modeling (in terms of modeling ODE and DAE systems) and simulation (solving) of such models works.

The technical requirements are:

|   | recommended | minimum | your |
| ----- | ---- | ---- | ---- |
| RAM | $\geq$ 16.0GB | 8.0GB | $(round(Sys.total_memory() / 2^30; digits=1))GB |
| OS | Windows | Windows / Linux | $(Sys.islinux() ? "Linux" : (Sys.iswindows() ? "Windows" : "unsupported"))
| Julia | 1.10 | 1.6 | $("" * string(Int(VERSION.major)) * "." * string(Int(VERSION.minor))) |

This said, we can start "programming"! The entire notebook is pre-implemented, so you can use it without writing a single line of code. Users new to Julia can use interactive UI elements to interact, while more advance users can view and manipulate corresponding code. Let's go! 
"""

# ╔═╡ 8a82d8c7-b781-4600-8780-0a0a003b676c
md"""
## Loading required Julia libraries
Before starting with the actual coding, we load in the required Julia libraries. 
This Pluto-Notebook installs all required packages automatically.
However, this will take some minutes when you start the notebook for the first time... it is recommended to not interact with the UI elements as long as the first compilation runs (orange status light in the bottom right corner).
"""

# ╔═╡ a02f77d1-00d2-46a3-91ba-8a7f5b4bbdc9
md"""
First, we load the Pluto UI elements:
"""

# ╔═╡ 02f0add7-9c4e-4358-8b5e-6863bae3ee75
md"""
Then, the three FMI-libraries we need for FMU loading, machine learning and the FMU itself:
"""

# ╔═╡ 85308992-04c4-4d20-a840-6220cab54680
md"""
Some additional libraries for machine learning and ODE solvers:
"""

# ╔═╡ 3e2579c2-39ce-4249-ad75-228f82e616da
md"""
To visualize a progress bar during training:
"""

# ╔═╡ 93fab704-a8dd-47ec-ac88-13f32be99460
md"""
And to do some benchmarking:
"""

# ╔═╡ 5cb505f7-01bd-4824-8876-3e0f5a922fb7
md""" 
Load in the plotting libraries ...
"""

# ╔═╡ 33d648d3-e66e-488f-a18d-e538ebe9c000
import PlotlyJS

# ╔═╡ 1e9541b8-5394-418d-8c27-2831951c538d
md"""
... and use the beautiful `plotly` backend for interactive plots.
"""

# ╔═╡ e6e91a22-7724-46a3-88c1-315c40660290
plotlyjs()

# ╔═╡ 44500f0a-1b89-44af-b135-39ce0fec5810
md"""
Next, we define some helper functions, that are not important to follow the workshop - they are hidden by default. However they are here, if you want to explore what it takes to write fully working code. If you do this workshop for the first time, it is recommended to skip the hidden part and directly go on.
"""

# ╔═╡ 74d23661-751b-4371-bf6b-986149124e81
md"""
Display the table of contents:
"""

# ╔═╡ c88b0627-2e04-40ab-baa2-b4c1edfda0c3
TableOfContents()

# ╔═╡ 915e4601-12cc-4b7e-b2fe-574e116f3a92
md"""
# Loading Model (FMU) and Data
We want to do hybrid modeling, so we need a simulation model and some data to work with. Fortunately, someone already prepared both for us. We start by loading some data from *FMIZoo.jl*, which is a collection of FMUs and corresponding data.
"""

# ╔═╡ f8e40baa-c1c5-424a-9780-718a42fd2b67
md"""
## Training Data
First, we need some data to train our hybrid model on. We can load data for our SCARA (here called `RobotRR`) with the following line:
"""

# ╔═╡ 74289e0b-1292-41eb-b13b-a4a5763c72b0
# load training data for the `RobotRR` from the FMIZoo
data_train = FMIZoo.RobotRR(:train)

# ╔═╡ 33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
begin

    # define the printing messages used at different places in this notebook
    LIVE_RESULTS_MESSAGE =
        md"""ℹ️ Live plotting is disabled to safe performance. Checkbox `Plot Results`."""
    LIVE_TRAIN_MESSAGE =
        md"""ℹ️ Live training is disabled to safe performance. Checkbox `Start Training`."""
    BENCHMARK_MESSAGE =
        md"""ℹ️ Live benchmarks are disabled to safe performance. Checkbox `Start Benchmark`."""
    HIDDEN_CODE_MESSAGE =
        md"""> 👻 Hidden Code | You probably want to skip this code section on the first run."""

    import FMI.FMIImport.FMICore: hasCurrentComponent, getCurrentComponent, FMU2Solution
    import Random

    function fmiSingleInstanceMode!(
        fmu::FMU2,
        mode::Bool,
        params = FMIZoo.getParameter(data_train, 0.0; friction = false),
        x0 = FMIZoo.getState(data_train, 0.0),
    )

        fmu.executionConfig = deepcopy(FMU2_EXECUTION_CONFIGURATION_NO_RESET)

        # for this model, state events are generated but don't need to be handled,
        # we can skip that to gain performance
        fmu.executionConfig.handleStateEvents = false

        fmu.executionConfig.loggingOn = false
        #fmu.executionConfig.externalCallbacks = true

        if mode
            # switch to a more efficient execution configuration, allocate only a single FMU instance, see:
            # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
            fmu.executionConfig.terminate = true
            fmu.executionConfig.instantiate = false
            fmu.executionConfig.reset = true
            fmu.executionConfig.setup = true
            fmu.executionConfig.freeInstance = false
            c, _ = FMIFlux.prepareSolveFMU(
                fmu,
                nothing,
                fmu.type,
                true, # instantiate 
                false, # free 
                true, # terminate 
                true, # reset 
                true, # setup
                params;
                x0 = x0,
            )
        else
            if !hasCurrentComponent(fmu)
                return nothing
            end
            c = getCurrentComponent(fmu)
            # switch back to the default execution configuration, allocate a new FMU instance for every run, see:
            # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
            fmu.executionConfig.terminate = true
            fmu.executionConfig.instantiate = true
            fmu.executionConfig.reset = true
            fmu.executionConfig.setup = true
            fmu.executionConfig.freeInstance = true
            FMIFlux.finishSolveFMU(
                fmu,
                c,
                true, # free 
                true,
            ) # terminate
        end
        return nothing
    end

    function prepareSolveFMU(fmu, parameters)
        FMIFlux.prepareSolveFMU(
            fmu,
            nothing,
            fmu.type,
            fmu.executionConfig.instantiate,
            fmu.executionConfig.freeInstance,
            fmu.executionConfig.terminate,
            fmu.executionConfig.reset,
            fmu.executionConfig.setup,
            parameters,
        )
    end

    function dividePath(values)
        last_value = values[1]
        paths = []
        path = []
        for j = 1:length(values)
            if values[j] == 1.0
                push!(path, j)
            end

            if values[j] == 0.0 && last_value != 0.0
                push!(path, j)
                push!(paths, path)
                path = []
            end

            last_value = values[j]
        end
        if length(path) > 0
            push!(paths, path)
        end
        return paths
    end

    function plotRobot(solution::FMU2Solution, t::Real)
        x = solution.states(t)
        a1 = x[5]
        a2 = x[3]

        dt = 0.01
        i = 1 + round(Integer, t / dt)
        v = solution.values.saveval[i]

        l1 = 0.2
        l2 = 0.1

        margin = 0.05
        scale = 1500
        fig = plot(;
            title = "Time $(round(t; digits=1))s",
            size = (
                round(Integer, (2 * margin + l1 + l2) * scale),
                round(Integer, (l1 + l2 + 2 * margin) * scale),
            ),
            xlims = (-margin, l1 + l2 + margin),
            ylims = (-l1 - margin, l2 + margin),
            legend = :bottomleft,
        )

        p0 = [0.0, 0.0]
        p1 = p0 .+ [cos(a1) * l1, sin(a1) * l1]
        p2 = p1 .+ [cos(a1 + a2) * l2, sin(a1 + a2) * l2]

        f_norm = collect(v[3] for v in solution.values.saveval)

        paths = dividePath(f_norm)
        drawing = collect(v[1:2] for v in solution.values.saveval)
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in drawing[path]),
                collect(v[2] for v in drawing[path]),
                label = :none,
                color = :black,
                style = :dot,
            )
        end

        paths = dividePath(f_norm[1:i])
        drawing_is = collect(v[4:5] for v in solution.values.saveval)[1:i]
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in drawing_is[path]),
                collect(v[2] for v in drawing_is[path]),
                label = :none,
                color = :green,
                width = 2,
            )
        end

        plot!(fig, [p0[1], p1[1]], [p0[2], p1[2]], label = :none, width = 3, color = :blue)
        plot!(fig, [p1[1], p2[1]], [p1[2], p2[2]], label = :none, width = 3, color = :blue)

        scatter!(
            fig,
            [p0[1]],
            [p0[2]],
            label = "R1 | α1=$(round(a1; digits=3)) rad",
            color = :red,
        )
        scatter!(
            fig,
            [p1[1]],
            [p1[2]],
            label = "R2 | α2=$(round(a2; digits=3)) rad",
            color = :purple,
        )

        scatter!(fig, [v[1]], [v[2]], label = "TCP | F=$(v[3]) N", color = :orange)
    end

    HIDDEN_CODE_MESSAGE

end # begin

# ╔═╡ 92ad1a99-4ad9-4b69-b6f3-84aab49db54f
@bind t_train_plot Slider(0.0:0.1:data_train.t[end], default = data_train.t[1])

# ╔═╡ f111e772-a340-4217-9b63-e7715f773b2c
md"""
Let's have a look on the data! It's the written word *train*.
You can use the slider to pick a specific point in time to plot the "robot" as recorded as part of the data. 

The current picked time is $(round(t_train_plot; digits=1))s.
"""

# ╔═╡ 909de9f1-2aca-4bf0-ba60-d3418964ba4a
plotRobot(data_train.solution, t_train_plot)

# ╔═╡ d8ca5f66-4f55-48ab-a6c9-a0be662811d9
md"""
> 👁️ Interestingly, the first part of the word "trai" is not significantly affected by the slip-stick-effect, the actual TCP trajectory (green) lays quite good on the target position (black dashed). However, the "n" is very jerky. This can be explained by the increasing lever, the motor needs more torque to overcome the static friction the further away the TCP (orange) is from the robot base (red).

Let's extract a start and stop time, as well as saving points for the later solving process:
"""

# ╔═╡ 41b1c7cb-5e3f-4074-a681-36dd2ef94454
tSave = data_train.t # time points to save the solution at

# ╔═╡ 8f45871f-f72a-423f-8101-9ce93e5a885b
tStart = tSave[1]    # start time for simulation of FMU and neural FMU

# ╔═╡ 57c039f7-5b24-4d63-b864-d5f808110b91
tStop = tSave[end]   # stop time for simulation of FMU and neural FMU

# ╔═╡ 4510022b-ad28-4fc2-836b-e4baf3c14d26
md"""
Finally, also the start state can be grabbed from *FMIZoo.jl*, as well as some default parameters for the simulation model we load in the next section. How to interpret the six states is discussed in the next section where the model is loaded.
"""

# ╔═╡ 9589416a-f9b3-4b17-a381-a4f660a5ee4c
x0 = FMIZoo.getState(data_train, tStart)

# ╔═╡ 326ae469-43ab-4bd7-8dc4-64575f4a4d3e
md"""
The parameter array only contains the path to the training data file, the trajectory writing "train".
"""

# ╔═╡ 8f8f91cc-9a92-4182-8f18-098ae3e2c553
parameters = FMIZoo.getParameter(data_train, tStart; friction = false)

# ╔═╡ 8d93a1ed-28a9-4a77-9ac2-5564be3729a5
md"""
## Validation Data
To check whether the hybrid model was not only able to *imitate*, but *understands* the training data, we need some unknown data for validation. In this case, the written word "validate".
"""

# ╔═╡ 4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# load validation data for the `RobotRR` from the FMIZoo
data_validation = FMIZoo.RobotRR(:validate)

# ╔═╡ dbde2da3-e3dc-4b78-8f69-554018533d35
@bind t_validate_plot Slider(0.0:0.1:data_validation.t[end], default = data_validation.t[1])

# ╔═╡ 6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
md"""
Let's have a look on the validation data!
Again, you can use the slider to pick a specific point in time. 

The current time is $(round(t_validate_plot; digits=1))s.
"""

# ╔═╡ d42d0beb-802b-4d30-b5b8-683d76af7c10
plotRobot(data_validation.solution, t_validate_plot)

# ╔═╡ e50d7cc2-7155-42cf-9fef-93afeee6ffa4
md"""
> 👁️ It looks similar to the effect we know from training data, the first part "valida" is not significantly affected by the slip-stick-effect, but the "te" is very jerky. Again, think of the increasing lever ...
"""

# ╔═╡ 3756dd37-03e0-41e9-913e-4b4f183d8b81
md"""
## Simulation Model (FMU)
The SCARA simulation model is called `RobotRR` for `Robot Rotational Rotational`, indicating that this robot consists of two rotational joints, connected by links. It is loaded with the following line of code:
"""

# ╔═╡ 2f83bc62-5a54-472a-87a2-4ddcefd902b6
# load the FMU named `RobotRR` from the FMIZoo
# the FMU was exported from Dymola (version 2023x)
# load the FMU in mode `model-exchange` (ME) 
fmu = fmiLoad("RobotRR", "Dymola", "2023x"; type = :ME)

# ╔═╡ c228eb10-d694-46aa-b952-01d824879287
begin

    # We activate the single instance mode, so only one FMU instance gets allocated and is reused again an again.
    fmiSingleInstanceMode!(fmu, true)

    using FMI.FMIImport: fmi2StringToValueReference

    # declare some model identifiers (inside of the FMU)
    STATE_I1 = fmu.modelDescription.stateValueReferences[2]
    STATE_I2 = fmu.modelDescription.stateValueReferences[1]
    STATE_A1 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational1.revolute1.phi",
    )
    STATE_A2 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational2.revolute1.phi",
    )
    STATE_dA1 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational1.revolute1.w",
    )
    STATE_dA2 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational2.revolute1.w",
    )

    DER_ddA2 = fmu.modelDescription.derivativeValueReferences[4]
    DER_ddA1 = fmu.modelDescription.derivativeValueReferences[6]

    VAR_TCP_PX = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.p_x")
    VAR_TCP_PY = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.p_y")
    VAR_TCP_VX = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.v_x")
    VAR_TCP_VY = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.v_y")
    VAR_TCP_F = fmi2StringToValueReference(fmu, "combiTimeTable.y[3]")

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ 16ffc610-3c21-40f7-afca-e9da806ea626
md"""
Let's check out some meta data of the FMU with `fmiInfo`:
"""

# ╔═╡ 052f2f19-767b-4ede-b268-fce0aee133ad
fmiInfo(fmu)

# ╔═╡ 746fbf6f-ed7c-43b8-8a6f-0377cd3cf85e
md"""
> 👁️ We can read the model name, tool information for the exporting tool, number of event indicators, states, inputs, outputs and whether the optionally implemented FMI features (like *directional derivatives*) are supported by this FMU.
"""

# ╔═╡ 08e1ff54-d115-4da9-8ea7-5e89289723b3
md"""
All six states are listed with all their alias identifiers, that might look a bit awkward the first time. The six states - human readable - are:

| variable reference | description |
| -------- | ------ |
| 33554432 | motor #2 current |
| 33554433 | motor #1 current |
| 33554434 | joint #2 angle |
| 33554435 | joint #2 angular velocity |
| 33554436 | joint #1 angle |
| 33554437 | joint #1 angular velocity |
"""

# ╔═╡ 70c6b605-54fa-40a3-8bce-a88daf6a2022
md"""
To simulate - or *solve* - the ME-FMU, we need an ODE solver. We use the *Tsit5* (explicit Runge-Kutta) here.
"""

# ╔═╡ 634f923a-5e09-42c8-bac0-bf165ab3d12a
solver = Tsit5()

# ╔═╡ f59b5c84-2eae-4e3f-aaec-116c090d454d
md"""
Let's define an array of values we want to be recorded during the first simulation of our FMU. The variable identifiers (like `DER_ddA2`) were pre-defined in the hidden code section above.
"""

# ╔═╡ 0c9493c4-322e-41a0-9ec7-2e2c54ae1373
recordValues = [
    DER_ddA2,
    DER_ddA1, # mechanical accelerations
    STATE_A2,
    STATE_A1, # mechanical angles
    VAR_TCP_PX,
    VAR_TCP_PY, # tool-center-point x and y
    VAR_TCP_VX,
    VAR_TCP_VY, # tool-center-point velocity x and y
    VAR_TCP_F,
] # normal force pen on paper

# ╔═╡ 325c3032-4c78-4408-b86e-d9aa4cfc3187
md"""
Let's simulate the FMU using `fmiSimulate`. In the solution object, different information can be found, like the number of ODE, jacobian or gradient evaluations: 
"""

# ╔═╡ 25e55d1c-388f-469d-99e6-2683c0508693
sol_fmu_train = fmiSimulate(
    fmu,    # our FMU
    (tStart, tStop);           # sim. from tStart to tStop
    solver = solver,    # use the Tsit5 solver
    parameters = parameters,     # the word "train"
    saveat = tSave,    # saving points for the sol.
    recordValues = recordValues,
) # values to record

# ╔═╡ 74c519c9-0eef-4798-acff-b11044bb4bf1
md"""
Now that we know our model and data a little bit better, it's time to care about our hybrid model topology.

# Experiments: $br Hybrid Model Topology

Today is opposite day! Instead of deriving a topology step by step, the final neural FMU topology is presented in the picture below... however, three experiments are intended to make clear why it looks the way it looks.

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_complete.png?raw=true)

The first experiment is on choosing a good interface between FMU and ANN. The second is on online data pre- and post-processing. And the third one on gates, that allow to control the influence of ANN and FMU on the resulting hybrid model dynamics. After you completed all three, you are equipped with the knowledge to cope the final challenge: Build your own neural FMU and train it!
"""

# ╔═╡ 786c4652-583d-43e9-a101-e28c0b6f64e4
md"""
## Choosing interface signals
**between the physical and machine learning domain**

When connecting an FMU with an ANN, technically different signals could be used: States, state derivatives, inputs, outputs, parameters, time itself or other observable variables. Depending on the use case, some signals are more clever to choose than others. In general, every additional signal costs a little bit of computational performance, as you will see. So picking the right subset is the key!

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e1.png?raw=true)
"""

# ╔═╡ 5d688c3d-b5e3-4a3a-9d91-0896cc001000
md"""
We start building our deep model as a `Chain` of layers. For now, there is only a single layer in it: The FMU `fmu` itself. The layer input `x` is interpreted as system state (compare to the figure above) and set in the fmu call via `x=x`. The current solver time `t` is set implicitly. Further, we want all state derivatives as layer outputs by setting `dx_refs=:all` and some additional outputs specified via `y_refs=CHOOSE_y_refs` (you can pick them using the checkboxes). 
"""

# ╔═╡ 68719de3-e11e-4909-99a3-5e05734cc8b1
md"""
Which signals are used for `y_refs`, can be selected:
"""

# ╔═╡ b42bf3d8-e70c-485c-89b3-158eb25d8b25
@bind CHOOSE_y_refs MultiCheckBox([
    STATE_A1 => "Angle Joint 1",
    STATE_A2 => "Angle Joint 2",
    STATE_dA1 => "Angular velocity Joint 1",
    STATE_dA2 => "Angular velocity Joint 2",
    VAR_TCP_PX => "TCP position x",
    VAR_TCP_PY => "TCP position y",
    VAR_TCP_VX => "TCP velocity x",
    VAR_TCP_VY => "TCP velocity y",
    VAR_TCP_F => "TCP (normal) force z",
])

# ╔═╡ 2e08df84-a468-4e99-a277-e2813dfeae5c
model = Chain(x -> fmu(; x = x, dx_refs = :all, y_refs = CHOOSE_y_refs))

# ╔═╡ c446ed22-3b23-487d-801e-c23742f81047
md"""
Let's pick a state `x1` one second after simulation start to determine sensitivities for:
"""

# ╔═╡ fc3d7989-ac10-4a82-8777-eeecd354a7d0
x1 = FMIZoo.getState(data_train, tStart + 1.0)

# ╔═╡ f4e66f76-76ff-4e21-b4b5-c1ecfd846329
begin
    using FMIFlux.FMISensitivity.ReverseDiff
    using FMIFlux.FMISensitivity.ForwardDiff

    prepareSolveFMU(fmu, parameters)
    jac_rwd = ReverseDiff.jacobian(x -> model(x), x1)
    A_rwd = jac_rwd[1:length(x1), :]
end

# ╔═╡ 0a7955e7-7c1a-4396-9613-f8583195c0a8
md"""
Depending on how many signals you select, the output of the FMU-layer is extended. The first six outputs are the state derivatives, the remaining are the $(length(CHOOSE_y_refs)) additional output(s) selected above.
"""

# ╔═╡ 4912d9c9-d68d-4afd-9961-5d8315884f75
begin
    dx_y = model(x1)
end

# ╔═╡ 19942162-cd4e-487c-8073-ea6b262d299d
md"""
Derivatives:
"""

# ╔═╡ 73575386-673b-40cc-b3cb-0b8b4f66a604
ẋ = dx_y[1:length(x1)]

# ╔═╡ 24861a50-2319-4c63-a800-a0a03279efe2
md"""
Additional outputs:
"""

# ╔═╡ 93735dca-c9f3-4f1a-b1bd-dfe312a0644a
y = dx_y[length(x1)+1:end]

# ╔═╡ 13ede3cd-99b1-4e65-8a18-9043db544728
md"""
For the later training, we need gradients and Jacobians.
"""

# ╔═╡ f7c119dd-c123-4c43-812e-d0625817d77e
md"""
If we use reverse-mode automatic differentiation via `ReverseDiff.jl`, the determined Jacobian $A = \frac{\partial \dot{x}}{\partial x}$ states: 
"""

# ╔═╡ b163115b-393d-4589-842d-03859f05be9a
md"""
Forward-mode automatic differentiation (using *ForwardDiff.jl*)is available, too.

We can determine further Jacobians for FMUs, for example the Jacobian $C = \frac{\partial y}{\partial x}$ states (using *ReverseDiff.jl*): 
"""

# ╔═╡ ac0afa6c-b6ec-4577-aeb6-10d1ec63fa41
begin
    C_rwd = jac_rwd[length(x1)+1:end, :]
end

# ╔═╡ 5e9cb956-d5ea-4462-a649-b133a77929b0
md"""
Let's check the performance of these calls, because they will have significant influence on the later training performance!
"""

# ╔═╡ 9dc93971-85b6-463b-bd17-43068d57de94
md"""
### Benchmark
The amount of selected signals has influence on the computational performance of the model. The more signals you use, the slower is inference and gradient determination. For now, you have picked $(length(CHOOSE_y_refs)) additional signal(s). 
"""

# ╔═╡ 476a1ed7-c865-4878-a948-da73d3c81070
begin
    CHOOSE_y_refs

    md"""
    🎬 **Start Benchmark** $(@bind BENCHMARK CheckBox())
    (benchmarking takes around 10 seconds)
    """
end

# ╔═╡ 0b6b4f6d-be09-42f3-bc2c-5f17a8a9ab0e
md"""
The current timing and allocations for inference are:
"""

# ╔═╡ a1aca180-d561-42a3-8d12-88f5a3721aae
begin
    if BENCHMARK
        @btime model(x1)
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ 3bc2b859-d7b1-4b79-88df-8fb517a6929d
md"""
Gradient and Jacobian computation takes a little longer of course. We use reverse-mode automatic differentiation via `ReverseDiff.jl` here:
"""

# ╔═╡ a501d998-6fd6-496f-9718-3340c42b08a6
begin
    if BENCHMARK
        prepareSolveFMU(fmu, parameters)
        function ben_rwd(x)
            return ReverseDiff.jacobian(model, x + rand(6) * 1e-12)
        end
        @btime ben_rwd(x1)
        #nothing
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ 83a2122d-56da-4a80-8c10-615a8f76c4c1
md"""
Further, forward-mode automatic differentiation is available too via `ForwardDiff.jl`, but a little bit slower than reverse-mode:
"""

# ╔═╡ e342be7e-0806-4f72-9e32-6d74ed3ed3f2
begin
    if BENCHMARK
        prepareSolveFMU(fmu, parameters)
        function ben_fwd(x)
            return ForwardDiff.jacobian(model, x + rand(6) * 1e-12)
        end
        @btime ben_fwd(x1) # second run for "benchmarking"
    #nothing
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ eaf37128-0377-42b6-aa81-58f0a815276b
md"""
> 💡 Keep in mind that the choice of interface might has a significant impact on your inference and training performance! However, some signals are simply required to be part of the interface, because the effect we want to train for depends on them.
"""

# ╔═╡ c030d85e-af69-49c9-a7c8-e490d4831324
md"""
## Online Data Pre- and Postprocessing
**is required for hybrid models**

Now that we have defined the signals that come *from* the FMU and go *into* the ANN, we need to think about data pre- and post-processing. In ML, this is often done before the actual training starts. In hybrid modeling, we need to do this *online*, because the FMU constantly generates signals that might not be suitable for ANNs. On the other hand, the signals generated by ANNs might not suit the expected FMU input. What *suitable* means gets more clear if we have a look on the used activation functions, like e.g. the *tanh*.

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e2.png?raw=true)

We simplify the ANN to a single nonlinear activation function. Let's see what's happening as soon as we put the derivative *angular velocity of joint 1* (dα1) from the FMU into a `tanh` function:
"""

# ╔═╡ 51c200c9-0de3-4e50-8884-49fe06158560
begin
    fig_pre_post1 = plot(
        layout = grid(1, 2, widths = (1 / 4, 3 / 4)),
        xlabel = "t [s]",
        legend = :bottomright,
    )

    plot!(fig_pre_post1[1], data_train.t, data_train.da1, label = :none, xlims = (0.0, 0.1))
    plot!(fig_pre_post1[1], data_train.t, tanh.(data_train.da1), label = :none)

    plot!(fig_pre_post1[2], data_train.t, data_train.da1, label = "dα1")
    plot!(fig_pre_post1[2], data_train.t, tanh.(data_train.da1), label = "tanh(dα1)")

    fig_pre_post1
end

# ╔═╡ 0dadd112-3132-4491-9f02-f43cf00aa1f9
md"""
In general, it looks like the velocity isn't saturated too much by `tanh`. This is a good thing and not always the case! However, the very beginning of the trajectory is saturated too much (the peak value of $\approx -3$ is saturated to $\approx -1$). This is bad, because the hybrid model velocity is *slower* in this time interval and the hybrid system won't reach the same angle over time as the original FMU.

We can add shift (=addition) and scale (=multiplication) operations before and after the ANN to bypass this issue. See how you can influence the output *after* the `tanh` (and the ANN respectively) to match the ranges. The goal is to choose pre- and post-processing parameters so that the signal ranges needed by the FMU are preserved by the hybrid model.
"""

# ╔═╡ bf6bf640-54bc-44ef-bd4d-b98e934d416e
@bind PRE_POST_SHIFT Slider(-1:0.1:1.0, default = 0.0)

# ╔═╡ 5c2308d9-6d04-4b38-af3b-6241da3b6871
md"""
Change the `shift` value $(PRE_POST_SHIFT):
"""

# ╔═╡ 007d6d95-ad85-4804-9651-9ac3703d3b40
@bind PRE_POST_SCALE Slider(0.1:0.1:2.0, default = 1.0)

# ╔═╡ 639889b3-b9f2-4a3c-999d-332851768fd7
md"""
Change the `scale` value $(PRE_POST_SCALE):
"""

# ╔═╡ ed1887df-5079-4367-ab04-9d02a1d6f366
begin
    fun_pre = ShiftScale([PRE_POST_SHIFT], [PRE_POST_SCALE])
    fun_post = ScaleShift(fun_pre)

    fig_pre_post2 = plot(; layout = grid(1, 2, widths = (1 / 4, 3 / 4)), xlabel = "t [s]")

    plot!(
        fig_pre_post2[2],
        data_train.t,
        data_train.da1,
        label = :none,
        title = "Shift: $(round(PRE_POST_SHIFT; digits=1)) | Scale: $(round(PRE_POST_SCALE; digits=1))",
        legend = :bottomright,
    )
    plot!(fig_pre_post2[2], data_train.t, tanh.(data_train.da1), label = :none)
    plot!(
        fig_pre_post2[2],
        data_train.t,
        fun_post(tanh.(fun_pre(data_train.da1))),
        label = :none,
    )

    plot!(fig_pre_post2[1], data_train.t, data_train.da1, label = "dα1", xlims = (0.0, 0.1))
    plot!(fig_pre_post2[1], data_train.t, tanh.(data_train.da1), label = "tanh(dα1)")
    plot!(
        fig_pre_post2[1],
        data_train.t,
        fun_post(tanh.(fun_pre(data_train.da1))),
        label = "post(tanh(pre(dα1)))",
    )

    fig_pre_post2
end

# ╔═╡ 0b0c4650-2ce1-4879-9acd-81c16d06700e
md"""
The left plot shows the negative spike at the very beginning in more detail. In *FMIFlux.jl*, there are ready to use layers for scaling and shifting, that can automatically select appropriate parameters. These parameters are trained together with the ANN parameters by default, so they can adapt to new signal ranges that might occur during training.
"""

# ╔═╡ b864631b-a9f3-40d4-a6a8-0b57a37a476d
md"""
> 💡 In many machine learning applications, pre- and post-processing is done offline. If we combine machine learning and physical models, we need to pre- and post-process online at the interfaces. This does at least improve training performance and is a necessity if the nominal values become very large or very small.
"""

# ╔═╡ 0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
md"""
## Introducing Gates
**to control how physical and machine learning model contribute and interact**

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e3.png?raw=true)
"""

# ╔═╡ 95e14ea5-d82d-4044-8c68-090d74d95a61
md"""
There are two obvious ways of connecting two blocks (the ANN and the FMU):
- In **series**, so one block is getting signals from the other block and is able to *manipulate* or *correct* these signals. This way, e.g. modeling or parameterization errors can be corrected.
- In **parallel**, so both are getting the same signals and calculate own outputs, these outputs must be merged afterwards. This way, additional system parts, like e.g. forces or momentum, can be learned and added to or augment the existing dynamics.

The good news is, you don't have to decide this beforehand. This is something that the optimizer can decide, if we introduce a topology with parameters, that allow for both modes. This structure is referred to as *gates*.
"""

# ╔═╡ cbae6aa4-1338-428c-86aa-61d3304e33ed
@bind GATE_INIT_FMU Slider(0.0:0.1:1.0, default = 1.0)

# ╔═╡ 2fa1821b-aaec-4de4-bfb4-89560790dc39
md"""
Change the opening of the **FMU gate** $(GATE_INIT_FMU) for dα1:
"""

# ╔═╡ 8c56acd6-94d3-4cbc-bc29-d249740268a0
@bind GATE_INIT_ANN Slider(0.0:0.1:1.0, default = 0.0)

# ╔═╡ 9b52a65a-f20c-4387-aaca-5292a92fb639
md"""
Change the opening of the **ANN gate** $(GATE_INIT_ANN) for dα1:
"""

# ╔═╡ 845a95c4-9a35-44ae-854c-57432200da1a
md"""
The FMU gate value for dα1 is $(GATE_INIT_FMU) and the ANN gate value is $(GATE_INIT_ANN). This means the hybrid model dα1 is composed of $(GATE_INIT_FMU*100)% of dα1 from the FMU and of $(GATE_INIT_ANN*100)% of dα1 from the ANN.
"""

# ╔═╡ 5a399a9b-32d9-4f93-a41f-8f16a4b102dc
begin
    function build_model_gates()
        Random.seed!(123)

        cache = CacheLayer()                        # allocate a cache layer
        cacheRetrieve = CacheRetrieveLayer(cache)   # allocate a cache retrieve layer, link it to the cache layer

        # we have two signals (acceleration, consumption) and two sources (ANN, FMU), so four gates:
        # (1) acceleration from FMU (gate=1.0 | open)
        # (2) consumption  from FMU (gate=1.0 | open)
        # (3) acceleration from ANN (gate=0.0 | closed)
        # (4) consumption  from ANN (gate=0.0 | closed)
        # the accelerations [1,3] and consumptions [2,4] are paired
        gates = ScaleSum([GATE_INIT_FMU, GATE_INIT_ANN], [[1, 2]]) # gates with sum

        # setup the neural FMU topology
        model_gates = Flux.f64(
            Chain(
                dx -> cache(dx),                    # cache `dx`
                Dense(1, 16, tanh),
                Dense(16, 1, tanh),  # pre-process `dx`
                dx -> cacheRetrieve(1, dx),       # dynamics FMU | dynamics ANN
                gates,
            ),
        )       # stack together

        model_input = collect([v] for v in data_train.da1)
        model_output = collect(model_gates(inp) for inp in model_input)
        ANN_output = collect(model_gates[2:3](inp) for inp in model_input)

        fig = plot(; ylims = (-3, 1), legend = :bottomright)
        plot!(fig, data_train.t, collect(v[1] for v in model_input), label = "dα1 of FMU")
        plot!(fig, data_train.t, collect(v[1] for v in ANN_output), label = "dα1 of ANN")
        plot!(
            fig,
            data_train.t,
            collect(v[1] for v in model_output),
            label = "dα1 of neural FMU",
        )

        return fig
    end
    build_model_gates()
end

# ╔═╡ fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
md"""
Some observations from the current gate openings are:

This equals the serial topology: $((GATE_INIT_FMU==0 && GATE_INIT_ANN==1) ? "✔️" : "❌") $br
This equals the parallel topology: $((GATE_INIT_FMU==1 && GATE_INIT_ANN==1) ? "✔️" : "❌") $br
The neural FMU dynamics equal the FMU dynamics: $((GATE_INIT_FMU==1 && GATE_INIT_ANN==0) ? "✔️" : "❌")
"""

# ╔═╡ 1cd976fb-db40-4ebe-b40d-b996e16fc213
md"""
> 💡 Gates allow to make parts of the architecture *learnable* while still keeping the training results interpretable.
"""

# ╔═╡ 93771b35-4edd-49e3-bed1-a3ccdb7975e6
md"""
> 💭 **Further reading:** Optimizing the gates together with the ANN parameters seems a useful strategy if we don't know how FMU and ANN need to interact in the later application. Technically, we keep a part of the architecture *parameterizable* and therefore learnable. How far can we push this game?
>
> Actually to the point, that the combination of FMU and ANN is described by a single *connection* equation, that is able to express all possible combinations of both models with each other - so a connection between every pair of inputs and outputs. This is discussed in detail as part of our article [*Learnable & Interpretable Model Combination in Dynamic Systems Modeling*](https://doi.org/10.48550/arXiv.2406.08093).
"""

# ╔═╡ e79badcd-0396-4a44-9318-8c6b0a94c5c8
md"""
Time to take care of the big picture next.
"""

# ╔═╡ 2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
md"""
# Building the neural FMU
**... putting everything together**

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_train.png?raw=true)
"""

# ╔═╡ 4454c8d2-68ed-44b4-adfa-432297cdc957
md"""
## FMU inputs
In general, you can use arbitrary values as input for the FMU layer, like system inputs, states or parameters. In this example, we want to use only system states as inputs for the FMU layer - to keep it easy - which are:
- currents of both motors
- angles of both joints
- angular velocities of both joints

To preserve the ODE topology (a mapping from state to state derivative), we use all system state derivatives as layer outputs. However, you can choose further outputs if you want to... and you definitely should.

## ANN inputs
As input to the ANN, we choose at least the angular accelerations of both joints - this is fixed:

- angular acceleration Joint 1
- angular acceleration Joint 2

Pick additional ANN layer inputs:
"""

# ╔═╡ d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
@bind y_refs MultiCheckBox([
    STATE_A2 => "Angle Joint 2",
    STATE_A1 => "Angle Joint 1",
    STATE_dA1 => "Angular velocity Joint 1",
    STATE_dA2 => "Angular velocity Joint 2",
    VAR_TCP_PX => "TCP position x",
    VAR_TCP_PY => "TCP position y",
    VAR_TCP_VX => "TCP velocity x",
    VAR_TCP_VY => "TCP velocity y",
    VAR_TCP_F => "TCP (normal) force z",
])

# ╔═╡ 06937575-9ab1-41cd-960c-7eef3e8cae7f
md"""
It might be clever to pick additional inputs, because the effect being learned (slip-stick of the pen) might depend on these additional inputs. However, every additional signal has a little negative impact on the computational performance and a risk of learning from wrong correlations.
"""

# ╔═╡ 356b6029-de66-418f-8273-6db6464f9fbf
md"""
## ANN size
"""

# ╔═╡ 5805a216-2536-44ac-a702-d92e86d435a4
md"""
The ANN shall have $(@bind NUM_LAYERS Select([2, 3, 4])) layers with a width of $(@bind LAYERS_WIDTH Select([8, 16, 32])) each.
"""

# ╔═╡ 53e971d8-bf43-41cc-ac2b-20dceaa78667
@bind GATES_INIT Slider(0.0:0.1:1.0, default = 0.0)

# ╔═╡ 68d57a23-68c3-418c-9c6f-32bdf8cafceb
md"""
The ANN gates shall be initialized with $(GATES_INIT), slide to change:
"""

# ╔═╡ e8b8c63b-2ca4-4e6a-a801-852d6149283e
md"""
ANN gates shall be initialized with $(GATES_INIT), meaning the ANN contributes $(GATES_INIT*100)% to the hybrid model derivatives, while the FMU contributes $(100-GATES_INIT*100)%. These parameters are adapted during training, these are only start values.
"""

# ╔═╡ c0ac7902-0716-4f18-9447-d18ce9081ba5
md"""
## Resulting neural FMU
Our final neural FMU topology looks like this:
"""

# ╔═╡ 84215a73-1ab0-416d-a9db-6b29cd4f5d2a
begin

    function build_topology(gates_init, add_y_refs, nl, lw)

        ANN_input_Vars = [recordValues[1:2]..., add_y_refs...]
        ANN_input_Vals = fmiGetSolutionValue(sol_fmu_train, ANN_input_Vars)
        ANN_input_Idcs = [4, 6]
        for i = 1:length(add_y_refs)
            push!(ANN_input_Idcs, i + 6)
        end

        # pre- and post-processing
        preProcess = ShiftScale(ANN_input_Vals)         # we put in the derivatives recorded above, FMIFlux shift and scales so we have a data mean of 0 and a standard deviation of 1
        #preProcess.scale[:] *= 0.1                         # add some additional "buffer"
        postProcess = ScaleShift(preProcess; indices = [1, 2])   # initialize the postProcess as inverse of the preProcess, but only take indices 1 and 2

        # cache
        cache = CacheLayer()                        # allocate a cache layer
        cacheRetrieve = CacheRetrieveLayer(cache)   # allocate a cache retrieve layer, link it to the cache layer

        gates = ScaleSum(
            [1.0 - gates_init, 1.0 - gates_init, gates_init, gates_init],
            [[1, 3], [2, 4]],
        ) # gates with sum

        ANN_layers = []
        push!(ANN_layers, Dense(2 + length(add_y_refs), lw, tanh)) # first layer 
        for i = 3:nl
            push!(ANN_layers, Dense(lw, lw, tanh))
        end
        push!(ANN_layers, Dense(lw, 2, tanh)) # last layer 

        model = Flux.f64(
            Chain(
                x -> fmu(; x = x, dx_refs = :all, y_refs = add_y_refs),
                dxy -> cache(dxy),                    # cache `dx`
                dxy -> dxy[ANN_input_Idcs],
                preProcess,
                ANN_layers...,
                postProcess,
                dx -> cacheRetrieve(4, 6, dx),       # dynamics FMU | dynamics ANN
                gates,                              # compute resulting dx from ANN + FMU
                dx -> cacheRetrieve(1:3, dx[1], 5, dx[2]),
            ),
        )

        return model

    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ bc09bd09-2874-431a-bbbb-3d53c632be39
md"""
We find a `Chain` consisting of multipl layers and the corresponding parameter counts. We can evaluate it, by putting in our start state `x0`. The model computes the resulting state derivative:
"""

# ╔═╡ f02b9118-3fb5-4846-8c08-7e9bbca9d208
md"""
On basis of this `Chain`, we can build a neural FMU very easy:
"""

# ╔═╡ d347d51b-743f-4fec-bed7-6cca2b17bacb
md"""
So let's get that thing trained!

# Training

After setting everything up, we can give it a try and train our created neural FMU. Depending on the chosen optimization hyperparameters, this will be more or less successful. Feel free to play around a bit, but keep in mind that for real application design, you should do hyper parameter optimization instead of playing around by yourself.
"""

# ╔═╡ d60d2561-51a4-4f8a-9819-898d70596e0c
md"""
## Hyperparameters
Besides the already introduced hyperparameters - the depth, width and initial gate opening of the hybrid model - further parameters might have significant impact on the training success.

### Optimizer
For this example, we use the well-known `Adam`-Optimizer with a step size `eta` of $(@bind ETA Select([1e-4 => "1e-4", 1e-3 => "1e-3", 1e-2 => "1e-2"])). 

### Batching 
Because data has a significant length, gradient computation over the entire simulation trajectory might not be effective. The most common approach is to *cut* data into slices and train on these subsets instead of the entire trajectory at once. In this example, data is cut in pieces with length of $(@bind BATCHDUR Select([0.05, 0.1, 0.15, 0.2])) seconds.
"""

# ╔═╡ c97f2dea-cb18-409d-9ae8-1d03647a6bb3
md"""
This results in a batch with $(round(Integer, data_train.t[end] / BATCHDUR)) elements.
"""

# ╔═╡ 366abd1a-bcb5-480d-b1fb-7c76930dc8fc
md"""
We use a simple `Random` scheduler here, that picks a random batch element for the next training step. Other schedulers are pre-implemented in *FMIFlux.jl*.
"""

# ╔═╡ 7e2ffd6f-19b0-435d-8e3c-df24a591bc55
md"""
### Loss Function
Different loss functions are thinkable here. Two quantities that should be considered are the motor currents and the motor revolution speeds. For this workshop we use the *Mean Average Error* (MAE) over the motor currents. Other loss functions can easily be deployed.
"""

# ╔═╡ caa5e04a-2375-4c56-8072-52c140adcbbb
# goal is to match the motor currents (they can be recorded easily in the real application)
function loss(solution::FMU2Solution, data::FMIZoo.RobotRR_Data)

    # determine the start/end indices `ts` and `te` (sampled with 100Hz)
    dt = 0.01
    ts = 1 + round(Integer, solution.states.t[1] / dt)
    te = 1 + round(Integer, solution.states.t[end] / dt)

    # retrieve simulation data from neural FMU ("where we are") and data from measurements ("where we want to be")
    i1_value = fmiGetSolutionState(solution, STATE_I1)
    i2_value = fmiGetSolutionState(solution, STATE_I2)
    i1_data = data.i1[ts:te]
    i2_data = data.i2[ts:te]

    # accumulate our loss value
    Δvalue = 0.0
    Δvalue += FMIFlux.Losses.mae(i1_value, i1_data)
    Δvalue += FMIFlux.Losses.mae(i2_value, i2_data)

    return Δvalue
end

# ╔═╡ 69657be6-6315-4655-81e2-8edef7f21e49
md"""
For example, the loss function value of the plain FMU is $(round(loss(sol_fmu_train, data_train); digits=6)).
"""

# ╔═╡ 23ad65c8-5723-4858-9abe-750c3b65c28a
md"""
## Summary
To summarize, your ANN has a **depth of $(NUM_LAYERS) layers** with a **width of $(LAYERS_WIDTH)** each. The **ANN gates are initialized with $(GATES_INIT*100)%**, so all FMU gates are initialized with $(100-GATES_INIT*100)%. You decided to batch your data with a **batch element length of $(BATCHDUR)** seconds. Besides the state derivatives, you **put $(length(y_refs)) additional variables** in the ANN. Adam optimizer will try to find a good minimum with **`eta` is $(ETA)**.

Batching takes a few seconds and training a few minutes (depending on the number of training steps), so this is not triggered automatically. If you are ready to go, choose a number of training steps and check the checkbox `Start Training`. This will start a training of $(@bind STEPS Select([0, 10, 100, 1000, 2500, 5000, 10000])) training steps. Alternatively, you can change the training mode to `demo` which loads parameters from a pre-trained model.
"""

# ╔═╡ abc57328-4de8-42d8-9e79-dd4020769dd9
md"""
Select training mode:
$(@bind MODE Select([:train => "Training", :demo => "Demo (pre-trained)"]))
"""

# ╔═╡ f9d35cfd-4ae5-4dcd-94d9-02aefc99bdfb
begin
    using JLD2

    if MODE == :train
        final_model = build_topology(GATES_INIT, y_refs, NUM_LAYERS, LAYERS_WIDTH)
    elseif MODE == :demo
        final_model = build_topology(
            0.2,
            [STATE_A2, STATE_A1, VAR_TCP_VX, VAR_TCP_VY, VAR_TCP_F],
            3,
            32,
        )
    end
end

# ╔═╡ f741b213-a20d-423a-a382-75cae1123f2c
final_model(x0)

# ╔═╡ 91473bef-bc23-43ed-9989-34e62166d455
begin
    neuralFMU = ME_NeuralFMU(
        fmu, # the FMU used in the neural FMU 
        final_model,    # the model we specified above 
        (tStart, tStop),# start and stop time for solving
        solver; # the solver (Tsit5)
        saveat = tSave,
    )   # time points to save the solution at
end

# ╔═╡ 404ca10f-d944-4a9f-addb-05efebb4f159
begin
    import Downloads
    demo_path = Downloads.download(
        "https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/20000.jld2?raw=true",
    )

    # in demo mode, we load parameters from a pre-trained model
    if MODE == :demo
        fmiLoadParameters(neuralFMU, demo_path)
    end

    HIDDEN_CODE_MESSAGE
end

# ╔═╡ e8bae97d-9f90-47d2-9263-dc8fc065c3d0
begin
    neuralFMU
    y_refs
    NUM_LAYERS
    LAYERS_WIDTH
    GATES_INIT
    ETA
    BATCHDUR
    MODE

    if MODE == :train
        md"""⚠️ The roughly estimated training time is **$(round(Integer, STEPS*10*BATCHDUR + 0.6/BATCHDUR)) seconds** (Windows, i7 @ 3.6GHz). Training might be faster if the system is less stiff than expected. Once you started training by clicking on `Start Training`, training can't be terminated easily.
      	
      🎬 **Start Training** $(@bind LIVE_TRAIN CheckBox())
      		"""
    else
        LIVE_TRAIN = false
        md"""ℹ️ No training in demo mode. Please continue with plotting results.
        """
    end
end

# ╔═╡ 2dce68a7-27ec-4ffc-afba-87af4f1cb630
begin

    function train(eta, batchdur, steps)

        if steps == 0
            return md"""⚠️ Number of training steps is `0`, no training."""
        end

        prepareSolveFMU(fmu, parameters)

        train_t = data_train.t
        train_data = collect([data_train.i2[i], data_train.i1[i]] for i = 1:length(train_t))

        #@info 
        @info "Started batching ..."

        batch = batchDataSolution(
            neuralFMU, # our neural FMU model
            t -> FMIZoo.getState(data_train, t), # a function returning a start state for a given time point `t`, to determine start states for batch elements
            train_t, # data time points
            train_data; # data cumulative consumption 
            batchDuration = batchdur, # duration of one batch element
            indicesModel = [1, 2], # model indices to train on (1 and 2 equal the `electrical current` states)
            plot = false, # don't show intermediate plots (try this outside of Pluto)
            showProgress = false,
            parameters = parameters,
        )

        @info "... batching finished!"

        # a random element scheduler
        scheduler = RandomScheduler(neuralFMU, batch; applyStep = 1, plotStep = 0)

        lossFct = (solution::FMU2Solution) -> loss(solution, data_train)

        maxiters = round(Int, 1e5 * batchdur)

        _loss =
            p -> FMIFlux.Losses.loss(
                neuralFMU, # the neural FMU to simulate
                batch; # the batch to take an element from
                p = p, # the neural FMU training parameters (given as input)
                lossFct = lossFct, # our custom loss function
                batchIndex = scheduler.elementIndex, # the index of the batch element to take, determined by the chosen scheduler
                logLoss = true, # log losses after every evaluation
                showProgress = false,
                parameters = parameters,
                maxiters = maxiters,
            )

        params = FMIFlux.params(neuralFMU)

        FMIFlux.initialize!(
            scheduler;
            p = params[1],
            showProgress = false,
            parameters = parameters,
            print = false,
        )

        BETA1 = 0.9
        BETA2 = 0.999
        optim = Adam(eta, (BETA1, BETA2))

        @info "Started training ..."

        @withprogress name = "iterating" begin
            iteration = 0
            function cb()
                iteration += 1
                @logprogress iteration / steps
                FMIFlux.update!(scheduler; print = false)
                nothing
            end

            FMIFlux.train!(
                _loss, # the loss function for training
                neuralFMU, # the parameters to train
                Iterators.repeated((), steps), # an iterator repeating `steps` times
                optim; # the optimizer to train
                gradient = :ReverseDiff, # use ReverseDiff, because it's much faster!
                cb = cb, # update the scheduler after every step 
                proceed_on_assert = true,
            ) # go on if a training steps fails (e.g. because of instability)  
        end

        @info "... training finished!"
    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ c3f5704b-8e98-4c46-be7a-18ab4f139458
let
    if MODE == :train
        if LIVE_TRAIN
            train(ETA, BATCHDUR, STEPS)
        else
            LIVE_TRAIN_MESSAGE
        end
    else
        md"""ℹ️ No training in demo mode. Please continue with plotting results.
        """
    end
end

# ╔═╡ 1a608bc8-7264-4dd3-a4e7-0e39128a8375
md"""
> 💡 Playing around with hyperparameters is fun, but keep in mind that this is not a suitable method for finding good hyperparameters in real world engineering. Do a hyperparameter optimization instead.
"""

# ╔═╡ ff106912-d18c-487f-bcdd-7b7af2112cab
md"""
# Results 
Now it's time to find out if it worked! Plotting results makes the notebook slow, so it's deactivated by default. Activate it to plot results of your training.

## Training results
Let's check out the *training* results of the freshly trained neural FMU.
"""

# ╔═╡ 51eeb67f-a984-486a-ab8a-a2541966fa72
begin
    neuralFMU
    MODE
    LIVE_TRAIN
    md"""
    🎬 **Plot results** $(@bind LIVE_RESULTS CheckBox()) 
    """
end

# ╔═╡ 27458e32-5891-4afc-af8e-7afdf7e81cc6
begin

    function plotPaths!(fig, t, x, N; color = :black, label = :none, kwargs...)
        paths = []
        path = nothing
        lastN = N[1]
        for i = 1:length(N)
            if N[i] == 0.0
                if lastN == 1.0
                    push!(path, (t[i], x[i]))
                    push!(paths, path)
                end
            end

            if N[i] == 1.0
                if lastN == 0.0
                    path = []
                end
                push!(path, (t[i], x[i]))
            end

            lastN = N[i]
        end
        if length(path) > 0
            push!(paths, path)
        end

        isfirst = true
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in path),
                collect(v[2] for v in path);
                label = isfirst ? label : :none,
                color = color,
                kwargs...,
            )
            isfirst = false
        end

        return fig
    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ 737e2c50-0858-4205-bef3-f541e33b85c3
md"""
### FMU
Simulating the FMU (training data):
"""

# ╔═╡ 5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
begin
    fmu_train = fmiSimulate(
        fmu,
        (data_train.t[1], data_train.t[end]);
        x0 = x0,
        parameters = Dict{String,Any}("fileName" => data_train.params["fileName"]),
        recordValues = [
            "rRPositionControl_Elasticity.tCP.p_x",
            "rRPositionControl_Elasticity.tCP.p_y",
            "rRPositionControl_Elasticity.tCP.N",
            "rRPositionControl_Elasticity.tCP.a_x",
            "rRPositionControl_Elasticity.tCP.a_y",
        ],
        showProgress = true,
        maxiters = 1e6,
        saveat = data_train.t,
        solver = Tsit5(),
    )
    nothing
end

# ╔═╡ 4f27b6c0-21da-4e26-aaad-ff453c8af3da
md"""
### Neural FMU
Simulating the neural FMU (training data):
"""

# ╔═╡ 1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
begin
    if LIVE_RESULTS
        result_train = neuralFMU(
            x0,
            (data_train.t[1], data_train.t[end]);
            parameters = Dict{String,Any}("fileName" => data_train.params["fileName"]),
            recordValues = [
                "rRPositionControl_Elasticity.tCP.p_x",
                "rRPositionControl_Elasticity.tCP.p_y",
                "rRPositionControl_Elasticity.tCP.N",
                "rRPositionControl_Elasticity.tCP.v_x",
                "rRPositionControl_Elasticity.tCP.v_y",
            ],
            showProgress = true,
            maxiters = 1e6,
            saveat = data_train.t,
        )
        nothing
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ b0ce7b92-93e0-4715-8324-3bf4ff42a0b3
let
    if LIVE_RESULTS
        loss_fmu = loss(fmu_train, data_train)
        loss_nfmu = loss(result_train, data_train)

        md"""
      #### The word `train`
      The loss function value of the FMU on training data is $(round(loss_fmu; digits=6)), of the neural FMU it is $(round(loss_nfmu; digits=6)). The neural FMU is about $(round(loss_fmu/loss_nfmu; digits=1)) times more accurate.
      """
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 919419fe-35de-44bb-89e4-8f8688bee962
let
    if LIVE_RESULTS
        fig = plot(; dpi = 300, size = (200 * 3, 60 * 3))
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ ed25a535-ca2f-4cd2-b0af-188e9699f1c3
md"""
#### The letter `a`
"""

# ╔═╡ 2918daf2-6499-4019-a04b-8c3419ee1ab7
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (40 * 10, 40 * 10),
            xlims = (0.165, 0.205),
            ylims = (-0.035, 0.005),
        )
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ d798a5d0-3017-4eab-9cdf-ee85d63dfc49
md"""
#### The letter `n`
"""

# ╔═╡ 048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (50 * 10, 40 * 10),
            xlims = (0.245, 0.295),
            ylims = (-0.04, 0.0),
        )
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ b489f97d-ee90-48c0-af06-93b66a1f6d2e
md"""
## Validation results
Let's check out the *validation* results of the freshly trained neural FMU.
"""

# ╔═╡ 4dad3e55-5bfd-4315-bb5a-2680e5cbd11c
md"""
### FMU
Simulating the FMU (validation data):
"""

# ╔═╡ ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
begin
    fmu_validation = fmiSimulate(
        fmu,
        (data_validation.t[1], data_validation.t[end]);
        x0 = x0,
        parameters = Dict{String,Any}("fileName" => data_validation.params["fileName"]),
        recordValues = [
            "rRPositionControl_Elasticity.tCP.p_x",
            "rRPositionControl_Elasticity.tCP.p_y",
            "rRPositionControl_Elasticity.tCP.N",
        ],
        showProgress = true,
        maxiters = 1e6,
        saveat = data_validation.t,
        solver = Tsit5(),
    )
    nothing
end

# ╔═╡ 35f52dbc-0c0b-495e-8fd4-6edbc6fa811e
md"""
### Neural FMU
Simulating the neural FMU (validation data):
"""

# ╔═╡ 51aed933-2067-4ea8-9c2f-9d070692ecfc
begin
    if LIVE_RESULTS
        result_validation = neuralFMU(
            x0,
            (data_validation.t[1], data_validation.t[end]);
            parameters = Dict{String,Any}("fileName" => data_validation.params["fileName"]),
            recordValues = [
                "rRPositionControl_Elasticity.tCP.p_x",
                "rRPositionControl_Elasticity.tCP.p_y",
                "rRPositionControl_Elasticity.tCP.N",
            ],
            showProgress = true,
            maxiters = 1e6,
            saveat = data_validation.t,
        )
        nothing
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 8d9dc86e-f38b-41b1-80c6-b2ab6f488a3a
begin
    if LIVE_RESULTS
        loss_fmu = loss(fmu_validation, data_validation)
        loss_nfmu = loss(result_validation, data_validation)
        md"""
      #### The word `validate`
      The loss function value of the FMU on validation data is $(round(loss_fmu; digits=6)), of the neural FMU it is $(round(loss_nfmu; digits=6)). The neural FMU is about $(round(loss_fmu/loss_nfmu; digits=1)) times more accurate.
      """
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 74ef5a39-1dd7-404a-8baf-caa1021d3054
let
    if LIVE_RESULTS
        fig = plot(; dpi = 300, size = (200 * 3, 40 * 3))
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 347d209b-9d41-48b0-bee6-0d159caacfa9
md"""
#### The letter `d`
"""

# ╔═╡ 05281c4f-dba8-4070-bce3-dc2f1319902e
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (35 * 10, 50 * 10),
            xlims = (0.188, 0.223),
            ylims = (-0.025, 0.025),
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 590d7f24-c6b6-4524-b3db-0c93d9963b74
md"""
#### The letter `t`
"""

# ╔═╡ 67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (25 * 10, 50 * 10),
            xlims = (0.245, 0.27),
            ylims = (-0.025, 0.025),
            legend = :topleft,
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ e6dc8aab-82c1-4dc9-a1c8-4fe9c137a146
md"""
#### The letter `e`
"""

# ╔═╡ dfee214e-bd13-4d4f-af8e-20e0c4e0de9b
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (25 * 10, 30 * 10),
            xlims = (0.265, 0.29),
            ylims = (-0.025, 0.005),
            legend = :topleft,
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 88884204-79e4-4412-b861-ebeb5f6f7396
md""" 
# Conclusion
Hopefully you got a good first insight in the topic hybrid modeling using FMI and collected your first sense of achievement. Did you find a nice optimum? In case you don't, some rough hyper parameters are given below.

## Hint
If your results are not *that* promising, here is a set of hyperparameters to check. It is *not* a optimal set of parameters, but a *good* set, so feel free to explore the *best*!

| Parameter | Value |
| ----- | ----- |
| eta | 1e-3 |
| layer count | 3 |
| layer width | 32 |
| initial gate opening | 0.2 |
| batch element length | 0.05s |
| training steps | $\geq$ 10 000 |
| additional variables | Joint 1 Angle $br Joint 2 Angle $br TCP velocity x $br TCP velocity y $br TCP nominal force |

## Citation
If you find this workshop useful for your own work and/or research, please cite our related publication:

Tobias Thummerer, Johannes Stoljar and Lars Mikelsons. 2022. **NeuralFMU: presenting a workflow for integrating hybrid neuralODEs into real-world applications.** Electronics 11, 19, 3202. DOI: 10.3390/electronics11193202

## Acknowlegments
- the FMU was created using the excellent Modelica library *Servomechanisms* $br (https://github.com/afrhu/Servomechanisms)
- the linked YouTube video in the introduction is by *Alexandru Babaian* $br (https://www.youtube.com/watch?v=ryIwLLr6yRA)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
FMI = "14a09403-18e3-468f-ad8a-74f8dda2d9ac"
FMIFlux = "fabad875-0d53-4e47-9446-963b74cae21f"
FMIZoo = "724179cf-c260-40a9-bd27-cccc6fe2f195"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.5.0"
FMI = "~0.13.3"
FMIFlux = "~0.12.2"
FMIZoo = "~0.3.3"
JLD2 = "~0.4.49"
PlotlyJS = "~0.18.13"
Plots = "~1.40.5"
PlutoUI = "~0.7.59"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.7"
manifest_format = "2.0"
project_hash = "79772b37e2cae2421c7159b63f3cbe881b42eaeb"

[[deps.ADTypes]]
git-tree-sha1 = "016833eb52ba2d6bea9fcb50ca295980e728ee24"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.7"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ed2ec3c9b483842ae59cd273834e5b46206d6dda"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.11.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "600078184f7de14b3e60efe13fc0ba5c59f6dca5"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.0"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "71f605effb24081b09cae943ba39ef9ca90c04f4"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.7.2"
weakdeps = ["SparseArrays"]

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "7aa7ad1682f3d5754e3491bb59b8103cae28e3a3"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.40"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Blink]]
deps = ["Base64", "Distributed", "HTTP", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Pkg", "Reexport", "Sockets", "WebIO"]
git-tree-sha1 = "bc93511973d1f949d45b0ea17878e6cb0ad484a1"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.9"

[[deps.BoundaryValueDiffEq]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "BandedMatrices", "ConcreteStructs", "DiffEqBase", "FastAlmostBandedMatrices", "FastClosures", "ForwardDiff", "LinearAlgebra", "LinearSolve", "Logging", "NonlinearSolve", "OrdinaryDiffEq", "PreallocationTools", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "Setfield", "SparseArrays", "SparseDiffTools"]
git-tree-sha1 = "005b55fa2eebaa4d7bf3cfb8097807f47116175f"
uuid = "764a87c0-6b3e-53db-9096-fe964310641d"
version = "5.7.1"

    [deps.BoundaryValueDiffEq.extensions]
    BoundaryValueDiffEqODEInterfaceExt = "ODEInterface"

    [deps.BoundaryValueDiffEq.weakdeps]
    ODEInterface = "54ca160b-1b9f-5127-a996-1867f4bc2a2c"

[[deps.BufferedStreams]]
git-tree-sha1 = "4ae47f9a4b1dc19897d3743ff13685925c5202ec"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.Cassette]]
git-tree-sha1 = "0970356c3bb9113309c74c27c87083cf9c73880a"
uuid = "7057c7e9-c182-5462-911a-8362d720325c"
version = "0.3.13"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "227985d885b4dbce5e18a96f9326ea1e836e5a03"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.69.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelayDiffEq]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "LinearAlgebra", "Logging", "OrdinaryDiffEq", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SimpleUnPack"]
git-tree-sha1 = "5959ae76ebd198f70e9af81153644543da0cfaf2"
uuid = "bcd4f6db-9728-5f36-b5f7-82caef46ccdb"
version = "5.47.3"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PreallocationTools", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Static", "StaticArraysCore", "Statistics", "Tricks", "TruncatedStacktraces"]
git-tree-sha1 = "03b9555f4c3a7c2f530bb1ae13e85719c632f74e"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.151.1"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDistributionsExt = "Distributions"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "Functors", "LinearAlgebra", "Markdown", "NLsolve", "Parameters", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "ee954c8b9d348b7a8a6aec5f28288bf5adecd4ee"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "2.37.0"
weakdeps = ["OrdinaryDiffEq", "Sundials"]

[[deps.DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "GPUArraysCore", "LinearAlgebra", "Markdown", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "Requires", "ResettableStacks", "SciMLBase", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "65cbbe1450ced323b4b17228ccd96349d96795a7"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.21.0"
weakdeps = ["ReverseDiff"]

    [deps.DiffEqNoiseProcess.extensions]
    DiffEqNoiseProcessReverseDiffExt = "ReverseDiff"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiableEigen]]
deps = ["ForwardDiffChainRules", "LinearAlgebra", "ReverseDiff"]
git-tree-sha1 = "6370fca72115d68efc500b3f49ecd627b715fda8"
uuid = "73a20539-4e65-4dcb-a56d-dc20f210a01b"
version = "0.2.0"

[[deps.DifferentiableFlatten]]
deps = ["ChainRulesCore", "LinearAlgebra", "NamedTupleTools", "OrderedCollections", "Requires", "SparseArrays"]
git-tree-sha1 = "f4dc2c1d994c7e2e602692a7dadd2ac79212c3a9"
uuid = "c78775a3-ee38-4681-b694-0504db4f5dc7"
version = "0.1.1"

[[deps.DifferentialEquations]]
deps = ["BoundaryValueDiffEq", "DelayDiffEq", "DiffEqBase", "DiffEqCallbacks", "DiffEqNoiseProcess", "JumpProcesses", "LinearAlgebra", "LinearSolve", "NonlinearSolve", "OrdinaryDiffEq", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SteadyStateDiffEq", "StochasticDiffEq", "Sundials"]
git-tree-sha1 = "8864b6a953eeba7890d23258aca468d90ca73fd6"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "7.12.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EllipsisNotation]]
deps = ["StaticArrayInterface"]
git-tree-sha1 = "3507300d4343e8e4ad080ad24e335274c2e297a9"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.8.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random"]
git-tree-sha1 = "3fb48f9c18de1993c477457265b85130756746ae"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.11.20"
weakdeps = ["SpecialFunctions"]

    [deps.Enzyme.extensions]
    EnzymeSpecialFunctionsExt = "SpecialFunctions"

[[deps.EnzymeCore]]
git-tree-sha1 = "1bc328eec34ffd80357f84a84bb30e4374e9bd60"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.6.6"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "32d418c804279c60dd38ac7868126696f3205a4f"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.102+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.ExponentialUtilities]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "GenericSchur", "LinearAlgebra", "PrecompileTools", "Printf", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "8e18940a5ba7f4ddb41fe2b79b6acaac50880a86"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.26.1"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "380053d61bb9064d6aa4a9777413b40429c79901"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.2.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FMI]]
deps = ["DifferentialEquations", "Downloads", "FMIExport", "FMIImport", "LinearAlgebra", "ProgressMeter", "Requires", "ThreadPools"]
git-tree-sha1 = "476e0317e86ecb2702d2ad744eb3a02626674fb1"
uuid = "14a09403-18e3-468f-ad8a-74f8dda2d9ac"
version = "0.13.3"

[[deps.FMICore]]
deps = ["ChainRulesCore", "Dates", "Requires"]
git-tree-sha1 = "ba0fb5ec972d3d2bb7f8c1ebd2f84a58268ab30b"
uuid = "8af89139-c281-408e-bce2-3005eb87462f"
version = "0.20.1"

[[deps.FMIExport]]
deps = ["Dates", "EzXML", "FMICore", "UUIDs"]
git-tree-sha1 = "daf601e31aeb73fb0010fe7ff942367bc75d5088"
uuid = "31b88311-cab6-44ed-ba9c-fe5a9abbd67a"
version = "0.3.2"

[[deps.FMIFlux]]
deps = ["Colors", "DifferentiableEigen", "DifferentialEquations", "FMIImport", "FMISensitivity", "Flux", "Optim", "Printf", "ProgressMeter", "Requires", "Statistics", "ThreadPools"]
git-tree-sha1 = "1315f3bfe3e273eb35ea872d71869814349541cd"
uuid = "fabad875-0d53-4e47-9446-963b74cae21f"
version = "0.12.2"

[[deps.FMIImport]]
deps = ["Downloads", "EzXML", "FMICore", "Libdl", "RelocatableFolders", "ZipFile"]
git-tree-sha1 = "b5b245bf7f1fc044ad16b016c7e2f08a2333a6f1"
uuid = "9fcbc62e-52a0-44e9-a616-1359a0008194"
version = "0.16.4"

[[deps.FMISensitivity]]
deps = ["FMICore", "ForwardDiffChainRules", "SciMLSensitivity"]
git-tree-sha1 = "43b9b68262af5d3602c9f153e978aff00b849569"
uuid = "3e748fe5-cd7f-4615-8419-3159287187d2"
version = "0.1.4"

[[deps.FMIZoo]]
deps = ["Downloads", "EzXML", "FilePaths", "FilePathsBase", "Glob", "Interpolations", "MAT", "Optim", "Requires", "ZipFile"]
git-tree-sha1 = "47f7e240ab988c1a24cc028f668eb70f73af5bd3"
uuid = "724179cf-c260-40a9-bd27-cccc6fe2f195"
version = "0.3.3"

[[deps.FastAlmostBandedMatrices]]
deps = ["ArrayInterface", "ArrayLayouts", "BandedMatrices", "ConcreteStructs", "LazyArrays", "LinearAlgebra", "MatrixFactorizations", "PrecompileTools", "Reexport"]
git-tree-sha1 = "a92b5820ea38da3b50b626cc55eba2b074bb0366"
uuid = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
version = "0.1.3"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "a6e756a880fc419c8b41592010aebe6a5ce09136"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.2.8"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastLapackInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cbf5edddb61a43669710cbc2241bc08b36d9e660"
uuid = "29a986be-02c6-4525-aec4-84b980013641"
version = "2.0.4"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "edacf029ed6276301e455e34d7ceeba8cc34078a"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.16"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.ForwardDiffChainRules]]
deps = ["ChainRulesCore", "DifferentiableFlatten", "ForwardDiff", "MacroTools"]
git-tree-sha1 = "088aae09132ee3a8b351bec6569e83985e5b961e"
uuid = "c9556dd2-1aed-4cfe-8560-1557cf593001"
version = "0.2.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.FunctionProperties]]
deps = ["Cassette", "DiffRules"]
git-tree-sha1 = "bf7c740307eb0ee80e05d8aafbd0c5a901578398"
uuid = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
version = "0.1.2"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8a66c07630d6428eaab3506a0eabfcf4a9edea05"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.11"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "5c9de6d5af87acd2cf719e214ed7d51e14017b7a"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.2.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "a846f297ce9d09ccba02ead0cae70690e072a119"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.25.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "3e527447a45901ea392fe12120783ad6ec222803"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.6"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "182c478a179b267dd7a741b6f8f4c3e0803795d6"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.6+0"

[[deps.GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "af49a0851f8113fcfae2ef5027c6d49d0acec39b"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.4"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "334d300809ae0a68ceee3444c6e99ded412bf0b3"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "38c8874692d48d5440d5752d6c74b0c6b0b60739"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.2+1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1d334207121865ac8c1c97eb7f42d0339e4635bf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.0+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "e7cbed5032c4c397a6ac23d1493f3289e01231c4"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.14"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Reexport", "Requires", "TranscodingStreams", "UUIDs", "Unicode"]
git-tree-sha1 = "84642bc18a79d715b39d3724b03cbdd2e7d48c62"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.49"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.JumpProcesses]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "Markdown", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SciMLBase", "StaticArrays", "SymbolicIndexingInterface", "UnPack"]
git-tree-sha1 = "ed08d89318be7d625613f3c435d1f6678fba4850"
uuid = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
version = "9.11.1"
weakdeps = ["FastBroadcast"]

    [deps.JumpProcesses.extensions]
    JumpProcessFastBroadcastExt = "FastBroadcast"

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "07649c499349dad9f08dde4243a4c597064663e9"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.6.0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "ed7167240f40e62d97c1f5f7735dea6de3cc5c49"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.18"
weakdeps = ["EnzymeCore"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "267dad6b4b7b5d529c76d40ff48d33f7e94cb834"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "839c82932db86740ae729779e610f07a1640be9a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.6.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "88b916503aac4fb7f701bb625cd84ca5dd1677bc"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.29+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays"]
git-tree-sha1 = "35079a6a869eecace778bcda8641f9a54ca3a828"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "1.10.0"
weakdeps = ["StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysStaticArraysExt = "StaticArrays"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LevyArea]]
deps = ["LinearAlgebra", "Random", "SpecialFunctions"]
git-tree-sha1 = "56513a09b8e0ae6485f34401ea9e2f31357958ec"
uuid = "2d8b4e74-eb68-11e8-0fb9-d5eb67b50637"
version = "1.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "b2e2dba60642e07c062eb3143770d7e234316772"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "2.30.2"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveEnzymeExt = ["Enzyme", "EnzymeCore"]
    LinearSolveFastAlmostBandedMatricesExt = ["FastAlmostBandedMatrices"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolveRecursiveArrayToolsExt = "RecursiveArrayTools"

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "4099bb6809ac109bfc17d521dad33763bcf026b7"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.1+1"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8c35d5420193841b2f367e658540e8d9e0601ed0"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.4.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "6731e0574fa5ee21c02733e397beb133df90de35"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "2.2.0"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "1b9e613f2ca3b6cdcbfe36381e17ca2b66d4b3a1"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.3"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "a7cefa21a2ff993bff0456bf7521f46fc077ddf1"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.19"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "MbedTLS", "Pkg", "Sockets"]
git-tree-sha1 = "7295d849103ac4fcbe3b2e439f229c5cc77b9b69"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "1.0.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "78de319bce99d1d8c1d4fe5401f7cfc2627df396"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.18"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "LazyArrays", "LineSearches", "LinearAlgebra", "LinearSolve", "MaybeInplace", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SparseArrays", "SparseDiffTools", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "dc0d78eeed89323526203b8a11a4fa6cdbe25cd6"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "3.11.0"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveBandedMatricesExt = "BandedMatrices"
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = "NLsolve"
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSymbolicsExt = "Symbolics"
    NonlinearSolveZygoteExt = "Zygote"

    [deps.NonlinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "195e0a19842f678dd3473ceafbe9d82dfacc583c"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.1"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "963a3f28a2e65bb87a68033ea4a616002406037d"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.5"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "a9de2f1fc98b92f8856c640bf4aec1ac9b2a0d86"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.3+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a028ee3cb5641cccc4c24e90c36b0a4f7707bdf5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6572fe0c5b74431aaeb0b18a4aa5ef03c84678be"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.3"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.OrdinaryDiffEq]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "ExponentialUtilities", "FastBroadcast", "FastClosures", "FillArrays", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "IfElse", "InteractiveUtils", "LineSearches", "LinearAlgebra", "LinearSolve", "Logging", "MacroTools", "MuladdMacro", "NonlinearSolve", "Polyester", "PreallocationTools", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "SimpleNonlinearSolve", "SimpleUnPack", "SparseArrays", "SparseDiffTools", "StaticArrayInterface", "StaticArrays", "TruncatedStacktraces"]
git-tree-sha1 = "75b0d2bf28d0df92931919004a5be5304c38cca2"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.80.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "PlotlyKaleido", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "e62d886d33b81c371c9d4e2f70663c0637f19459"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.13"

    [deps.PlotlyJS.extensions]
    CSVExt = "CSV"
    DataFramesExt = ["DataFrames", "CSV"]
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyJS.weakdeps]
    CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyKaleido]]
deps = ["Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "2650cd8fb83f73394996d507b3411a7316f6f184"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "082f0c4b70c202c37784ce4bfbc33c9f437685bf"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.5"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PoissonRandom]]
deps = ["Random"]
git-tree-sha1 = "a0f1159c33f846aa77c3f30ebbc69795e5327152"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.4"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "9ff799e8fb8ed6717710feee3be3bc20645daa97"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.15"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "406c29a7f46706d379a3bce45671b4e3a39ddfbc"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.22"
weakdeps = ["ReverseDiff"]

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "3400ce27995422fb88ffcd3af9945565aad947f0"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.23.1"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "6db1a75507051bc18bfa131fbc7c3f169cc4b2f6"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.23"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "cc6cd622481ea366bb9067859446a8b01d92b468"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "7a6c5c8c38d2e37f45d4686c3598c20c1aebf48e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.41.3"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Setfield", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "10499f619ef6e890f3f4a38914481cc868689cd5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.8"

[[deps.SciMLSensitivity]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "DiffEqBase", "DiffEqCallbacks", "DiffEqNoiseProcess", "Distributions", "EllipsisNotation", "Enzyme", "FiniteDiff", "ForwardDiff", "FunctionProperties", "FunctionWrappersWrappers", "Functors", "GPUArraysCore", "LinearAlgebra", "LinearSolve", "Markdown", "OrdinaryDiffEq", "Parameters", "PreallocationTools", "QuadGK", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "ReverseDiff", "SciMLBase", "SciMLOperators", "SparseDiffTools", "StaticArrays", "StaticArraysCore", "Statistics", "StochasticDiffEq", "Tracker", "TruncatedStacktraces", "Zygote"]
git-tree-sha1 = "3b0fde1944502bd736bcdc3d0df577dddb54d189"
uuid = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
version = "7.51.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "cfdd1200d150df1d3c055cc72ee6850742e982d7"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.4.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "DiffResults", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "Reexport", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "c020028bb22a2f23cbd88cb92cf47cbb8c98513f"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "1.8.0"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolvePolyesterForwardDiffExt = "PolyesterForwardDiff"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveStaticArraysExt = "StaticArrays"
    SimpleNonlinearSolveTrackerExt = "Tracker"
    SimpleNonlinearSolveZygoteExt = "Zygote"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseDiffTools]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "PackageExtensionCompat", "Random", "Reexport", "SciMLOperators", "Setfield", "SparseArrays", "StaticArrayInterface", "StaticArrays", "Tricks", "UnPack", "VertexSafeGraphs"]
git-tree-sha1 = "cce98ad7c896e52bb0eded174f02fc2a29c38477"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "2.18.0"

    [deps.SparseDiffTools.extensions]
    SparseDiffToolsEnzymeExt = "Enzyme"
    SparseDiffToolsPolyesterExt = "Polyester"
    SparseDiffToolsPolyesterForwardDiffExt = "PolyesterForwardDiff"
    SparseDiffToolsSymbolicsExt = "Symbolics"
    SparseDiffToolsZygoteExt = "Zygote"

    [deps.SparseDiffTools.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.Sparspak]]
deps = ["Libdl", "LinearAlgebra", "Logging", "OffsetArrays", "Printf", "SparseArrays", "Test"]
git-tree-sha1 = "342cf4b449c299d8d1ceaf00b7a49f4fbc7940e7"
uuid = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
version = "0.3.9"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "8963e5a083c837531298fc41599182a759a87a6d"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "20833c5b7f7edf0e5026f23db7f268e4f23ec577"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.6"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.SteadyStateDiffEq]]
deps = ["ConcreteStructs", "DiffEqBase", "DiffEqCallbacks", "LinearAlgebra", "Reexport", "SciMLBase"]
git-tree-sha1 = "a735fd5053724cf4de31c81b4e2cc429db844be5"
uuid = "9672c7b4-1e72-59bd-8a11-6ac3964bc41f"
version = "2.0.1"

[[deps.StochasticDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqNoiseProcess", "DocStringExtensions", "FiniteDiff", "ForwardDiff", "JumpProcesses", "LevyArea", "LinearAlgebra", "Logging", "MuladdMacro", "NLsolve", "OrdinaryDiffEq", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "97e5d0b7e5ec2e68eec6626af97c59e9f6b6c3d0"
uuid = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"
version = "6.65.1"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
deps = ["Test"]
git-tree-sha1 = "010dc73c7146869c042b49adcdb6bf528c12e859"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.Sundials]]
deps = ["CEnum", "DataStructures", "DiffEqBase", "Libdl", "LinearAlgebra", "Logging", "PrecompileTools", "Reexport", "SciMLBase", "SparseArrays", "Sundials_jll"]
git-tree-sha1 = "e15f5a73f0d14b9079b807a9d1dac13e4302e997"
uuid = "c3572dad-4567-51f8-b174-8c6c989267f4"
version = "4.24.0"

[[deps.Sundials_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "SuiteSparse_jll", "libblastrampoline_jll"]
git-tree-sha1 = "ba4d38faeb62de7ef47155ed321dce40a549c305"
uuid = "fb77eaff-e24c-56d4-86b1-d163f2edb164"
version = "5.2.2+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "a5f6f138b740c9d93d76f0feddd3092e6ef002b7"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.22"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.Tracker]]
deps = ["Adapt", "ChainRulesCore", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5158100ed55411867674576788e710a815a0af02"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.34"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
git-tree-sha1 = "d73336d81cafdc277ff45558bb7eaa2b04a8e472"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.10"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "3064e780dbb8a9296ebb3af8f440f787bb5332af"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.80"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "be986ad9dac14888ba338c2554dcfec6939e1393"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.2.1"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "dd260903fdabea27d9b6021689b3cd5401a57748"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.20.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "bf2c553f25e954a9b38c9c0593a59bb13113f9e5"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.5"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "e7f5b81c65eb858bed630fe006837b935518aca5"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.70"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "0eef0765186f7452e52236fa42ca8c9b3c11c6e3"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.21"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "4162e95e05e79922e44b9952ccbc262832e4ad07"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.6.0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "d9717ce3518dc68a99e6b96300813760d887a01d"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.1+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "19c586905e78a26f7e4e97f81716057bd6b1bc54"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.70"
weakdeps = ["Colors", "Distances", "Tracker"]

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─1470df0f-40e1-45d5-a4cc-519cc3b28fb8
# ╟─7d694be0-cd3f-46ae-96a3-49d07d7cf65a
# ╟─10cb63ad-03d7-47e9-bc33-16c7786b9f6a
# ╟─1e0fa041-a592-42fb-bafd-c7272e346e46
# ╟─6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
# ╟─8a82d8c7-b781-4600-8780-0a0a003b676c
# ╟─a02f77d1-00d2-46a3-91ba-8a7f5b4bbdc9
# ╠═a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
# ╟─02f0add7-9c4e-4358-8b5e-6863bae3ee75
# ╠═72604eef-5951-4934-844d-d2eb7eb0292c
# ╠═21104cd1-9fe8-45db-9c21-b733258ff155
# ╠═9d9e5139-d27e-48c8-a62e-33b2ae5b0086
# ╟─85308992-04c4-4d20-a840-6220cab54680
# ╠═eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
# ╠═98c608d9-c60e-4eb6-b611-69d2ae7054c9
# ╟─3e2579c2-39ce-4249-ad75-228f82e616da
# ╠═ddc9ce37-5f93-4851-a74f-8739b38ab092
# ╟─93fab704-a8dd-47ec-ac88-13f32be99460
# ╠═de7a4639-e3b8-4439-924d-7d801b4b3eeb
# ╟─5cb505f7-01bd-4824-8876-3e0f5a922fb7
# ╠═45c4b9dd-0b04-43ae-a715-cd120c571424
# ╠═33d648d3-e66e-488f-a18d-e538ebe9c000
# ╟─1e9541b8-5394-418d-8c27-2831951c538d
# ╠═e6e91a22-7724-46a3-88c1-315c40660290
# ╟─44500f0a-1b89-44af-b135-39ce0fec5810
# ╟─33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
# ╟─74d23661-751b-4371-bf6b-986149124e81
# ╠═c88b0627-2e04-40ab-baa2-b4c1edfda0c3
# ╟─915e4601-12cc-4b7e-b2fe-574e116f3a92
# ╟─f8e40baa-c1c5-424a-9780-718a42fd2b67
# ╠═74289e0b-1292-41eb-b13b-a4a5763c72b0
# ╟─f111e772-a340-4217-9b63-e7715f773b2c
# ╟─92ad1a99-4ad9-4b69-b6f3-84aab49db54f
# ╟─909de9f1-2aca-4bf0-ba60-d3418964ba4a
# ╟─d8ca5f66-4f55-48ab-a6c9-a0be662811d9
# ╠═41b1c7cb-5e3f-4074-a681-36dd2ef94454
# ╠═8f45871f-f72a-423f-8101-9ce93e5a885b
# ╠═57c039f7-5b24-4d63-b864-d5f808110b91
# ╟─4510022b-ad28-4fc2-836b-e4baf3c14d26
# ╠═9589416a-f9b3-4b17-a381-a4f660a5ee4c
# ╟─326ae469-43ab-4bd7-8dc4-64575f4a4d3e
# ╠═8f8f91cc-9a92-4182-8f18-098ae3e2c553
# ╟─8d93a1ed-28a9-4a77-9ac2-5564be3729a5
# ╠═4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# ╟─6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
# ╟─dbde2da3-e3dc-4b78-8f69-554018533d35
# ╠═d42d0beb-802b-4d30-b5b8-683d76af7c10
# ╟─e50d7cc2-7155-42cf-9fef-93afeee6ffa4
# ╟─3756dd37-03e0-41e9-913e-4b4f183d8b81
# ╠═2f83bc62-5a54-472a-87a2-4ddcefd902b6
# ╟─c228eb10-d694-46aa-b952-01d824879287
# ╟─16ffc610-3c21-40f7-afca-e9da806ea626
# ╠═052f2f19-767b-4ede-b268-fce0aee133ad
# ╟─746fbf6f-ed7c-43b8-8a6f-0377cd3cf85e
# ╟─08e1ff54-d115-4da9-8ea7-5e89289723b3
# ╟─70c6b605-54fa-40a3-8bce-a88daf6a2022
# ╠═634f923a-5e09-42c8-bac0-bf165ab3d12a
# ╟─f59b5c84-2eae-4e3f-aaec-116c090d454d
# ╠═0c9493c4-322e-41a0-9ec7-2e2c54ae1373
# ╟─325c3032-4c78-4408-b86e-d9aa4cfc3187
# ╠═25e55d1c-388f-469d-99e6-2683c0508693
# ╟─74c519c9-0eef-4798-acff-b11044bb4bf1
# ╟─786c4652-583d-43e9-a101-e28c0b6f64e4
# ╟─5d688c3d-b5e3-4a3a-9d91-0896cc001000
# ╠═2e08df84-a468-4e99-a277-e2813dfeae5c
# ╟─68719de3-e11e-4909-99a3-5e05734cc8b1
# ╟─b42bf3d8-e70c-485c-89b3-158eb25d8b25
# ╟─c446ed22-3b23-487d-801e-c23742f81047
# ╠═fc3d7989-ac10-4a82-8777-eeecd354a7d0
# ╟─0a7955e7-7c1a-4396-9613-f8583195c0a8
# ╟─4912d9c9-d68d-4afd-9961-5d8315884f75
# ╟─19942162-cd4e-487c-8073-ea6b262d299d
# ╟─73575386-673b-40cc-b3cb-0b8b4f66a604
# ╟─24861a50-2319-4c63-a800-a0a03279efe2
# ╟─93735dca-c9f3-4f1a-b1bd-dfe312a0644a
# ╟─13ede3cd-99b1-4e65-8a18-9043db544728
# ╟─f7c119dd-c123-4c43-812e-d0625817d77e
# ╟─f4e66f76-76ff-4e21-b4b5-c1ecfd846329
# ╟─b163115b-393d-4589-842d-03859f05be9a
# ╟─ac0afa6c-b6ec-4577-aeb6-10d1ec63fa41
# ╟─5e9cb956-d5ea-4462-a649-b133a77929b0
# ╟─9dc93971-85b6-463b-bd17-43068d57de94
# ╟─476a1ed7-c865-4878-a948-da73d3c81070
# ╟─0b6b4f6d-be09-42f3-bc2c-5f17a8a9ab0e
# ╟─a1aca180-d561-42a3-8d12-88f5a3721aae
# ╟─3bc2b859-d7b1-4b79-88df-8fb517a6929d
# ╟─a501d998-6fd6-496f-9718-3340c42b08a6
# ╟─83a2122d-56da-4a80-8c10-615a8f76c4c1
# ╟─e342be7e-0806-4f72-9e32-6d74ed3ed3f2
# ╟─eaf37128-0377-42b6-aa81-58f0a815276b
# ╟─c030d85e-af69-49c9-a7c8-e490d4831324
# ╟─51c200c9-0de3-4e50-8884-49fe06158560
# ╟─0dadd112-3132-4491-9f02-f43cf00aa1f9
# ╟─5c2308d9-6d04-4b38-af3b-6241da3b6871
# ╟─bf6bf640-54bc-44ef-bd4d-b98e934d416e
# ╟─639889b3-b9f2-4a3c-999d-332851768fd7
# ╟─007d6d95-ad85-4804-9651-9ac3703d3b40
# ╟─ed1887df-5079-4367-ab04-9d02a1d6f366
# ╟─0b0c4650-2ce1-4879-9acd-81c16d06700e
# ╟─b864631b-a9f3-40d4-a6a8-0b57a37a476d
# ╟─0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
# ╟─95e14ea5-d82d-4044-8c68-090d74d95a61
# ╟─2fa1821b-aaec-4de4-bfb4-89560790dc39
# ╟─cbae6aa4-1338-428c-86aa-61d3304e33ed
# ╟─9b52a65a-f20c-4387-aaca-5292a92fb639
# ╟─8c56acd6-94d3-4cbc-bc29-d249740268a0
# ╟─845a95c4-9a35-44ae-854c-57432200da1a
# ╟─5a399a9b-32d9-4f93-a41f-8f16a4b102dc
# ╟─fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
# ╟─1cd976fb-db40-4ebe-b40d-b996e16fc213
# ╟─93771b35-4edd-49e3-bed1-a3ccdb7975e6
# ╟─e79badcd-0396-4a44-9318-8c6b0a94c5c8
# ╟─2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
# ╟─4454c8d2-68ed-44b4-adfa-432297cdc957
# ╟─d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
# ╟─06937575-9ab1-41cd-960c-7eef3e8cae7f
# ╟─356b6029-de66-418f-8273-6db6464f9fbf
# ╟─5805a216-2536-44ac-a702-d92e86d435a4
# ╟─68d57a23-68c3-418c-9c6f-32bdf8cafceb
# ╟─53e971d8-bf43-41cc-ac2b-20dceaa78667
# ╟─e8b8c63b-2ca4-4e6a-a801-852d6149283e
# ╟─c0ac7902-0716-4f18-9447-d18ce9081ba5
# ╟─84215a73-1ab0-416d-a9db-6b29cd4f5d2a
# ╟─f9d35cfd-4ae5-4dcd-94d9-02aefc99bdfb
# ╟─bc09bd09-2874-431a-bbbb-3d53c632be39
# ╠═f741b213-a20d-423a-a382-75cae1123f2c
# ╟─f02b9118-3fb5-4846-8c08-7e9bbca9d208
# ╠═91473bef-bc23-43ed-9989-34e62166d455
# ╟─404ca10f-d944-4a9f-addb-05efebb4f159
# ╟─d347d51b-743f-4fec-bed7-6cca2b17bacb
# ╟─d60d2561-51a4-4f8a-9819-898d70596e0c
# ╟─c97f2dea-cb18-409d-9ae8-1d03647a6bb3
# ╟─366abd1a-bcb5-480d-b1fb-7c76930dc8fc
# ╟─7e2ffd6f-19b0-435d-8e3c-df24a591bc55
# ╠═caa5e04a-2375-4c56-8072-52c140adcbbb
# ╟─69657be6-6315-4655-81e2-8edef7f21e49
# ╟─23ad65c8-5723-4858-9abe-750c3b65c28a
# ╟─abc57328-4de8-42d8-9e79-dd4020769dd9
# ╟─e8bae97d-9f90-47d2-9263-dc8fc065c3d0
# ╟─2dce68a7-27ec-4ffc-afba-87af4f1cb630
# ╟─c3f5704b-8e98-4c46-be7a-18ab4f139458
# ╟─1a608bc8-7264-4dd3-a4e7-0e39128a8375
# ╟─ff106912-d18c-487f-bcdd-7b7af2112cab
# ╟─51eeb67f-a984-486a-ab8a-a2541966fa72
# ╟─27458e32-5891-4afc-af8e-7afdf7e81cc6
# ╟─737e2c50-0858-4205-bef3-f541e33b85c3
# ╟─5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
# ╟─4f27b6c0-21da-4e26-aaad-ff453c8af3da
# ╟─1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
# ╟─b0ce7b92-93e0-4715-8324-3bf4ff42a0b3
# ╟─919419fe-35de-44bb-89e4-8f8688bee962
# ╟─ed25a535-ca2f-4cd2-b0af-188e9699f1c3
# ╟─2918daf2-6499-4019-a04b-8c3419ee1ab7
# ╟─d798a5d0-3017-4eab-9cdf-ee85d63dfc49
# ╟─048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
# ╟─b489f97d-ee90-48c0-af06-93b66a1f6d2e
# ╟─4dad3e55-5bfd-4315-bb5a-2680e5cbd11c
# ╟─ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
# ╟─35f52dbc-0c0b-495e-8fd4-6edbc6fa811e
# ╟─51aed933-2067-4ea8-9c2f-9d070692ecfc
# ╟─8d9dc86e-f38b-41b1-80c6-b2ab6f488a3a
# ╟─74ef5a39-1dd7-404a-8baf-caa1021d3054
# ╟─347d209b-9d41-48b0-bee6-0d159caacfa9
# ╟─05281c4f-dba8-4070-bce3-dc2f1319902e
# ╟─590d7f24-c6b6-4524-b3db-0c93d9963b74
# ╟─67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
# ╟─e6dc8aab-82c1-4dc9-a1c8-4fe9c137a146
# ╟─dfee214e-bd13-4d4f-af8e-20e0c4e0de9b
# ╟─88884204-79e4-4412-b861-ebeb5f6f7396
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
