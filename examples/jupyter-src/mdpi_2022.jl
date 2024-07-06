# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# Loading in the required libraries
using FMIFlux       # for NeuralFMUs
using FMI           # import FMUs into Julia 
using FMIZoo        # a collection of demo models, including the VLDM
using FMIFlux.Flux  # Machine Learning in Julia

import FMI.DifferentialEquations: Tsit5     # import the Tsit5-solver
import FMI: FMU2Solution
using JLD2                                  # data format for saving/loading parameters

# plotting
import Plots        # default plotting framework

# for interactive plotting
# import PlotlyJS     # plotting (interactive)
# Plots.plotlyjs()    # actiavte PlotlyJS as default plotting backend

# Let's fix the random seed to make our program determinsitic (ANN layers are initialized indeterminsitic otherwise)
import Random 
Random.seed!(1234)

# we use the Tsit5 solver for ODEs here 
solver = Tsit5();   

# load our FMU (we take one from the FMIZoo.jl, exported with Dymola 2022x)
fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME, logLevel=FMI.FMIImport.FMULogLevelInfo)  # `FMULogLevelInfo` = "Log everything that might be interesting!", default is `FMULogLevelWarn`

# let's have a look on the model meta data
fmiInfo(fmu)

# load data from FMIZoo.jl, gather simulation parameters for FMU
data = FMIZoo.VLDM(:train)
tStart = data.cumconsumption_t[1]
tStop = data.cumconsumption_t[end]
tSave = data.cumconsumption_t

# have a look on the FMU parameters (these are the file paths to the characteristic maps)
data.params

# let's run a simulation from `tStart` to `tStop`, use the parameters we just viewed for the simulation run
resultFMU = fmiSimulate(fmu, (tStart, tStop); parameters=data.params) 
fig = fmiPlot(resultFMU)                                                                        # Plot it, but this is a bit too much, so ...
fig = fmiPlot(resultFMU; stateIndices=6:6)                                                      # ... only plot the state #6 and ...
fig = fmiPlot(resultFMU; stateIndices=6:6, ylabel="Cumulative consumption [Ws]", label="FMU")   # ... add some helpful labels!

# further plot the (measurement) data values `consumption_val` and deviation between measurements `consumption_dev`
Plots.plot!(fig, data.cumconsumption_t, data.cumconsumption_val; label="Data", ribbon=data.cumconsumption_dev, fillalpha=0.3)

for i in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    println("tanh($(i)) = $(tanh(i))")
end

# variable we want to manipulate - why we are picking exactly these three is shown a few lines later ;-)
manipulatedDerVars = ["der(dynamics.accelerationCalculation.integrator.y)",
                      "der(dynamics.accelerationCalculation.limIntegrator.y)",
                      "der(result.integrator.y)"]
# alternative: manipulatedDerVars = fmu.modelDescription.derivativeValueReferences[4:6]

# reference simulation to record the derivatives 
resultFMU = fmiSimulate(fmu, (tStart, tStop), parameters=data.params, recordValues=:derivatives, saveat=tSave) 
vals = fmiGetSolutionValue(resultFMU, manipulatedDerVars)

# what happens without propper transformation between FMU- and ANN-domain?
Plots.plot(resultFMU.values.t, vals[1,:][1]; label="vehicle velocity");
Plots.plot!(resultFMU.values.t, tanh.(vals[1,:][1]); label="tanh(velocity)")

# setup shift/scale layers for pre-processing
preProcess = ShiftScale(vals)

# check what it's doing now ...
testVals = collect(preProcess(collect(val[t] for val in vals))[1] for t in 1:length(resultFMU.values.t))
Plots.plot(resultFMU.values.t, testVals; label="velocity (pre-processed)");
Plots.plot!(resultFMU.values.t, tanh.(testVals); label="tanh(velocity)")

# add some additional "buffer"
preProcess.scale[:] *= 0.5 

# and check again what it's doing now ...
testVals = collect(preProcess(collect(val[t] for val in vals))[1] for t in 1:length(resultFMU.values.t))
Plots.plot(resultFMU.values.t, testVals; label="velocity (pre-processed)");
Plots.plot!(resultFMU.values.t, tanh.(testVals); label="tanh(velocity)")

# ... also check the consumption
testVals = collect(preProcess(collect(val[t] for val in vals))[3] for t in 1:length(resultFMU.values.t))
Plots.plot(resultFMU.values.t, testVals; label="vehicle consumption (pre-processed)");
Plots.plot!(resultFMU.values.t, tanh.(testVals); label="tanh(consumption)")

# setup scale/shift layer (inverse transformation) for post-processing
# we don't an inverse transform for the entire preProcess, only for the 2nd element (acceleration)
postProcess = ScaleShift(preProcess; indices=2:2)

# setup cache layers 
cache = CacheLayer()
cacheRetrieve = CacheRetrieveLayer(cache)

gates = ScaleSum([1.0, 0.0]) # signal from FMU (#1 = 1.0), signal from ANN (#2 = 0.0)

# setup the NeuralFMU topology
net = Chain(x -> fmu(; x=x, dx_refs=:all),      # take `x`, put it into the FMU, retrieve all `dx`
            dx -> cache(dx),                    # cache `dx`
            dx -> dx[4:6],                      # forward only dx[4, 5, 6]
            preProcess,                         # pre-process `dx`
            Dense(3, 32, tanh),                 # Dense Layer 3 -> 32 with `tanh` activasion
            Dense(32, 1, tanh),                 # Dense Layer 32 -> 1 with `tanh` activasion 
            postProcess,                        # post process `dx`
            dx -> cacheRetrieve(5:5, dx),       # dynamics FMU | dynamics ANN
            gates,                              # compute resulting dx from ANN + FMU
            dx -> cacheRetrieve(1:4, dx, 6:6))  # stack together: dx[1,2,3,4] from cache + dx from ANN + dx[6] from cache

# build NeuralFMU
neuralFMU = ME_NeuralFMU(fmu, net, (tStart, tStop), solver; saveat=tSave)
neuralFMU.modifiedState = false # speed optimization (no ANN before the FMU)

# get start state vector from data (FMIZoo)
x0 = FMIZoo.getStateVector(data, tStart)

# simulate and plot the (uninitialized) NeuralFMU
resultNFMU_original = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true) 
fig = fmiPlot(resultNFMU_original; stateIndices=5:5, label="NeuralFMU (original)", ylabel="velocity [m/s]")

# plot the original FMU and data
fmiPlot!(fig, resultFMU; stateIndices=5:5, values=false)
Plots.plot!(fig, data.speed_t, data.speed_val, label="Data")

# prepare training data (array of arrays required)
train_data = collect([d] for d in data.cumconsumption_val)
train_t = data.cumconsumption_t 

# switch to a more efficient execution configuration, allocate only a single FMU instance, see:
# https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NOTHING
c, _ = FMIFlux.prepareSolveFMU(neuralFMU.fmu, nothing, neuralFMU.fmu.type, true, false, false, false, true, data.params; x0=x0)

# batch the data (time, targets), train only on model output index 6, plot batch elements
batch = batchDataSolution(neuralFMU, t -> FMIZoo.getStateVector(data, t), train_t, train_data;
    batchDuration=10.0, indicesModel=6:6, plot=false, parameters=data.params, showProgress=false) # try `plot=true` to show the batch elements, try `showProgress=true` to display simulation progress

# limit the maximum number of solver steps to 1e5 and maximum simulation/training duration to 30 minutes
solverKwargsTrain = Dict{Symbol, Any}(:maxiters => 1e5, :max_execution_duration => 10.0*60.0)

# picks a modified MSE, which weights the last time point MSE with 25% and the remaining element MSE with 75%
# this promotes training a continuous function, even when training on batch elements
function lossFct(solution::FMU2Solution)
    ts = dataIndexForTime(solution.states.t[1])
    te = dataIndexForTime(solution.states.t[end])

    a = fmiGetSolutionState(solution, 6; isIndex=true)
    b = train_data[ts:te]
    return FMIFlux.Losses.mse_last_element_rel(a, b, 0.5)
end

# initialize a "worst error growth scheduler" (updates all batch losses, pick the batch element with largest error increase)
# apply the scheduler after every training step, plot the current status every 25 steps and update all batch element losses every 5 steps
scheduler = LossAccumulationScheduler(neuralFMU, batch, lossFct; applyStep=1, plotStep=25, updateStep=5)
updateScheduler = () -> update!(scheduler)

# defines a loss for the entire batch (accumulate error of batch elements)
batch_loss = p -> FMIFlux.Losses.batch_loss(neuralFMU, batch; 
    showProgress=false, p=p, parameters=data.params, update=true, lossFct=lossFct, logLoss=true, solverKwargsTrain...) # try `showProgress=true` to display simulation progress

# loss for training, take element from the worst element scheduler
loss = p -> FMIFlux.Losses.loss(neuralFMU, batch; 
    showProgress=false, p=p, parameters=data.params, lossFct=lossFct, batchIndex=scheduler.elementIndex, logLoss=false, solverKwargsTrain...) # try `showProgress=true` to display simulation progress

# we start with a slightly opended ANN gate (1%) and a almost completely opened FMU gate (99%)
gates.scale[:] = [0.99, 0.01] 

# gather the parameters from the NeuralFMU
params = FMIFlux.params(neuralFMU)

params[1][end-1] = 0.99
params[1][end] = 0.01

# for training, we use the Adam optimizer with step size 1e-3
optim = Adam(1e-3) 

# let's check the loss we are starting with ...
loss_before = batch_loss(params[1])

# initialize the scheduler 
initialize!(scheduler; parameters=data.params, p=params[1], showProgress=false)

batchLen = length(batch)

# we use ForwardDiff for gradinet determination, because the FMU throws multiple events per time instant (this is not supported by reverse mode AD)
# the chunk_size controls the nuber of forward evaluations of the model (the bigger, the less evaluations)
FMIFlux.train!(loss, neuralFMU, Iterators.repeated((), batchLen), optim; gradient=:ForwardDiff, chunk_size=32, cb=updateScheduler) 
loss_after = batch_loss(params[1])

# save the parameters (so we can use them tomorrow again)
paramsPath = joinpath(@__DIR__, "params_$(scheduler.step)steps.jld2")
fmiSaveParameters(neuralFMU, paramsPath)

# switch back to the default execution configuration, see:
# https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NO_RESET
FMIFlux.finishSolveFMU(neuralFMU.fmu, c, false, true)

# check what had been learned by the NeuralFMU, simulate it ...
resultNFMU_train = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true, recordValues=manipulatedDerVars, maxiters=1e7) # [120s]

# Load parameters 
fmiLoadParameters(neuralFMU, paramsPath)

# are we better?
mse_NFMU = FMIFlux.Losses.mse(data.cumconsumption_val, fmiGetSolutionState(resultNFMU_train, 6; isIndex=true))
mse_FMU  = FMIFlux.Losses.mse(data.cumconsumption_val, fmiGetSolutionState(resultFMU, 6; isIndex=true))

# ... and plot it
fig = fmiPlot(resultNFMU_train; stateIndices=6:6, values=false, stateEvents=false, label="NeuralFMU", title="Training Data");
fmiPlot!(fig, resultFMU; stateIndices=6:6, stateEvents=false, values=false, label="FMU");
Plots.plot!(fig, train_t, data.cumconsumption_val, label="Data", ribbon=data.cumconsumption_dev, fillalpha=0.3)

# clean-up
fmiUnload(fmu) 