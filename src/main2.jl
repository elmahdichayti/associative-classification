using CSV
using JuMP
using MultiJuMP
using CPLEX
using DataFrames

include("functions.jl")
include("functions_MultiJuMP.jl")

dataSet = "titanic"
dataFolder = "../data/"
resultsFolder = "../res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
train, test = createFeatures(dataFolder, dataSet)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
rules = createRules_withMultiJuMP(dataSet, resultsFolder, train)

orderedRules = sortRules_MultiJuMP(dataSet, resultsFolder, train, rules)

recall = getPrecision(orderedRules, test)
accuracy = getPrecision(orderedRules, train)

println("Recall: ", recall)
println("Accuracy: ", accuracy)
