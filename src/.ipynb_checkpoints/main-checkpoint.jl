using CSV
using JuMP
using CPLEX
using DataFrames

include("functions.jl")

dataSet = "titanic"
# dataSet = "haberman"
dataFolder = "../data/"
resultsFolder = "../res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
train, test = createFeatures(dataFolder, dataSet)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
rules = createRules(dataSet, resultsFolder, train)

orderedRules = sortRules(dataSet, resultsFolder, train, rules)

recall = getPrecision(orderedRules, test)
accuracy = getPrecision(orderedRules, train)

println("Recall: ", recall)
println("Accuracy: ", accuracy)
