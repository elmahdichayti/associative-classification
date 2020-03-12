# Tolerance
epsilon = 0.0001

# Take in table "data" the column named "header" and create in table "features" columns which correspond to its binarization according to the values in "intervals"
#
# Attributes:
# - header: column of table "data" that will be binarized
# - intervals: array of values which delimits the binarization (ex : [2, 4, 6, 8] will lead to 3 columns respectively equal to 1 if the value of column "header" is in [2, 3], [4, 5] and [6, 7])
#
# Example:
#  createColumns(:Age, [1, 17, 50, Inf], data, features) will create 3 binary columns in features named "Age1-16", "Age17-49", "Age50-Inf"
function createColumns(header::Symbol, intervals, data::DataFrames.DataFrame, features::DataFrames.DataFrame)
    for i in 1:size(intervals, 1) - 1
        lb = intervals[i]
        ub = intervals[i+1]
        features[Symbol(header, lb, "-", (ub-1))] = ifelse.((data[header] .>= lb) .& (data[header] .< ub), 1, 0) 
        #features[Symbol(header, lb, "-", (ub-1))] = ifelse.(data[header] .< ub, 1, 0) 
    end
end


# Create the train and test tables related to a data set
#
# Note 1: the input file name is: dataFolder/dataSet.csv
# Note 2: the first column of the output tables must correspond to the class of each individual
function createFeatures(dataFolder::String, dataSet::String)

    rawDataPath = dataFolder * dataSet * ".csv"

    if !isfile(rawDataPath)
        println("Error in createFeatures: Input file not found: ", rawDataPath)
        return
    end

    rawData = CSV.read(rawDataPath,  header=true)
    
    trainDataPath = dataFolder * dataSet * "_train.csv"
    testDataPath = dataFolder * dataSet * "_test.csv"

    # If the train or the test file do not exist
    if !isfile(trainDataPath) || !isfile(testDataPath)

        println("=== Creating the features")

        # Create the table that will contain the features
        features = DataFrames.DataFrame()
        
        # Create the features of the titanic data set
        if dataSet == "titanic"
            
            features[:Survived] = rawData[:Survived]
            features[:Sex] = ifelse.(rawData[:Sex] .== "female", 1, 0)
            createColumns(:Age, [0, 11, 18, 22, 27, 33, 40, 66, Inf], rawData, features)
            createColumns(:Fare, [7.91, 14.45, 31, 99, 250, Inf], rawData, features)
            createColumns(:Pclass, [1, 2, 3, Inf], rawData, features)
#             createColumns(Symbol("Siblings/Spouses Aboard"), [0, 1, Inf], rawData, features)
#             createColumns(Symbol("Parents/Children Aboard"), [0, 1, Inf], rawData, features)


        end
        
        if dataSet == "haberman"
            
            features[:SurvivalStatus] = ifelse.(rawData[:SurvivalStatus] .== 2, 1, 0)
            createColumns(:Age, [30, 40, 50, 60, 70, 80, Inf], rawData, features)
            createColumns(:NumPosAuxNodes, [0, 5, 10, 20, 30, Inf], rawData, features)
            createColumns(:YearOfOperation, [55, 60, 65, Inf], rawData, features)
            
        end

        # Shuffle the individuals
        features = features[shuffle(1:size(features, 1)),:] 
        trainLimit = round.(Int, size(features, 1) * 2/3)

        train = features[1:trainLimit, :]
        test = features[(trainLimit+1):end, :]
        
        CSV.write(trainDataPath, train)
        CSV.write(testDataPath, test)

        # If the train and test file already exist
    else
        println("=== Loading the features")
        train = CSV.read(trainDataPath)
        test = CSV.read(testDataPath)
    end
    
    println("=== ... ", size(train, 1), " individuals in the train set")
    println("=== ... ", size(test, 1), " individuals in the test set")
    println("=== ... ", size(train, 2), " features")
    
    return train, test
end 


# Create the association rules related to a training set
#
# - train: individuals of the training set (each line is an individual, each column a feature except the first which is the class)
# - output: table of rules (each line is a rule, the first column corresponds to the rules class)
function createRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame)

    rulesPath = resultsFolder * dataSet * "_rules.csv"
    rules = []

    if !isfile(rulesPath)

        println("=== Generating the rules")
        
        # Transactions
        t = train[:, 2:end]

        # Class of the transactions
        transactionClass = train[:, 1]

        # Number of features
        d = size(t, 2)

        # Number of transactions
        n = size(t, 1)

        mincov = 0.05
        iterlim = 5
        RgenX = 0.1 / n
        Rgenb = 0.1 / (n * d)
        
        ##################
        # Find the rules for each class
        ##################
        rules = DataFrames.DataFrame(Int, 0, d + 1)
        
        for y = 0:1

            println("-- Classe $y")
            
            s_bar, iter, s_bar_x = 0, 1, n 
            
           # probleme d'optimisation initial:
        
            m = Model( solver = CplexSolver(CPX_PARAM_SCRIND=0) )
            
            @variable(m, x[i in 1:n], Bin)
            @variable(m, b[j in 1:d], Bin)
            @constraint(m, [i in 1:n, j in 1:d], x[i] <= 1 + (train[i,j] - 1) * b[j]  )
            @constraint(m, [i in 1:n], x[i] >= 1 + sum((train[i,j] - 1) * b[j] for j=1:d) )
            @constraint(m, C, sum(x) <= s_bar_x )
            @objective(m, Max, sum(x[transactionClass .== y]) - RgenX * sum(x) - Rgenb * sum(b))
            i = 1
            
            while (s_bar_x >= n * mincov)
                
                if iter == 1
                  solve(m)
                  x_star = getvalue(x)
                  b_star = getvalue(b)
                  s_bar = sum(x_star[transactionClass .== y])
                  println("rule $i ; s_bar = $s_bar ; s_bar_x = $s_bar_x")
                  iter += 1
                  push!(rules, append!([y], b_star) )
                  i += 1
                  @constraint(m, sum(b[b_star .==0]) + sum( ( ones(d)-b )[b_star .==1] ) >=1 )
                end    
                
                if iter < iterlim
                    solve(m)
                    x_star = getvalue(x)
                    
                    if sum( x_star[transactionClass .== y]) < s_bar
                        s_bar_x = min(s_bar_x - 1, sum(x_star))
                        JuMP.setRHS(C, s_bar_x)
                        iter = 1
                    else
                        iter += 1    
                    end  
                        
                else
                    s_bar_x -= 1
                    JuMP.setRHS(C, s_bar_x)
                    iter = 1
                end  
                        
           end
                    
        end
        
        CSV.write(rulesPath, rules)
                
    else
        println("=== Loading the rules")
        rules = CSV.read(rulesPath)
    end
    println("=== ... ", size(rules, 1), " rules obtained") 

    return rules
end
                
# Sort the rules and keep the 
function sortRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame, rules::DataFrames.DataFrame)

    orderedRulesPath = resultsFolder * dataSet * "_ordered_rules.csv"

    if !isfile(orderedRulesPath)

        println("=== Sorting the rules")
        
        # Transactions
        t = train[:, 2:end]

        # Class of the transactions
        transactionClass = train[:, 1:1]

        # Number of features
        d = size(t, 2)

        # Number of transactions
        n = size(t, 1)
        
        # Add the two null rules
        df = DataFrames.DataFrame(Int, 0, d+1)
        push!(df, append!([1], zeros(d)))
        push!(df, append!([0], zeros(d)))
        
        rules = append!(df,rules)
        rules = unique(rules)

#        append!(rules, DataFrames.DataFrame(append!([0], zeros(d))))
#        append!(rules, DataFrames.DataFrame(append!([1], zeros(d))))
        #        rules = [rules; zeros(2, d)]
        #        ruleClass = [ruleClass 0 1]

        # Number of rules
        L = size(rules)[1]

        Rrank = 1/L

        ################
        # Compute the v_il and p_il constants
        # p_il = :
        #  0 if rule l does not apply to transaction i
        #  1 if rule l applies to transaction i and   correctly classifies it
        # -1 if rule l applies to transaction i and incorrectly classifies it
        ################
        p = zeros(n, L)

        # For each transaction and each rule
        for i in 1:n
            for l in 1:L

                # If rule l applies to transaction i
                # i.e., if the vector t_i - r_l does not contain any negative value
                if !any(x->(x<-epsilon), [sum(t[i, k]-rules[l, k+1]) for k in 1:d])

                    # If rule l correctly classifies transaction i
                    if transactionClass[i, 1] == rules[l, 1]
                        p[i, l] = 1
                    else
                        p[i, l] = -1 
                    end
                end
            end
        end

        v = abs.(p)

        ################
        # Create and solve the model
        ###############
        m  =   Model(solver=CplexSolver(CPX_PARAM_TILIM=600))
#         m  =   Model(solver=CplexSolver(CPX_PARAM_TILIM=1200))

        # u_il: rule l is the highest which applies to transaction i
        @variable(m, u[1:n, 1:L], Bin)

        # r_l: rank of rule l
        @variable(m, 1 <= r[1:L] <= L, Int)

        # rstar: rank of the highest null rule
        @variable(m, 1 <= rstar <= L)
        @variable(m, 1 <= rB <= L)

        # g_i: rank of the highest rule which applies to transaction i
        @variable(m, 1 <= g[1:n] <= L, Int)

        # s_lk: rule l is assigned to rank k
        @variable(m, s[1:L,1:L], Bin)

        # Rank of null rules
        rA = r[L-1]
        rB = r[L]

        # rstar == rB?
        @variable(m, alpha, Bin)

        # rstar == rA?
        @variable(m, 0 <= beta <= 1)

        # Maximize the classification accuracy
        @objective(m, Max, sum(p[i, l] * u[i, l] for i in 1:n for l in 1:L)
                   + Rrank * rstar)

        # Only one rule is the highest which applies to transaction i
        @constraint(m, [i in 1:n], sum(u[i, l] for l in 1:L) == 1)

        # g constraints
        @constraint(m, [i in 1:n, l in 1:L], g[i] >= v[i, l] * r[l])
        @constraint(m, [i in 1:n, l in 1:L], g[i] <= v[i, l] * r[l] + L * (1 - u[i, l]))

        # Relaxation improvement
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] >= 1 - g[i] + v[i, l] * r[l])
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] <= v[i, l]) 

        # r constraints
        @constraint(m, [k in 1:L], sum(s[l, k] for l in 1:L) == 1)
        @constraint(m, [l in 1:L], sum(s[l, k] for k in 1:L) == 1)
        @constraint(m, [l in 1:L], r[l] == sum(k * s[l, k] for k in 1:L))

        # rstar constraints
        @constraint(m, rstar >= rA)
        @constraint(m, rstar >= rB)
        @constraint(m, rstar - rA <= (L-1) * alpha)
        @constraint(m, rA - rstar <= (L-1) * alpha)
        @constraint(m, rstar - rB <= (L-1) * beta)
        @constraint(m, rB - rstar <= (L-1) * beta)
        @constraint(m, alpha + beta == 1)

        # u_il == 0 if rstar > rl (also improve relaxation)
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] <= 1 - (rstar - r[l])/ (L - 1))

        solve(m)

        ###############
        # Write the rstar highest ranked rules and their corresponding class
        ###############

        # Number of rules kept in the classifier
        # (all the rules ranked lower than rstar are removed)
        relevantNbOfRules=L-round(Int, getvalue(rstar))+1

        # Sort the rules and their class by decreasing rank
        rulesOrder = getvalue(r[:])
        orderedRules = rules[sortperm(L .- rulesOrder), :]
        
        orderedRules = orderedRules[1:relevantNbOfRules, :]
        
        CSV.write(orderedRulesPath, orderedRules)

    else
        println("=== Loading the sorting rules")
        orderedRules = CSV.read(orderedRulesPath)
    end 

    return orderedRules

end


function getPrecision(orderedRules::DataFrames.DataFrame, transactions::DataFrames.DataFrame)

    # Number of transactions
    n = size(transactions, 1)

    accuracy = 0
    
    # For all transaction i
    for i in 1:n
        
        # Get the first rule satisfied by transaction i
#         ruleId = findfirst( all( Array{Float64, 2}(orderedRules[2:end][:])  .<= Array{Float64, 2}(transactions[i, 2:end]), 2) )
        ruleId = findfirst( all( Array{Float64, 2}(orderedRules[2:end][:])  .<= transpose(Array{Float64, 1}(transactions[i, 2:end])), dims = 2) )[1]
        if orderedRules[ruleId, 1] == transactions[i, 1]
            accuracy += 1
        end
    end

    accuracy /= n

    return accuracy
    
end 
