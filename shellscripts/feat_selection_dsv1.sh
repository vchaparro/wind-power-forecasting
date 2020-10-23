# Incremental feature selection dsv_1

### WF1 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF1
kedro run --pipeline fe --params wf:WF1

# KNN
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:2

# MARS
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:2

# RF
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:2

# SVM
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:2



### WF2 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF2
kedro run --pipeline fe --params wf:WF2

# KNN
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:2

									 
# MARS                               
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:2

									 
# RF                                 
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:2

									 
# SVM                                
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:2



### WF3 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF3
kedro run --pipeline fe --params wf:WF3

# KNN
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:2


# MARS
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:2


# RF
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:2


# SVM
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:2



### WF4 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF4
kedro run --pipeline fe --params wf:WF4

# KNN
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:2

									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:2

									   
# RF                                   
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:2

# SVM                                  
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:2


### WF5 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF5
kedro run --pipeline fe --params wf:WF5

# KNN
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:2

									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:2

									  
# RF                                   
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:2

									   
# SVM                                  
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:2



### WF6 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF6
kedro run --pipeline fe --params wf:WF6

# KNN
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:2

									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:2

									   
# RF                                   
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:2

# SVM                                  
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:2






