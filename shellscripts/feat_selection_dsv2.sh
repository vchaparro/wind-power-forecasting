# Incremental feature selection dsv_2

### WF1 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF1
kedro run --pipeline fe --params wf:WF1,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:6

# MARS
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:6

# RF
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:6

# SVM
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:6


### WF2 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF2
kedro run --pipeline fe --params wf:WF2,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:6
									 
# MARS                               
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:6
									 
# RF                                 
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:6
									 
# SVM                                
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:6


### WF3 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF3
kedro run --pipeline fe --params wf:WF3,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:6

# MARS
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:6

# RF
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:6

# SVM
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:6


### WF4 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF4
kedro run --pipeline fe --params wf:WF4,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:6
									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:6
									   
# RF                                   
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:6
									   
# SVM                                  
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:6


### WF5 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF5
kedro run --pipeline fe --params wf:WF5,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:6
									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:6
									  
# RF                                   
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:6
									   
# SVM                                  
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:6


### WF6 ###########################################################################
###################################################################################

kedro run --pipeline de --params wf:WF6
kedro run --pipeline fe --params wf:WF6,add_cycl_feat:True,add_interactions:True

# KNN
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:3
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:4
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:5
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:6
									   
# MARS                                 
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:3
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:4
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:5
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:6
									   
# RF                                   
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:3
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:4
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:5
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:6
									   
# SVM                                  
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:1
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:3
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:4
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:5
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:6





