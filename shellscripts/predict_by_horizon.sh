# Forecasting horizons

### WF1 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF1,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF1
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF1,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF1
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF1,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF1
kedro run --pipeline mdl --params wf:WF1,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF1,alg:SVM,max_k_bests:2




### WF2 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF2,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF2
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF2,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF2
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF2,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF2
kedro run --pipeline mdl --params wf:WF2,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF2,alg:SVM,max_k_bests:2




### WF3 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF3,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF3
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF3,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF3
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF3,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF3
kedro run --pipeline mdl --params wf:WF3,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF3,alg:SVM,max_k_bests:2



### WF4 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF4,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF4
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF4,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF4
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF4,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF4
kedro run --pipeline mdl --params wf:WF4,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF4,alg:SVM,max_k_bests:2



### WF5 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF5,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF5
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF5,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF5
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF5,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF5
kedro run --pipeline mdl --params wf:WF5,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF5,alg:SVM,max_k_bests:2




### WF6 ###########################################################################
###################################################################################

# day
kedro run --pipeline de --params wf:WF6,split_date:"2019-01-14 23:00:00"
kedro run --pipeline fe --params wf:WF6
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:2

# week
kedro run --pipeline de --params wf:WF6,split_date:"2019-01-08 23:00:00"
kedro run --pipeline fe --params wf:WF6
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:2

# month
kedro run --pipeline de --params wf:WF6,split_date:"2018-12-14 23:00:00"
kedro run --pipeline fe --params wf:WF6
kedro run --pipeline mdl --params wf:WF6,alg:KNN,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:MARS,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:RF,max_k_bests:2
kedro run --pipeline mdl --params wf:WF6,alg:SVM,max_k_bests:2







