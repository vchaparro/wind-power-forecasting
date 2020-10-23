kedro run --pipeline dcnr --params wf:WF1
kedro run --pipeline fecnr --params wf:WF1,max_k_bests:2
kedro run --pipeline pcnr --params wf:WF1,alg:MARS
 
kedro run --pipeline dcnr --params wf:WF2
kedro run --pipeline fecnr --params wf:WF2,max_k_bests:5
kedro run --pipeline pcnr --params wf:WF2,alg:SVM

kedro run --pipeline dcnr --params wf:WF3
kedro run --pipeline fecnr --params wf:WF3,max_k_bests:6
kedro run --pipeline pcnr --params wf:WF3,alg:SVM

kedro run --pipeline dcnr --params wf:WF4
kedro run --pipeline fecnr --params wf:WF4,max_k_bests:6
kedro run --pipeline pcnr --params wf:WF4,alg:SVM

kedro run --pipeline dcnr --params wf:WF5
kedro run --pipeline fecnr --params wf:WF5,max_k_bests:5
kedro run --pipeline pcnr --params wf:WF5,alg:SVM

kedro run --pipeline dcnr --params wf:WF6
kedro run --pipeline fecnr --params wf:WF6,max_k_bests:3
kedro run --pipeline pcnr --params wf:WF6,alg:SVM
 
