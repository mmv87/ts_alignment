Mulitmodal timeseries data: 
To reproduce the methodology as implemented in "Towards Time Series Reasoning with LLMs" referecen:http://arxiv.org/abs/2409.11376
As states model training constitutes two stages
	1.Stage : 1 
		Time-series textual data alignment : TS-encoder (trainable)with backbone LLM (frozen)
	2. Stage: 2 
		TS-encoder and the LLM trained together
	The current repo has the datapipelin for univariate timeseries data with Phi_4_mini model used a LLM backbone
	
