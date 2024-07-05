SadNet/
│
└── convert.py           #Convert Data Types if you need  
└── extract aC.py        #Generate seq.fasta, which will be used in parse data.py  
└── get_name.py          #Generate a training directory based on the PDBbind dataset, which is necessary for training.py  
└── training.py          #Execute training, the file will automatically call parse_data.py, GraphPairDataset.py, and finally generate the training model  
└── parse_data.py        #Process the data set and divide it into training set, validation set, and test set  
└── GraphPairDataset.py  #Build graph data for input intothe model  
└── model_test.py        #test your model  
└── vision.py            #Generate visualizations, as shown in the paper  
