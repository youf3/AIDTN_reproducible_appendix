# AIDTN reproducibility initiative appendix for result reproduction

## Artifact Identification

**Title:** AIDTN: Towards a Real-Time AI Optimized DTN System with NVMeoF

**Authors:** Se-young Yu (Northwestern University), Qingyang Zeng (Zhejiang University), Jim Chen (Northwestern University), Yan Chen (Northwestern University), Joe Mambretti (Northwestern University)

**Abstract:** AIDTN is the first effort to provide a unique AI framework designed to incorporate NVMe over Fabrics (NVMeoF) and optimize coordination among multiple components supporting large-scale, multi-domain Wide Area Network (WAN) data-intensive science. This artifact analyzes performance dataset collected from multiple research platforms, builds AI model using machine learning techniques (Bi-LSTM and XGBoost) and examines accruacy of the AI model.

##  Artifact Dependencies and Requirements

**Hardware resources required:** An x86 system capable of running Keras and Tensorflow

**Operating systems required:** GNU/Linux

**Software libraries needed:** Git, Docker, Keras, TensorFlow, Numpy, Pandas, matplotlib, xgboost 

**NOTE: Software libraries are already included in the docker images distributed**

**Input dataset needed:** dataset directory contains datasets required to reproduce results

## Artifact Installation and Deployment Process

Use git to clone the repository and use docker to run aidtn images inside the directory.

i.e.)

`$ git clone git@github.com:youf3/AIDTN_reproducible_appendix.git`

`$ cd AIDTN_graphs`

## Reproducibility of Experiments

Each directory in the dataset contains a training and evaluation dataset with real-time performance data gathered from PRP and MRP.

`dataset` contains real-time CPU, memory, NVMe and throughput utilization and packet loss data from data movement using AIDTN in each CI.

`analyze_performance.py` reads the real-time resource utilization during the data movement and visualizes the goodput and packet losses. Execution time for this process should be less then a minute with a modern system.

`build-realtime-model.py` reads the real-time resource utilization during the data movement and build real-time prediction model using BI-LSTM. Execution time for this process should be less then 30 minutes with a modern system.

`get_score.py` calculates RMSE from the training dataset (`dataset/trainning_data`) generated from PRP and MRP during data movement with different parameters. It also uses a real-time resource utilization  (`dataset/training_data/prp/` and `dataset/training_data/mrp/`) from data movement using AIDTN to calculate RMSE for the prediction during the optimized data movement. Execution time for this process should be less then 30 minutes for each dataset with a modern system.

To generate the result, run docker container youf3/aidtn:latest with dataset mounted to /dataset inside the container and run two automated python modules, analyze_performance.py, build-realtime-model.py and get_score.py

i.e.)

`$ docker run -it -v $PWD/dataset:/dataset -v $PWD/results:/results youf3/aidtn:latest analyze_performance.py`

`$ docker run -it -v $PWD/dataset:/dataset -v $PWD/model:/model youf3/aidtn:latest build-realtime-model.py $cluster` where cluster is either prp (default) or mrp

`$ docker run -it -v $PWD/dataset:/dataset youf3/aidtn:latest get_score.py`

Once they run successfully, you will have performance results as pdf files in the results folder and AI model in the model folder. 
RMSE of the training and evaluation dataset will be shown in the standard output.

The expected output of each process should be identical plots used in the article.

congestion.pdf - Figure 3

prp.pdf - Figure 4

AI_comparison.pdf - Figure 5

features_comparison.pdf - Figure 6

nvmeof_mrp.pdf - Figure 7


