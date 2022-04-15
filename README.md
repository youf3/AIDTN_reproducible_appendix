# AIDTN AE/AD reproducibility appendix result reproduction for ICPP 2022

Each directory in the dataset contains a training and evaluation dataset with real-time performance data gathered from PRP and MRP.

`dataset` contains real-time CPU, memory, NVMe and throughput utilization and packet loss data from data movement using AIDTN in each CI.

`build-realtime-model.py` reads the real-time resource utilization during the data movement and build real-time prediction model using BI-LSTM.

`analyze_performance.py` reads the real-time resource utilization during the data movement and visualizes the goodput and packet losses.

`get_score.py` calculates RMSE from the training dataset (`dataset/trainning_data`) generated from PRP and MRP during data movement with different parameters. It also uses a real-time resource utilization  (`dataset/training_data/prp/` and `dataset/training_data/mrp/`) from data movement using AIDTN to calculate RMSE for the prediction during the optimized data movement.

To generate the result, run docker container youf3/aidtn:latest with dataset mounted to /dataset inside the container and run two automated python modules, analyze_performance.py, build-realtime-model.py and get_score.py

i.e.)

`$ git clone git@github.com:youf3/AIDTN_reproducible_appendix.git`

`$ cd AIDTN_graphs`

`$ docker run -it -v $PWD/dataset:/dataset -v $PWD/results:/results youf3/aidtn:latest analyze_performance.py`

`$ docker run -it -v $PWD/dataset:/dataset -v $PWD/model:/model youf3/aidtn:latest build-realtime-model.py $cluster` where cluster is either prp (default) or mrp

`$ docker run -it -v $PWD/dataset:/dataset youf3/aidtn:latest get_score.py`

Once they run successfully, you will have performance results as pdf files in the results folder and AI model in the model folder. 
RMSE of the training and evaluation dataset will be shown in the standard output.# AIDTN_reproducible_appendix
