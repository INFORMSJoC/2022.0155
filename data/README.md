# Data for replication

1. Here is the data needed for replicating the experiments. Before running the codes, please download the data sets from the following links, put them inside the folder `data`, and unzip them into the folder.
* The link for the empirical data set used in Table 1 of the paper: [HP data set](https://drive.google.com/file/d/1yHZzEEmfAb8iHmFnmuuVPy_3Ag4d0FF5/view?usp=sharing).
* The link for the synthetic data sets used in Figure 4 of the paper: [Synthetic data sets with different different levels of training or testing anomalies](https://drive.google.com/file/d/1CV71PaW24BPA6o-TA42FaqnAW1PrMpuv/view?usp=sharing).
* The link for the synthetic data set used in Table A10 of the online supplemental material: [Synthetic data set that contains simulations of product diversion to unauthorized merchants](https://drive.google.com/file/d/1wkI8IPgz2Q57sBoEYGxPqcPACKW3arh5/view?usp=sharing).

2. Each of the above data sets contains three data files:
* A file with the name `partner_index_id_label_traintest.txt`: each line corresponds to one distributor, which includes its index, ID, ground-truth label (0 for the normal, 1 for anomaly), and an indicator denoting whether it is in the training, testing, or validation data set. We give some sampled lines of this file as follows:
```
0,2-O3B-407,0,test
1,2-2JA7W7,0,train
2,2-5Q23KY,0,train
3,2-O3D-32407,0,test
4,2-LKEW-11,0,train
```
* A file with the name `hp_sequence_input.npz` for the empirical data or the name `hp_sim_sequence_input.npz` for the synthetic data: this file includes the normalized TSOQs that are input to the model, and could be read and visualized by the script [data/read_npz_file.py](read_npz_file.py).
* A file with the name `sequence_pearson_correlation.txt` for the empirical data or the name `hp_sim_sequence_pearson_correlation.txt` for the synthetic data: each line corresponds to the index of the first distributor, the index of the second distributor, and the pairwise pearson correlation of the two distributors which is computed based on the two distributors' TSOQs. We give some sampled lines of this file as follows:
```
0,1,-0.12666009927622474
0,2,0.9007994088632443
0,3,-0.25453469060801115
0,4,-0.18237303234178187
```
