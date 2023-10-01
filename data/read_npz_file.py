import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot(hp_sequence, index2label, random_select_index_set, dataset_name):
    print ("hp_sequence: ", hp_sequence.shape)

    timelens = hp_sequence.shape[1]
    for select_index in random_select_index_set:
        label = index2label[select_index]
        data = np.transpose(hp_sequence[select_index, :, :])
        # print ("data: ", data.shape) # data:  (896, 12)
        data_labels = np.zeros((data.shape[0], 1))
        data = np.concatenate((data, data_labels), axis=1)
        # print ("data: ", data.shape) # data:  (896, 13)

        mts_df = pd.DataFrame(data=data)
        # print ("mts_df: ", mts_df.shape) # (896, 13)
        num_cols = mts_df.shape[1]
        with plt.style.context(("ggplot", "seaborn")):
            fig = plt.figure(figsize=(10, 6))
            if label == "normal":
                pd.plotting.parallel_coordinates(mts_df, data.shape[1] - 1, axvlines=False, cols=range(1, num_cols), use_columns=True, \
                    color=["k"], axvlines_kwds={"color":"whitesmoke"})
            else:
                pd.plotting.parallel_coordinates(mts_df, data.shape[1] - 1, axvlines=False, cols=range(1, num_cols), use_columns=True, \
                    color=["#6e9ece"], axvlines_kwds={"color":"whitesmoke"})
            plt.gca().legend_.remove()
            plt.tight_layout()
            plt.savefig("%s_%s_%s.pdf" % (dataset_name, select_index, index2label[select_index]), dpi=300)
            # plt.show()


if __name__ == '__main__':
    data = np.load("./dataset_synthetic_2/hp_sim_sequence_input.npz")["feature"]
    print ("[dataset_synthetic_2] data.shape: ", data.shape) # (9000, 12, 896)

    index2label = dict()
    with open("./dataset_synthetic_2/partner_index_id_label_traintest.txt") as f:
        for line in f:
            index, _id, label, traintest = line.strip().split(",")
            index = int(index)
            if label == "0":
                index2label[index] = "normal"
            else:
                index2label[index] = "anomaly"
    
    random_select_index_set = [0, 4]
    dataset_name = 'dataset_synthetic_2'
    plot(data, index2label, random_select_index_set, dataset_name)

    data = np.load("./dataset_hp/hp_sequence_input.npz")["feature"]
    print ("[dataset_hp] data.shape: ", data.shape) # (4063, 12, 896)

    index2label = dict()
    with open("./dataset_hp/partner_index_id_label_traintest.txt") as f:
        for line in f:
            index, _id, label, traintest = line.strip().split(",")
            index = int(index)
            if label == "0":
                index2label[index] = "normal"
            else:
                index2label[index] = "anomaly"

    random_select_index_set = [21, 443]
    dataset_name = 'dataset_hp'
    plot(data, index2label, random_select_index_set, dataset_name)
