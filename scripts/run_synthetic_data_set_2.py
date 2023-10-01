#####################################################################################
# Script for reproducing the performance of our proposed model on the synthetic data 
# set with product diversion to unauthorized merchants, which is shown in Table A10 
# of the online supplemental materials. 
#####################################################################################

import re
import os
import subprocess
import numpy as np

trim_coefficient = 0.9
lr = 0.01
num_epochs = 500
gmm_k = 2
latent_dim = 5
latent_categories = 5
lambda_seq_proximity = 1e-7
lambda_energy = 1e-3
lambda_cov_diag = 1e-7
lambda_disc = 1e-1
use_cuda = True
use_different_lr = False

os.chdir("../src/")

print ("Start running")
input_data_path = "../data/dataset_synthetic_2/hp_sim_sequence_input.npz"
index_id_label_path = "../data/dataset_synthetic_2/partner_index_id_label_traintest.txt"
input_sequence_dist_path = "../data/dataset_synthetic_2/hp_sim_sequence_pearson_correlation.txt"
result_filepath = "../results/result_dataset_synthetic_2.txt"

try:
    os.remove(result_filepath)
except:
    pass

for i in range(0, 10):
    command = "python main.py --trim_coefficient %.8f --lr %.8f --num_epochs %d --gmm_k %d --latent_dim %d \
        --latent_categories %d --lambda_seq_proximity %.8f --lambda_energy %.8f --lambda_cov_diag %.8f --lambda_disc %.8f \
        --use_different_lr %s --use_cuda %s --input_data_path %s --index_id_label_path %s --input_sequence_dist_path %s >> %s" \
        % (trim_coefficient, lr, num_epochs, gmm_k, latent_dim, latent_categories, lambda_seq_proximity, lambda_energy, \
            lambda_cov_diag, lambda_disc, str(use_different_lr), str(use_cuda), input_data_path, index_id_label_path, \
            input_sequence_dist_path, result_filepath)
    c = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    print (c.stdout.decode('ascii'))
print ("Finish running")

print ("Start computing average performance over multiple runs")
auroc_list = []
auprc_list = []
with open(result_filepath) as f:
    for line in f:
        linestr = line.strip()
        if "AUROC : " in linestr:
            auroc_list.append(float(re.search(r'AUROC : (.*?),', linestr).group(1)))
        if "AUPRC : " in linestr:
            # print (linestr.split('AUPRC : ')[1])
            auprc_list.append(float(linestr.split('AUPRC : ')[1]))
print('-'*50)
print ("Number of runs: ", len(auroc_list), ", Mean of AUROC: ", np.mean(auroc_list), ", Standard deviations of AUROC: ", np.std(auroc_list))
print ("Number of runs: ", len(auprc_list), ", Mean of AUPRC: ", np.mean(auprc_list), ", Standard deviations of AUPRC: ", np.std(auprc_list))
print ("Finish computing average performance over multiple runs")
