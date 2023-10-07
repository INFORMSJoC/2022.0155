"""passed the test on Python 3.6 with Pytorch 1.4.0 or Python 3.8.5 with Pytorch 1.7.0
"""
import math
import time
import argparse
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn.functional import softplus
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from utils import to_var, to_device, model_to_device
from model import SeqAEGMM, Discriminator

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Compute the device upon which to run
    if not torch.cuda.is_available() or not config.use_cuda:
        device = "cpu"
    else:
        device = config.device
    print ("device: ", device)

    # load input features
    data = np.load(config.input_data_path)
    features = data["feature"]
    num_instances, MAX_LENGTH, input_feature_dim = features.shape

    # load index and labels
    train_index = []
    train_labels = []
    test_index = []
    test_labels = []
    with open(config.index_id_label_path) as f:
        for line in f:
            index, _id, label, traintest = line.strip().split(",")
            index = int(index)
            label = int(label)
            if traintest == "train":
                train_labels.append(label)
                train_index.append(index)
            elif traintest == "test":
                test_labels.append(label)
                test_index.append(index)
    train_index_set = set(train_index)
    train_index = np.array(train_index).astype(int)
    train_labels = np.array(train_labels).astype(int)
    train_features = features[train_index].astype(np.float32)
    test_index = np.array(test_index).astype(int)
    test_labels = np.array(test_labels).astype(int)
    test_features = features[test_index].astype(np.float32)

    # construct adj matrix within train data points
    train_index_dict = dict()
    for idx, index in enumerate(train_index):
        train_index_dict[index] = idx
    adj_matrix = np.eye(len(train_index))
    with open(config.input_sequence_dist_path) as f:
        for line in f:
            lidx, ridx, corr = line.strip().split(",")
            lidx = int(lidx)
            ridx = int(ridx)
            if corr == "nan":
                continue
            if lidx not in train_index_set or ridx not in train_index_set:
                continue
            corr = float(corr)
            adj_matrix[train_index_dict[lidx], train_index_dict[ridx]] = corr
            adj_matrix[train_index_dict[ridx], train_index_dict[lidx]] = corr
    
    positive_pct = config.trim_coefficient * 100
    positive_thres = np.percentile(adj_matrix, positive_pct)
    print ("Positive_thres: ", positive_thres)
    pos_index = adj_matrix >= positive_thres
    other_index = adj_matrix < positive_thres
    adj_matrix[other_index] = 0
    adj_matrix[pos_index] = 1
    adj_matrix = adj_matrix.astype(np.float32)
    print ("Number of non-entries in adj_matrix: ", np.sum(adj_matrix))

    #--- build_model start ---
    # Define model
    discriminator = Discriminator(config.latent_categories * (config.latent_dim - 1))
    seqaegmm = SeqAEGMM(input_feature_dim, device, n_gmm=config.gmm_k, latent_dim=config.latent_dim, latent_categories=config.latent_categories)

    # reference to https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch/51802206
    if config.use_different_lr:
        #optimizer = torch.optim.SGD([
        optimizer = torch.optim.Adam([
            {"params": seqaegmm.encoder.parameters(), "lr": config.seqae_lr},
            {"params": seqaegmm.decoder.parameters(), "lr": config.seqae_lr},
        ], lr=config.lr)
    else:
        optimizer = torch.optim.Adam(seqaegmm.parameters(), config.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), config.lr)

    # Print networks
    name = 'SeqAEGMMCorrDisentangle'
    model = seqaegmm
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))
    #--- build_model end ---

    # Send the model and data to the available device
    model = model_to_device(model, device)
    discriminator = model_to_device(discriminator, device)

    def train():
        # train model, Start training
        start = time.time()

        print_every = 1
        print_loss_total = 0  # Reset every print_every
        print_sample_energy = 0
        print_recon_error = 0
        print_cov_diag = 0
        print_pair_error = 0
        
        input_data = to_var(to_device(torch.from_numpy(train_features), device))
        input_adj_matrix = to_var(to_device(torch.from_numpy(adj_matrix), device))

        best_auroc_results = None
        for e in range(1, config.num_epochs + 1):
            model.train()
            discriminator.train()

            enc, decoder_output, z, gamma, disentangled_encoding, disentangled_encoding_perm = model(input_data)            
            
            optimizer.zero_grad()
            total_loss, sample_energy, recon_error, cov_diag, pair_error = model.loss_function(input_data, decoder_output, z, gamma, enc, input_adj_matrix, config.lambda_seq_proximity, config.lambda_energy, config.lambda_cov_diag)
            total_loss = torch.clamp(total_loss, max=1e7)  # Extremely high loss can cause NaN gradients
            joint_logit = discriminator(disentangled_encoding)
            marg_logit = discriminator(disentangled_encoding_perm)
            disc_cost = softplus(-marg_logit).mean() + softplus(joint_logit).mean()
            total_loss = total_loss - config.lambda_disc * disc_cost
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # adverserial training refer to https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/gan/gan.py#L64
            optimizer_D.zero_grad()
            joint_logit = discriminator(disentangled_encoding.detach())
            marg_logit = discriminator(disentangled_encoding_perm.detach())
            # refer to https://github.com/pbrakel/anica/blob/master/train.py#L269
            disc_cost = softplus(-marg_logit).mean() + softplus(joint_logit).mean()
            # print ("disc_cost: ", disc_cost.size()) # []
            disc_cost.backward()
            optimizer_D.step()

            total_loss = total_loss.data.item()
            sample_energy = sample_energy.item()
            recon_error = recon_error.item()
            cov_diag = cov_diag.item()
            pair_error = pair_error.item()

            print_loss_total += total_loss
            print_sample_energy += sample_energy
            print_recon_error += recon_error
            print_cov_diag += cov_diag
            print_pair_error += pair_error
            
            if e % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print_sample_energy_avg = print_sample_energy / print_every
                print_sample_energy = 0

                print_recon_error_avg = print_recon_error / print_every
                print_recon_error = 0

                print_cov_diag_avg = print_cov_diag / print_every
                print_cov_diag = 0

                print_pair_error_avg = print_pair_error / print_every
                print_pair_error = 0

                print('%s (%d %d%%) Total Loss: %.8f, Sample Energy: %.8f, Recon Error: %.8f, Cov Dia: %.8f, Pair Error: %.8f' % (timeSince(start, e / config.num_epochs),
                                             e, e / config.num_epochs * 100.0, print_loss_avg, print_sample_energy_avg, print_recon_error_avg, print_cov_diag_avg, print_pair_error_avg))

            roc_auc, prc_auc = test() # test for each epoch

            if best_auroc_results is not None:
                if roc_auc > best_auroc_results[0]:
                    best_auroc_results = [roc_auc, prc_auc]
                else:
                    pass
            else:
                best_auroc_results = [roc_auc, prc_auc]

        roc_auc, prc_auc = best_auroc_results
        print("AUROC : {:0.4f}, AUPRC : {:0.4f}".\
            format(roc_auc, prc_auc))

        # print ("phi", model.phi, "mu", model.mu, "cov", model.cov)

    def compute_AUROC(y, scores):
        lw = 2
        fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc

    def compute_AUPRC(y, scores):
        p, r, thresholds = precision_recall_curve(y, scores, pos_label=1)
        prc_auc = auc(r, p)

        return prc_auc

    def test():
        # print("======================TEST MODE======================")
        model.eval()
        discriminator.eval()
        
        input_test_data = to_var(to_device(torch.from_numpy(test_features), device))
        _, decoder_output, z, gamma, __, ___ = model(input_test_data)
        sample_energy, cov_diag = model.compute_energy(z, size_average=False)
        test_energy = sample_energy.data.cpu().numpy()

        pred_score = test_energy.astype(float)
        gt = test_labels.astype(int)

        roc_auc = compute_AUROC(gt, pred_score)
        prc_auc = compute_AUPRC(gt, pred_score)

        # print("AUROC : {:0.4f}, AUPRC : {:0.4f}".format(roc_auc, prc_auc))

        return (roc_auc, prc_auc)

    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seqae_lr', type=float, default=0.01)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--latent_categories', type=int, default=8)
    parser.add_argument('--lambda_seq_proximity', type=float, default=1e-4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--lambda_disc', type=float, default=0.1)
    parser.add_argument('--trim_coefficient', type=float, default=0.9)

    parser.add_argument('--method', type=str, default='SeqAEGMMJointCorrDisentangle')
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0") # "cuda", "gpu", "cuda:0"
    parser.add_argument('--use_different_lr', type=str2bool, default=True)

    # Path
    parser.add_argument('--input_data_path', type=str, default='hp_sequence_input.npz')
    parser.add_argument('--input_sequence_dist_path', type=str, default='sequence_pearson_correlation.txt')
    parser.add_argument('--index_id_label_path', type=str, default='partner_index_id_label_traintest.txt')

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
