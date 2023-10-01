import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
from utils import to_var, to_device, resample_rows_per_column


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # print ("input_dim: ", input_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print (x.size())
        x_flat = x.view(x.size(0), -1)
        validity = self.model(x_flat)

        return validity


def weights_init_1(m):
    classname = m.__class__.__name__
    # print (classname)
    if classname == "Linear":
        nn.init.kaiming_uniform_(m.weight)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SeqAEGMM(nn.Module):
    """Residual Block."""
    def __init__(self, input_feature_dim, device, n_gmm=2, latent_dim=3, latent_categories=8):
        super(SeqAEGMM, self).__init__()

        self.encoder = nn.GRU(input_feature_dim, latent_dim - 1, batch_first=True,  num_layers=1, bias=True, dropout=0.0)
        self.decoder = nn.GRU(input_feature_dim, latent_dim - 1, batch_first=True,  num_layers=1, bias=True, dropout=0.0)
        self.hidden2output = nn.Linear(latent_dim - 1, input_feature_dim)

        self.latent_categories = latent_categories
        # prototypical intention vector for each intention
        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(latent_dim - 1) *
                                                         (1 / np.sqrt(latent_dim - 1)))
                                            for _ in range(latent_categories)])

        self.layerNorm_z = LayerNorm(latent_dim - 1, eps=1e-12)
        self.layerNorm_prot = LayerNorm(latent_dim - 1, eps=1e-12)
        self.layerNorm_aggr = LayerNorm(latent_dim - 1, eps=1e-12)
        self.beta_seq = nn.Parameter(torch.randn(latent_categories, latent_dim - 1) *
                                           (1 / np.sqrt(latent_dim - 1)))

        layers = []
        layers += [nn.Linear(latent_dim, 10)] # for model without SA
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)
        self.device = device

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def _intention_clustering(self, z: torch.Tensor) -> torch.Tensor:
        # refer to https://github.com/abinashsinha330/DSSRec/blob/main/modules.py#L203
        """
        Method to measure how likely the primary intention of each sequence
        is related with kth latent category
        :param z:
        :return:
        """
        z = self.layerNorm_z(z)
        # print ("z: ", z.size()) # (batch_size, latent_dim - 1)
        hidden_size = z.shape[-1]
        exp_normalized_numerators = list()
        i = 0
        for prototype_k in self.prototypes:
            prototype_k = self.layerNorm_prot(prototype_k) 
            # print ("prototype_k: ", prototype_k.size()) # (latent_dim - 1)
            numerator = torch.matmul(z, prototype_k)
            # print ("numerator: ", numerator.size()) # (batch_size)
            exp_normalized_numerator = torch.exp(numerator / np.sqrt(hidden_size)) # (batch_size, latent_dim - 1)
            # print ("exp_normalized_numerator: ", exp_normalized_numerator.size()) # (batch_size)
            exp_normalized_numerators.append(exp_normalized_numerator)
            if i == 0:
                denominator = exp_normalized_numerator
            else:
                denominator = torch.add(denominator, exp_normalized_numerator)
            i = i + 1

        all_attentions_p_k_i = [torch.div(k, denominator)
                                for k in exp_normalized_numerators]  # # (batch_size) K times
        all_attentions_p_k_i = torch.stack(all_attentions_p_k_i, -1)  # (batch_size, latent_categories)

        return all_attentions_p_k_i

    def _intention_aggr(self,
                        z: torch.Tensor,
                        attention_weights_p_k_i: torch.Tensor) -> torch.Tensor:
        # refer to https://github.com/abinashsinha330/DSSRec/blob/main/modules.py#L253
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param z:
        :param attention_weights_p_k_i:
        :return:
        """
        """
        attention_weights = torch.mul(attention_weights_p_k_i, attention_weights_p_i)  # [B, S, K]
        attention_weights_transpose = attention_weights.transpose(1, 2)  # [B, K, S]
        disentangled_encoding = self.beta_seq + torch.matmul(attention_weights_transpose, z)
        """
        # print ("self.beta_seq: ", self.beta_seq.size()) # (latent_categories, latent_dim - 1)
        attention_weights_p_k_i = torch.unsqueeze(attention_weights_p_k_i, 2)
        # print ("attention_weights_p_k_i: ", attention_weights_p_k_i.size()) # (batch_size, latent_categories, 1)
        z = torch.unsqueeze(z, 1)
        # print ("z: ", z.size()) # (batch_size, 1, latent_dim - 1)
        disentangled_encoding = self.beta_seq + torch.matmul(attention_weights_p_k_i, z)
        
        disentangled_encoding = self.layerNorm_aggr(disentangled_encoding)
        # print ("disentangled_encoding: ", disentangled_encoding.size()) # [batch_size, latent_categories, latent_dim - 1]

        return disentangled_encoding

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim)

    def forward(self, input_data):
        # Prepare input of decoder
        batch_size, max_length, input_dim = input_data.size()

        # 1. Encode the timeseries to make use of the last hidden state.
        encoder_output, encoder_hidden = self.encoder(input_data, None)

        # 2. Use hidden state as initialization for our Decoder-LSTM
        decoder_hidden = encoder_hidden
        # print ("decoder_hidden: ", decoder_hidden.size()) # (1, batch_size, hidden_size)
        
        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = to_var(to_device(torch.Tensor(input_data.size()).zero_(), self.device))
        for i in reversed(range(max_length)):
            # print (output[:, i, :].size()) # (batch_size, input_feature_dim)
            # print (decoder_hidden[0].size()) # (batch_size, hidden_size)
            # print (self.hidden2output(decoder_hidden[0]).size()) # (batch_size, input_feature_dim)
            output[:, i, :] = self.hidden2output(decoder_hidden[0])

            if self.training:
                # print (input_data[:, i, :].unsqueeze(1).size()) # (batch_size, 1, input_feature_dim)
                _, decoder_hidden = self.decoder(input_data[:, i, :].unsqueeze(1), decoder_hidden)
                # print ("decoder_hidden: ", decoder_hidden.size()) # (1, batch_size, hidden_size)
            else:
                _, decoder_hidden = self.decoder(output[:, i, :].unsqueeze(1), decoder_hidden)
        
        # print ("output: ", output.size()) # (batch_size, max_length, input_feature_dim)
        rec_cosine = F.cosine_similarity(input_data.view(batch_size, -1), output.view(batch_size, -1), dim=1)
        # print ("rec_cosine: ", rec_cosine.size()) # (batch_size)

        attention_weights_p_k_i = self._intention_clustering(encoder_hidden[0])
        # print ("attention_weights_p_k_i:", attention_weights_p_k_i.size()) # (batch_size, latent_categories)

        disentangled_encoding = self._intention_aggr(encoder_hidden[0], attention_weights_p_k_i)
        # print ("disentangled_encoding: ", disentangled_encoding.size()) # (batch_size, latent_categories, latent_dim - 1)

        disentangled_encoding_perm = resample_rows_per_column(disentangled_encoding, self.device)
        # print ("disentangled_encoding_perm: ", disentangled_encoding_perm.size()) # (batch_size, latent_categories, latent_dim - 1)
        
        z = torch.mean(disentangled_encoding, 1) # for model without SA and feature attention
        # print ("z: ", z.size()) # (batch_size, latent_dim - 1)

        # Concatenate latent representation, cosine similarity and relative Euclidean distance between x and dec(enc(x))
        z = torch.cat([z, rec_cosine.unsqueeze(-1)], dim=1)
        # print ("z: ", z.size()) # (batch_size, latent_dim)

        # run with GMM estimation
        gamma = self.estimation(z)
        # print ("gamma: ", gamma.size()) # (batch_size, n_gmm)

        return encoder_hidden[0], output, z, gamma, disentangled_encoding, disentangled_encoding_perm

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        # batch_size * K * 1, batch_size * 1 * hidden_size
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(to_device(self.phi, self.device))
        if mu is None:
            mu = to_var(to_device(self.mu, self.device))
        if cov is None:
            cov = to_var(to_device(self.cov, self.device))

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(to_device(torch.eye(D) * eps, self.device))
            pinv = np.linalg.pinv(cov_k.data.cpu().numpy())
            cov_inverse.append(to_var(to_device(torch.from_numpy(pinv), self.device)).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            #if np.min(eigvals) < 0:
            #    logging.warning(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        """
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(to_device(torch.eye(D)*eps, self.device))
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            
            # det_cov.append((torch.potrf(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
        """

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(to_device(torch.from_numpy(np.float32(np.array(det_cov))), self.device))
        
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        """
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        """
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov) + eps).unsqueeze(0), dim=1) + eps)
        
        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, seq_hidden, adj_mini_batch, lambda_second_order_proximity, lambda_energy, lambda_cov_diag):
        batch_size, _, __ = x.size()
        recon_error = torch.mean((x - x_hat) ** 2)

        # refer to https://github.com/suanrong/SDNE/blob/master/model/sdne.py
        D = torch.diag(torch.sum(adj_mini_batch, 1))
        L = D - adj_mini_batch ## L is laplation-matriX
        # print (L.dtype, seq_hidden.dtype)
        pairwise_error = 2 * torch.trace(torch.matmul(torch.matmul(torch.transpose(seq_hidden, 0, 1),L),seq_hidden))

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_second_order_proximity * pairwise_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag, pairwise_error
