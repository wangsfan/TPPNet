import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _single, _pair, _triple
# from arguments import get_args
# args = get_args()


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def custom_regularization(saver_net, trainer_net, mini_batch_size):
        
    sigma_weight_reg_sum = 0
    sigma_bias_reg_sum = 0
    sigma_weight_normal_reg_sum = 0
    sigma_bias_normal_reg_sum = 0
    mu_weight_reg_sum = 0
    mu_bias_reg_sum = 0
    L1_mu_weight_reg_sum = 0
    L1_mu_bias_reg_sum = 0
    loss = 0

    sigma_import_sum = 0
    
    out_features_max = 512
    
    
    prev_weight_strength = nn.Parameter(torch.Tensor(1,1).uniform_(0,0)).cuda()
    
    for (n, saver_layer), (name_curr, trainer_layer) in zip(saver_net.named_modules(), trainer_net.named_modules()):
        
        if isinstance(trainer_layer, BayesianLinear)==False:
            continue
        # calculate mu regularization
        # print(n)
        trainer_weight_mu = trainer_layer.weight_mu
        saver_weight_mu = saver_layer.weight_mu
        trainer_bias = trainer_layer.bias
        saver_bias = saver_layer.bias
        
        fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
        
        trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
        saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

        
        
        if isinstance(trainer_layer, BayesianLinear):
            std_init = 0.1 * math.sqrt((2 / fan_in) * 0.5)
        
        saver_weight_strength = (std_init / saver_weight_sigma)


        if len(saver_weight_mu.shape) == 4:
            out_features, in_features, _, _ = saver_weight_mu.shape
            curr_strength = saver_weight_strength.expand(out_features,in_features,1,1)
            prev_strength = prev_weight_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)
        
        else:
            out_features, in_features = saver_weight_mu.shape
            curr_strength = saver_weight_strength.expand(out_features,in_features)
            if len(prev_weight_strength.shape) == 4:
                feature_size = in_features // (prev_weight_strength.shape[0])
                prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0],-1)
                prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                prev_weight_strength = prev_weight_strength.reshape(-1,1)
            prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
        
        L2_strength = torch.max(curr_strength, prev_strength)
        bias_strength = torch.squeeze(saver_weight_strength)

        L1_sigma = saver_weight_sigma
        bias_sigma = torch.squeeze(saver_weight_sigma)
        
        if 'query' in name_curr:
            prev_weight_strength_query = saver_weight_strength
        elif 'key' in name_curr:
            prev_weight_strength_key = saver_weight_strength  
        elif 'value' in name_curr:
            # prev_weight_strength = (saver_weight_strength + prev_weight_strength_query + prev_weight_strength_key)/3
            prev_weight_strength = torch.max(torch.max(saver_weight_strength, prev_weight_strength_query), prev_weight_strength_key)
        else :
            prev_weight_strength = saver_weight_strength  
        

        mu_weight_reg = (L2_strength * (trainer_weight_mu-saver_weight_mu)).norm(2)**2
        mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2

        L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
        L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)

        L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
        L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
        weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
        sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()

        
        mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
        mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg

        L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
        L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg

    loss = loss + 1 * (mu_weight_reg_sum + mu_bias_reg_sum) / (2* mini_batch_size)
    # L1 loss
    loss = loss + 1 * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
    # # sigma regularization
    loss = loss + 1 * (sigma_weight_reg_sum) / (mini_batch_size)
        
    return loss

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        # self.mu = mu.cuda()
        self.mu = mu
        # self.rho = rho.cuda()
        self.rho = rho
        # self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        # epsilon = self.normal.sample(self.mu.size()).cuda()
        # print("1")
        epsilon = torch.normal(0,1,self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon   

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        gain = 1 # Var[w] + sigma^2 = 2/fan_in
        
        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var
        
        noise_std, mu_std = 0.1 * math.sqrt(noise_var), 0.1 * math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0))
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias
        else:
            weight = self.weight.mu
            bias = self.bias

        return F.linear(input, weight, bias)

