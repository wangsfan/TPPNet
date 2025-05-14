import torch
import torch.nn as nn
from network.inclearn.lib.utils import logger
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.vision_transformer import _init_vit_weights
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed
from typing import Union
from collections import OrderedDict
import copy
from bayes_layer import BayesianLinear

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1, bayes=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        # self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.bayes = bayes

        if self.bayes:
            self.fc1 = BayesianLinear(in_features, hidden_features)
            self.fc2 = BayesianLinear(in_features, hidden_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(in_features, hidden_features)


    def forward(self, x, sample=None):
        res = x
        if self.bayes:
            x = self.fc1(x, sample)
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.drop1(x)

        if self.bayes:
            x = self.fc2(x, sample)
        else:
            x = self.fc2(x)

        x = self.drop2(x)
        # x += res
        return x


class Mlp_1(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        # x += res
        return x
    


class ExtraTokens(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.tokens = torch.nn.ParameterDict()
        self.embedding_dim = embedding_dim
        # self.task_indices = {}
        self.current_task = -1
        self.num_tokens = 1

    def set_frozen(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_first_tokens(self, all_tokens):
        if all_tokens is not None:
            logger.warning(f'To overwrite origin tokens:\n\t {self.tokens}'
                           f'\n new tokens\n\t {all_tokens}')
            self.tokens['0'] = all_tokens

    def add_tokens(self, num_tokens, is_new_task=True, rand=False):
        logger.info('Added new Tokens into APG.')
        if is_new_task:
            if len(self.tokens):
                new_tokens = nn.Parameter(torch.rand(num_tokens, self.embedding_dim).cuda())
            else:
                new_tokens = nn.Parameter(torch.zeros(num_tokens, self.embedding_dim).cuda())
                torch.nn.init.normal_(new_tokens, mean=0, std=.02)
            self.current_task += 1
            self.tokens[str(self.current_task)] = new_tokens

    def add_tokens_spec(self, classes, tokens, is_new_task=True):
        classes = list(classes)
        for cls, token in zip(classes, tokens):
            self.tokens[str(cls)] = nn.Parameter(token)
        if is_new_task:
            self.task_indices[self.current_task] = classes
            self.current_task += 1

    def get_old_tokens(self):
        if self.current_task == 0:
            return []
        old_tokens = self.tokens[str(self.current_task - 1)]
        return [old_tokens]

    def get_all_tokens(self, return_dict=False):
        if return_dict:
            return self.tokens
        all_tokens = list(self.tokens.values())
        return torch.cat(all_tokens).cuda().reshape(1, -1, all_tokens[0].shape[-1])

    def reset_tokens(self):
        for w in self.tokens.values():
            nn.init.kaiming_normal_(w)


class AdaptivePromptGenerator(nn.Module):
    def __init__(self, embedding_dim, MLP_num=2, num_heads=4, attn_drop=0., proj_drop=0., more_prompts=1, attn_depth=1,
                 use_update_tokens=False, cross_attn_wo_x=False, residual=True, bayes=False):
        super().__init__()
        # logger.info(f'MLP_num: [{MLP_num}],'
        #             f'Num_heads [{num_heads}],'
        #             f'attn_drop [{attn_drop}],'
        #             f'proj_drop [{proj_drop}]'
        #             f'cross_attn_wo_x [{cross_attn_wo_x}],'
        #             f'residual [{residual}]')
        self.embedding_dim = embedding_dim
        self.attn_depth = attn_depth
        self.use_update_tokens = use_update_tokens
        self.MLPs_in = Mlp(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU, bayes=bayes)
        self.MLPs_out = Mlp(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU, bayes=bayes)

        # self.knowladge = torch.nn.Parameter(torch.FloatTensor(1,200,768), requires_grad=True).to('cuda')
        # nn.init.orthogonal_(self.knowladge)

        # self.MLPs_in = Mlp_1(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU)
        # self.MLPs_out = Mlp_1(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU)
        self.task_count = 0
        self.old_num_k = 0
        self.new_num_k = 0
        self.bayes = bayes

        self.num_heads = num_heads
        self.cross_attn_wo_x = cross_attn_wo_x
        self.attn = nn.ModuleDict(OrderedDict({f'attn{i}': MyAttention(embedding_dim,
                                                                       num_heads=self.num_heads,
                                                                       qkv_bias=True,
                                                                       attn_drop=attn_drop,
                                                                       proj_drop=proj_drop,
                                                                       more_prompts=more_prompts,
                                                                       cross_attn_wo_x=self.cross_attn_wo_x,
                                                                       residual=residual,
                                                                       bayes=bayes) for i in range(attn_depth)}))
        
        self.norms = nn.ModuleDict(
            OrderedDict({f'norm{i}': nn.LayerNorm(self.embedding_dim) for i in range(attn_depth - 1)}))
        
        # logger.info(f'current_attn_layers:{self.attn}')

        self.all_tokens = ExtraTokens(embedding_dim)

        self.norm3 = nn.LayerNorm(self.embedding_dim)
        self.norm4 = nn.LayerNorm(self.embedding_dim)

        self.apply(_init_vit_weights)

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        # pt = int(self.e_pool_size / (self.n_tasks))
        # s = int(self.task_count * pt)
        # f = int((self.task_count + 1) * pt)
        s = self.old_num_k
        f = self.new_num_k
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def process_task_count(self, old_num_k, new_num_k):
        self.task_count += 1
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k
        self.knowladge = self.gram_schmidt(self.knowladge.squeeze(0)).unsqueeze(0)

    def forward(self, img_feat, text_tokens=None, task_id=None, all_tokens=True, specific=None, return_attn=False, sample=None):
        # assert all_tokens or (not all_tokens and specific is not None)
        # tokens = self.all_tokens.get_all_tokens() if all_tokens else specific

        out = dict()
        if task_id > 0:
            # tokens = text_tokens[:,:20,:]
            tokens = torch.cat((text_tokens[:,:task_id*10,:].detach().clone(),text_tokens[:,task_id*10:(task_id+1)*10,:]), dim=1)
            # tokens = torch.cat((self.knowladge[:,:task_id*10,:].detach().clone(),self.knowladge[:,task_id*10:(task_id+1)*10,:]), dim=1)
        else:
            tokens = text_tokens[:,:10,:]
            # tokens = self.knowladge[:,:10,:]
        # tokens = text_tokens[:,:20,:]
            
        # print(tokens[:,:10,:])

        q = self.MLPs_in(img_feat, sample=sample)

        q_new = self.norm3(q)
        attn = torch.tensor([]).cuda()
        # attn = torch.tensor([])
        if self.use_update_tokens:
            for i in range(self.attn_depth):
                if i == 0:
                    q_new, attn_new, other_tokens = self.attn[f'attn{i}'](q_new, tokens, return_all_tokens=True, sample=sample)
                else:
                    q_new, attn_new, other_tokens = self.attn[f'attn{i}'](q_new, other_tokens, return_all_tokens=True, sample=sample)

                if i != self.attn_depth - 1:
                    q_new = self.norms[f'norm{i}'](q_new)

                attn = torch.cat((attn, attn_new), dim=0)
        else:
            for i in range(self.attn_depth):
                q_new, attn_new = self.attn[f'attn{i}'](q_new, tokens, sample=sample)
                if i != self.attn_depth - 1:
                    q_new = self.norms[f'norm{i}'](q_new)
                attn = torch.cat((attn, attn_new), dim=0)
        
        
        q_new = self.MLPs_out(q_new, sample=sample)
        out['e_prompt'] = q_new
        out['attn'] = attn
        if return_attn:
            # return q_new, attn
            return out
        else:
            # return q_new
            return out


class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., more_prompts=1,
                 cross_attn_wo_x=False, residual=True, bayes=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.more_prompts = more_prompts  # output #self.more_prompts prompts.
        # if self.more_prompts > 1:
        #     logger.warning(f'Output prompts #{self.more_prompts} instead of one.')
        # else:
        #     logger.info(f'Output Prompts #{self.more_prompts}')
        self.scale = qk_scale or head_dim ** -0.5

        # self.q = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        # self.k = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cross_attn_wo_x = cross_attn_wo_x
        self.residual = residual
        self.bayes = bayes

        if self.bayes:
            self.q = BayesianLinear(dim, dim * self.more_prompts)
            self.k = BayesianLinear(dim, dim * self.more_prompts)
            self.v = BayesianLinear(dim, dim * self.more_prompts)
            self.proj = BayesianLinear(dim, dim)
        else:
            self.q = nn.Linear(dim, dim * self.more_prompts)
            self.k = nn.Linear(dim, dim * self.more_prompts)
            self.v = nn.Linear(dim, dim * self.more_prompts)
            self.proj = nn.Linear(dim, dim)

    def forward(self, x, prompts, return_all_tokens=False, sample=None):
        res_x = x
        B, N, C = x.shape
        assert N == 1
        _, N_prompts, C_prompts = prompts.shape
        assert C == C_prompts  # assert with same dim

        if self.cross_attn_wo_x:
            x_with_prompts = prompts.expand(B, N_prompts, C)
            k_v_length = N_prompts
        else:
            x_with_prompts = torch.cat((x, prompts.expand(B, N_prompts, C)), dim=1)
            k_v_length = N_prompts + N

        if return_all_tokens:
            res_x = res_x[:, 0, ...]
            assert self.more_prompts == 1
            if self.bayes:
                q = self.q(x_with_prompts, sample).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                                C // self.num_heads).permute(0, 2, 1, 3)
            else:
                q = self.q(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                                   C // self.num_heads).permute(0, 2, 1, 3)

        else:
            if self.bayes:
                q = self.q(x, sample).unsqueeze(1).reshape(B, N, self.more_prompts * self.num_heads,
                                                C // self.num_heads).permute(0, 2, 1, 3)
            else:
                q = self.q(x).unsqueeze(1).reshape(B, N, self.more_prompts * self.num_heads,
                                                C // self.num_heads).permute(0, 2, 1, 3)

        if self.bayes:
            k = self.k(x_with_prompts, sample).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                            C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                            C // self.num_heads).permute(0, 2, 1, 3)
                        
        q = q * self.scale
        if self.bayes:
            v = self.v(x_with_prompts, sample).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                            C // self.num_heads).permute(0, 2, 1, 3)
        else:
            v = self.v(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                            C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        raw_attn = attn
        attn = attn.softmax(dim=-1)
        attn = attn * self.scale

        attn = self.attn_drop(attn)

        if return_all_tokens:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, N_prompts + self.more_prompts, C)
        else:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, self.more_prompts, C)

        if self.bayes:
            x_cls = self.proj(x_cls, sample)
        else:
            x_cls = self.proj(x_cls)
            
        x_cls = self.proj_drop(x_cls)

        if return_all_tokens:
            other_tokens = x_cls[:, 1:, ...]
            x_cls = x_cls[:, 0, ...]
            if self.residual:
                x_cls += res_x
            x_cls = x_cls.unsqueeze(1)
            return x_cls, raw_attn, other_tokens
        else:
            if self.residual:
                x_cls += res_x
            return x_cls, raw_attn



if __name__ == '__main__':
    print('dfh')
    atten = MyAttention(dim=36, num_heads=2, more_prompts=1)
    APG = AdaptivePromptGenerator(embedding_dim=36, MLP_num=2, num_heads=4, attn_drop=0., proj_drop=0., more_prompts=1, attn_depth=1,
                 use_update_tokens=True, cross_attn_wo_x=True, residual=True)
    x = torch.rand(16, 1, 36)
    prompt = torch.rand(1, 50, 36)
    out = APG(x, prompt, task_id=0)
    # out = atten(x, prompt)
    print(out['e_prompt'].shape)
    # net = SelfPromptDeit_mytiny()
    # params = 0
    # for param in net.parameters():
    #     params += param.numel()
    # print(params)
