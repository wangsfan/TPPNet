import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
# from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torch.nn.functional as F
import copy





def load_clip_to_cpu():
    url = clip._MODELS['ViT-L/14']
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict()).eval()
    return model.float()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # x = prompts + self.positional_embedding[:50].type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class PromptLearner(nn.Module):
    def __init__(self, args = None,clip_model =None,class_list=None):
        super().__init__()

        #prompt_learner-----------------------
        dtype = clip_model.dtype
        self.classes_list = class_list
        self.class_list = [self.classes_list[i:i + 10] for i in range(0, 100, 10)]
        self.class_task_1 = [' '.join(itm) for itm in self.class_list]
        self.class_task = ['a photo of ' + itm for itm in self.class_task_1]
        self.class_task_class = ['a photo of ' + a for a in self.classes_list]

        self.e_pool_size = 100
        self.n_tasks = 10
        self.task_count = 0
        self.old_num_k = 0
        self.new_num_k = 0

        n_cls = len(self.class_task)
        n_cls_class = len(self.class_task_class)

        n_ctx = 10
        ctx_init = False
        ctx_dim = 768

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if True:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_class_vectors = torch.empty(n_cls_class, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_class_vectors, std=0.02)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ct_class = nn.Parameter(ctx_class_vectors)  # to be optimized

        self.ctx = torch.nn.Parameter(ctx_vectors, requires_grad=True)  
        self.ct_class = torch.nn.Parameter(ctx_class_vectors, requires_grad=True)

        #task_level prompt_learner-----------------------
        classnames = [name.replace("_", " ") for name in self.class_task]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # tokenized_prompts_task = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts_task = torch.cat([clip.tokenize(p) for p in classnames])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_task).type(dtype)
        
        self.embedding = embedding

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        #class_level prompt_learner-----------------------
        classnames_class = [name.replace("_", " ") for name in self.class_task_class]
        prompt_class = [prompt_prefix + " " + name + "." for name in classnames_class]
        tokenized_prompts_class = torch.cat([clip.tokenize(p) for p in prompt_class])
        with torch.no_grad():
            embedding_class = clip_model.token_embedding(tokenized_prompts_class).type(dtype)
        # print(embedding_class.shape)
        self.register_buffer("token_prefix_class", embedding_class[:, :1])  # SOS
        self.register_buffer("token_suffix_class", embedding_class[:, 1 + n_ctx :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.tokenized_prompts_class = tokenized_prompts_class  # torch.Tensor
        self.tokenized_prompts_task = tokenized_prompts_task

        self.class_token_position = 'end'


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
        # ct_class = self.ct_class
        # ct_class = self.gram_schmidt(ct_class)
        # self.ct_class = ct_class
        self.ct_class = self.gram_schmidt(self.ct_class)
        # self.ctx = self.gram_schmidt(self.ctx)

    def forward(self):

            ctx = self.ctx
            ctx_class = self.ct_class
            text_task = self.embedding

            prefix_class = self.token_prefix_class
            suffix_class = self.token_suffix_class

            prefix_task = self.token_prefix
            suffix_task = self.token_suffix

            if self.class_token_position == "end":

                prompts_task = torch.cat(
                    [
                        prefix_task,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix_task,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
                # prompts_task = text_task

                prompts_class = torch.cat(
                    [
                        prefix_class,  # (n_cls, 1, dim)
                        ctx_class,     # (n_cls, n_ctx, dim)
                        suffix_class,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                raise ValueError

            return prompts_class, prompts_task