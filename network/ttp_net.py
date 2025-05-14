import copy
import this
import numpy as np
import torch
from torch import nn
from torch.distributions import relaxed_bernoulli
import torch.nn.functional as F
from .base import SimpleLinear

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.convit import ClassAttention 
from models.convit import Block as ConBlock
from network.APG import AdaptivePromptGenerator as APG
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from clip.text_encoder import TextEncoder, PromptLearner, load_clip_to_cpu

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_KPrompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p 


class TTP_Net(nn.Module):
    def __init__(self,class_name, args):
        super(TTP_Net,self).__init__()
        self.args=args
        self.dataset_name=args["dataset"]
        self.clas_w=nn.ModuleList()
        self.ts_prompts_1=nn.ModuleList()
        self.ts_prompts_2=nn.ModuleList()
        self.class_name = class_name
 
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, MLP_num=args['APG_MLP_num'],num_heads_APG=args['APG_num_heads'],
                            proj_drop=args['APG_proj_drop'],attn_drop=['APG_attn_drop'],
                            more_prompts=args['APG_more_prompts'],attn_depth=args['APG_attn_depth'],
                            use_update_tokens=args['APG_use_update_tokens'],cross_attn_wo_x=args['APG_cross_attn_wo_x'],
                            residual=args['APG_residual'], bayes=False)
        
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
        self.task_tokens = copy.deepcopy(self.image_encoder.cls_token)
        
        self.keys=tensor_prompt(self.args["num_classes"], self.image_encoder.embed_dim, ortho=True)
        self.knowladge = tensor_prompt(1, 100, 768)
        # self.knowladge = torch.nn.Parameter(torch.FloatTensor(1,200,768), requires_grad=True).to('cuda')
        clip_model = load_clip_to_cpu()
        self.text_encoder = TextEncoder(clip_model)
        self.key_task = PromptLearner(args=args, clip_model=clip_model, class_list=self.class_name)
        self.tokenized_prompts_key_task_class = self.key_task.tokenized_prompts_class
        self.class_feature = torch.nn.Parameter(torch.zeros(200,768), requires_grad=False)

        for name, param in self.image_encoder.named_parameters():
            if name.startswith('APG'):
                param.requires_grad=True
            else:
                param.requires_grad=False
                param.grad=None

    def update_fc(self,nb_classes,cur_task_nbclasses):
        self.aux_cla=self.generate_fc(self.image_encoder.embed_dim,cur_task_nbclasses)

        cla_w=self.generate_fc(self.image_encoder.embed_dim,cur_task_nbclasses)
        self.clas_w.append(cla_w)
 
        vitprompt_1=nn.Linear(self.image_encoder.embed_dim, 50, bias=False)
        
        self.ts_prompts_1.append(vitprompt_1)
        vitprompt_2=nn.Linear(self.image_encoder.embed_dim, 50, bias=False)
        self.ts_prompts_2.append(vitprompt_2)

        if len(self.clas_w)>1:
            self.ts_prompts_1[-1].load_state_dict(self.ts_prompts_1[-2].state_dict())
            self.ts_prompts_2[-1].load_state_dict(self.ts_prompts_2[-2].state_dict())
        
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def aux_forward(self, image, sample):
        i=len(self.clas_w)-1

        prompts_class, prompts_task = self.key_task()
        tokenized_prompts_class = self.tokenized_prompts_key_task_class
        text_features_class = self.text_encoder(prompts_class, tokenized_prompts_class)
        self.class_feature.data.copy_(text_features_class)

        image_features = self.image_encoder(image, instance_tokens=self.ts_prompts_1[i].weight,second_pro=self.ts_prompts_2[i].weight, returnbeforepool=True, gen_pro=None, task_id=i, sample=sample, knowladge=self.knowladge)
        feature=image_features[:,0,:]
        logits=self.aux_cla(feature)['logits']
        return logits,feature

    def forward(self, image, gen_p, train, task_id, sample):
        prompts_class, prompts_task = self.key_task()
        tokenized_prompts_class = self.tokenized_prompts_key_task_class
        text_features_class = self.text_encoder(prompts_class, tokenized_prompts_class)
        self.class_feature.data.copy_(text_features_class)

        image_features = self.image_encoder(image, instance_tokens=gen_p[0],second_pro=gen_p[1], returnbeforepool=True, gen_pro=None, task_id=task_id, sample=sample, knowladge=self.knowladge)
        feature=image_features[:,0,:]
        if train:
            i=len(self.clas_w)-1
            return self.clas_w[i](feature)['logits']
        for i in range(len(self.clas_w)):
            if i==0:
                logits=self.clas_w[i](feature)['logits']
            else:
                logit=self.clas_w[i](feature)['logits']
                logits=torch.cat((logits,logit),1)
        return logits
    
    def fix_branch_layer(self):
        for param in self.clas_w.parameters():
            param.requires_grad=False
            param.grad=None

        for param in self.ts_prompts_1.parameters():
            param.requires_grad=False
            param.grad=None
            
        for param in self.ts_prompts_2.parameters():
            param.requires_grad=False
            param.grad=None
        
class ViT_KPrompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, MLP_num= 1,
            num_heads_APG=4,proj_drop=0,attn_drop=0,more_prompts=1,attn_depth=1,use_update_tokens=False,cross_attn_wo_x=True,
            residual=True, bayes=False):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, MLP_num= 1,
            num_heads_APG=4,proj_drop=0,attn_drop=0,more_prompts=1,attn_depth=1,use_update_tokens=False,cross_attn_wo_x=False,
            residual=False, bayes=False)

    def forward(self, x, instance_tokens=None, second_pro=None, returnbeforepool=False,gen_pro=None, task_id=None, sample=None, knowladge=None, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed.to(x.dtype)
        x = self.pos_drop(x)
        # if gen_pro is None:
        #     if instance_tokens is not None and instance_tokens.shape[0]!=16:
        #         instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        #     x = x + self.pos_embed.to(x.dtype)
        #     if instance_tokens is not None:
        #         x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
        #     x = self.pos_drop(x)
        #     # print("x:",x.shape)
        #     x=self.blocks[:5](x)
        #     if second_pro is not None:
        #         second_pro=second_pro.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device)
        #         x = torch.cat([x[:,:1+instance_tokens.shape[1],:], second_pro, x[:,1+instance_tokens.shape[1]:,:]], dim=1)
        #     x=self.blocks[5:](x)

        for i, block in enumerate(self.blocks):
                if i == 0:
                    x = block(x)
                else:
                    res = self.APG[i](x[:,0,:].unsqueeze(1), text_tokens = knowladge, task_id=task_id, sample=sample)
                    e_prompt = res['e_prompt']
                    # print(e_prompt)
                    x = block(x, e_prompt)
            
        # else:
        #     for i in range(len(instance_tokens)):
        #         if instance_tokens[i].shape[1]==768:
        #             instance_tokens[i]=instance_tokens[i].to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        #         else:
        #             instance_tokens[i]=instance_tokens[i].to(x.dtype)
            
        #     x = x + self.pos_embed.to(x.dtype)
        #     x = torch.cat([x[:,:1,:], instance_tokens[0], x[:,1:,:]], dim=1)
        #     x = self.pos_drop(x)
        #     x=self.blocks[0](x)
        #     for i in range(len(instance_tokens)-1):
        #         x = torch.cat([x[:,:1+instance_tokens[0].shape[1]*(i+1),:], instance_tokens[i+1], x[:,1+instance_tokens[0].shape[1]*(i+1):,:]], dim=1)
        #         x=self.blocks[i+1](x)
        #     x=self.blocks[len(instance_tokens):](x)


        if returnbeforepool == True:
            return x
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x
    