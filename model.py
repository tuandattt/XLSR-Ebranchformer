import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
from Ebranchformer import EbranchformerBlock
from MultiConvFormer import MultiConvformerBlock
from pooling import MultiHeadAttentionPooling
from fairseq.fairseq import checkpoint_utils
from types import SimpleNamespace
from thop import profile

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones(ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, list_attn_weight

class MySwitchTransformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MySwitchTransformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones(SwitchTransformerBlock(dim = emb_size, dim_head=self.dim_head, heads= heads, mult = ffmult, num_experts=4), n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, list_attn_weight

class MyEbranchformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, n_encoders=1):
    super(MyEbranchformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones(EbranchformerBlock(input_size = emb_size, output_size = emb_size, attention_heads = heads, dim_head = self.dim_head), n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    B = x.shape[0]
    cls_tokens = self.class_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x, mask = None) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding=x[:,0,:]
    #embedding = x
    out=self.fc5(embedding) #[bs,2]
    return out, list_attn_weight

class ClassifierNoClassToken(nn.Module):
  def __init__(self, emb_size=128, heads=4, n_encoders=1):
    super(ClassifierNoClassToken, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones(EbranchformerBlock(input_size = emb_size, output_size = emb_size, attention_heads = heads, dim_head = self.dim_head), n_encoders)
    self.last_pooling = MultiHeadAttentionPooling(emb_size)
    self.fc5 = nn.Linear(emb_size*2, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x, mask = None) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding = torch.sum(x, 1)
    embedding = embedding.unsqueeze(1)
    embedding = self.last_pooling(embedding.permute(0, 2, 1))
    out = self.fc5(embedding.squeeze(-1))
    return out, list_attn_weight

class SSLModel(nn.Module): #W2V
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = '/home/stud_dat/tcm_add/checkpoint/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        self.agg = 'SEA'

        if self.agg == 'SEA':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_att_merge = nn.Sequential(
                nn.Linear(24, 8, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(8, 24, bias=False),
                nn.Sigmoid()
            )
        elif self.agg == 'WeightedSum':
            self.weight_hidd = nn.Parameter(torch.ones(24))  # Initialize weights for weighted sum
        elif self.agg == 'AttM':
            self.n_feat = self.out_dim
            self.W = nn.Parameter(torch.randn(self.n_feat, 1))
            self.W1 = nn.Parameter(torch.randn(24, int(24 // 2)))
            self.W2 = nn.Parameter(torch.randn(int(24 // 2), 24))
            self.hidden = int(24 * self.n_feat / 4)
            self.linear_proj = nn.Linear(24 * self.n_feat, self.n_feat)
            self.SWISH = nn.SiLU()

        return
    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True, layer=24)

        if self.agg == 'SEA':
            return self._SE_merge(emb)
        elif self.agg == 'WeightedSum':
            return self._weighted_sum(emb)
        elif self.agg == 'AttM':
            return self._Att_merge(emb)
        else:
            return emb['x']

    def _weighted_sum(self, x):
        feature = [layer_tuple[0].transpose(0,1) for layer_tuple in x['layer_results']]
        layer_num = len(feature)
        stacked_feature = torch.stack(feature, dim=0)
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weight_hidd[:layer_num], dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        return weighted_feature

    def _SE_merge(self, x):
        feature = [layer_tuple[0].transpose(0,1) for layer_tuple in x['layer_results']]
        stacked_feature = torch.stack(feature, dim=1)
        b, c, _, _ = stacked_feature.size()
        y = self.avg_pool(stacked_feature).view(b, c)
        y = self.fc_att_merge(y).view(b, c, 1, 1)
        stacked_feature = stacked_feature * y.expand_as(stacked_feature)
        weighted_feature = torch.sum(stacked_feature, dim=1)
        return weighted_feature

    def _Att_merge(self, x):
        feature = [layer_tuple[0].transpose(0,1) for layer_tuple in x['layer_results']]
        x = torch.stack(feature, dim=1)
        x_input = x
        x = torch.mean(x, dim=2, keepdim=True)  # X2 = AVG(X1) AVG across time dim
        x = self.SWISH(torch.matmul(x, self.W))  # X3
        x = self.SWISH(torch.matmul(x.view(-1, 24), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2)))  # X4
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input)  # X5
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1)  # concatenate
        weighted_feature = self.linear_proj(x)
        return weighted_feature

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        #self.conformer=MySwitchTransformer(emb_size=args.emb_size, n_encoders=args.num_encoders, heads=args.heads)
        self.conformer = MyEbranchformer(emb_size = args.emb_size, n_encoders = args.num_encoders, heads = args.heads)
        #self.conformer = ClassifierNoClassToken(emb_size = args.emb_size, n_encoders = args.num_encoders, heads = args.heads)
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score =self.conformer(x,self.device)
       
        return out, attn_score

if __name__ == "__main__":
    args = SimpleNamespace(
        emb_size=256,
        num_encoders=6,
        heads=8
    )
    model = Model(args, "cuda")
    x = torch.rand((2, 66400)).to("cuda")
    macs, params = profile(model, inputs=(x,))
    print(params)
    print(macs*2)
