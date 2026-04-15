from collections import OrderedDict

import ucf_option
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from mamba import Model as mamba
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):  # 输入通道数，中间通道数（若无参数，则默认为输入通道数的一半），维度，是否下采样，是否使用BN层
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]  # 维度只能为1维，或2维，或3维
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        # 根据dimension参数，选择不同的卷积和池化层和批量归一化层bn
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,   # ？？？？？？？？不是很懂  gpt说：用于生成关键特征的卷积层
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)  # W[1].weight设置为0
            nn.init.constant_(self.W[1].bias, 0)    # W[1].bias设置为0
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)  # W.weight设置为0
            nn.init.constant_(self.W.bias, 0)    # W.bias设置为0

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # F_c1 in paper
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  #  F_c2 in paper

        f = torch.matmul(theta_x, phi_x)  # M=(F_c1)(F_c2)^T
        N = f.size(-1)  # 取最后一维的长度作为N
        f_div_C = f / N  #直接使用f/N作为归一化因子

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()  #contigous()函数确保张量在内存中是连续的
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)  # F_c4=Conv1x1(MF_c3)
        z = W_y + x  # A skip connection is added, z is F_TSA

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):  # 输入通道数，中间通道数，是否下采样，是否使用BN层
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):  # MTN
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d  # 创建一维批量归一化层
        self.len_feature = len_feature

        # conv_1, conv_2, conv_3:使用了不同扩张率的卷积核，实现金字塔扩展卷积（PDC）以捕捉不同尺度的依赖关系
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(    # 这里的卷积核大小为1
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False),  # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(int(len_feature/4), sub_sample=False, bn_layer=True)
        self.sum = 0

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = x.float()
        out = x.permute(0, 2, 1)
        residual = out

        # The module on the left uses the pyramid dilated convolutions to capture the local consecutive snippets
        # dependency over different temporal scales
        out1 = self.conv_1(out)  # PDC1
        out2 = self.conv_2(out)  # PDC2
        out3 = self.conv_3(out)  # PDC3
        out_d = torch.cat((out1, out2, out3), dim=1)

        # The module on the right relies on a self-attention network to compute the global temporal correlations
        out = self.conv_4(out)
        out = self.non_local(out)  # TSA

        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)  # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)
        return out
class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

args = ucf_option.parser.parse_args()

class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 configs,
                 device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.configs = configs
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.word_linear = nn.Linear(visual_width, 1280)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width + args.text_features, (visual_width + args.text_features) * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear((visual_width + args.text_features) * 4, (visual_width + args.text_features)))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width + args.text_features, (visual_width + args.text_features) * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear((visual_width + args.text_features) * 4, visual_width + args.text_features))
        ]))
        self.mlp3 = nn.Sequential(OrderedDict([
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear((visual_width + args.text_features), visual_width))
        ]))
        self.classifier = nn.Linear(visual_width + args.text_features, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()
        self.mamba = mamba(self.configs)
        self.Aggregate = Aggregate(args.visual_width)
        self.Aggregate_text = Aggregate(args.text_features)
        self.drop_out = nn.Dropout(0.7)
    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask


    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)

        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)  #
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output



    def LGM_Mamba(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        x = self.mamba(x)
        x = self.Aggregate(x)
        x = self.drop_out(x)
        return x

    def fusion(self,visual, text):
        if visual.shape[1] < text.shape[1]:
            text = text[:, :(visual.shape[1] - text.shape[1]), :]
        elif visual.shape[1] > text.shape[1]:
            text = torch.cat((text, text[:, (text.shape[1] - visual.shape[1]):, :]), dim=1)
        visual = torch.cat([visual, text], dim=2)
        return visual


    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)


        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        text_features = self.word_linear(text_features)

        return text_features



    def forward(self, visual, text, padding_mask, word, lengths):
        visual_features = self.LGM_Mamba(visual, padding_mask, lengths)
        text_features = self.Aggregate_text(text)
        text_features = self.drop_out(text_features)
        fusion_features = self.fusion(visual_features, text_features)

        logits1 = self.classifier(fusion_features + self.mlp2(fusion_features))
        word_features_ori = self.encode_textprompt(word)
        logits_attn = logits1.permute(0, 2, 1)

        visual_attn = logits_attn @ fusion_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], word_features_ori.shape[0], visual_attn.shape[2])
        word_features = word_features_ori.unsqueeze(0)
        word_features = word_features.expand(visual_attn.shape[0], word_features.shape[1], word_features.shape[2])
        word_features = word_features + visual_attn
        word_features = word_features + self.mlp1(word_features)

        fusion_features_norm = fusion_features / fusion_features.norm(dim=-1, keepdim=True)
        word_features_norm = word_features / word_features.norm(dim=-1, keepdim=True)
        word_features_norm = word_features_norm.permute(0, 2, 1)
        logits2 = fusion_features_norm @ word_features_norm.type(fusion_features_norm.dtype) / 0.07

        return word_features_ori, logits1, logits2