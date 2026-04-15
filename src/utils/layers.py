from math import sqrt
from torch import FloatTensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear, self).__init__()
        self.weight = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = x.matmul(self.weight)
        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, residual=True):  # in_features是输入特征, out_features输出特征, bias是否使用偏执项, residual是否使用残差连接
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))  # 创建一个可训练的参数 weight，其初始化为一个 in_features x out_features 的浮点张量。
        if bias:  #如果 bias 参数为 True
            self.bias = Parameter(FloatTensor(out_features))   # 创建一个可训练的参数 bias，其初始化为一个 out_features 的浮点张量。
        else:  # 如果 bias 参数为 False，则注册一个 None 参数，表示没有偏置项。
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0  #如果不使用残差连接（not residual），则残差函数是一个恒等于0的函数。
        elif (in_features == out_features):
            self.residual = lambda x: x  #如如果输入和输出特征数量相同（in_features == out_features），则残差函数是一个恒等函数。
        else:
            # self.residual = linear(in_features, out_features)
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)  #如果输入和输出特征数量不同（in_features != out_features），则使用一个一维卷积层作为残差函数。
    def reset_parameters(self):  #重置参数
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight)  #将 weight 参数初始化为一个 xavier_uniform 随机初始化。
        if self.bias is not None:  #如 bias 参数不为 None
            self.bias.data.fill_(0.1)  #将 bias 参数初始化为 0.1。

    def forward(self, input, adj):  #input是输入特征，adj是邻接矩阵。
        # To support batch operations
        support = input.matmul(self.weight)  #计算输入特征和权重矩阵的矩阵乘法，得到支持向量。
        output = adj.matmul(support)  #使用邻接矩阵和支持向量进行矩阵乘法，实现图卷积操作。

        if self.bias is not None:  #若有偏置项，则加上偏置项。
            output = output + self.bias  #加上偏置项。
        if self.in_features != self.out_features and self.residual:  #如输入和输出特征数量不同且使用残差连接，则将输入特征和残差连接后的特征进行拼接。
            input = input.permute(0,2,1)  #变换特征
            res = self.residual(input)  #计算残差特征。
            res = res.permute(0,2,1)  #变回输入特征的维度。
            output = output + res  #拼接。
        else:
            output = output + self.residual(input)  #除了上述情况以外其他情况都直接将输入特征和残差连接后的特征进行相加。

        return output  #输出结果。

    def __repr__(self):  #'类名（输入特征 -> 输出特征）'    如果有一个 GraphConvolution 类的实例，其输入特征为 64，输出特征为 128，则打印实例时会显示：'GraphConvolution (64 -> 128)'
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

######################################################

class SimilarityAdj(Module):

    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input, seq_len):
        # To support batch operations
        theta = torch.matmul(input, self.weight0)
        phi = torch.matmul(input, self.weight0)
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        output = torch.zeros_like(sim_graph)
        if seq_len is None:
            for i in range(sim_graph.shape[0]):
                tmp = sim_graph[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = F.softmax(adj2, dim=1)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = F.softmax(adj2, dim=1)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



"""
            这个类可能用于创建一个距离敏感的邻接矩阵，用于图卷积网络或其他需要考虑节点间距离的应用。
            通过将距离矩阵通过高斯核函数转换，可以使得模型在处理节点间距离时更加平滑和连续。
"""
class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):  # 传入批次和序列的最大长度
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)  # 生成0-max_seqlen-1的序列，并且将其变形为列向量。
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)  # 计算距离矩阵，并将其转化为浮点数。‘cityblock’表示采用曼哈顿距离作为距离度量。也称为“城市街区距离”
        self.dist = torch.from_numpy(squareform(dist)).to('cuda')  # 将距离矩阵变形方阵，然后转化为张量。
        # 对距离张量应用高斯核函数，其中sigma是高斯核的参数。这里使用指数函数来实现高斯核的计算。
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))  # 进行归一化的权重调整
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to('cuda')  # 扩展张量的维度，使其与输入张量具有相同的批次大小。
        return self.dist

if __name__ == '__main__':
    d = DistanceAdj()
    dist = d(1, 256).squeeze(0)
    print(dist.softmax(dim=-1))