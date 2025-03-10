from torch import nn
import torch

class FullQueryLayer(nn.Module):
    def __init__(self):
        super(FullQueryLayer, self).__init__()

    def forward(self, x,K):
        """
        计算self-cost volume V
        Args:
            x: feature map S of shape (B, E, H, W)
            K: queries Q of shape (B, Q, E)
        Returns:
            energy map: Q 个查询对应的能量图 of shape (B, Q, H,W)
            summary_embedding: 总结后的嵌入表示 of shape (B, Q, E)
        """
        n,c,h,w = x.shape  # B,E,H,W
        _,cout,ck = K.shape  # B,Q,E
        assert c == ck, "输入特征图 x 的通道数必须与查询矩阵 K 的嵌入维度匹配"
        # 计算查询对特征图的相似度
        # 先将x变换为形状(B,H*W,E)(展平空间维度)
        # 进行矩阵乘法，得到[B,H*W,Q]的能量分布
        y = torch.matmul(  # [B,H*W,Q]
            x.view(n,c,h*w).permute(0,2,1),# [B,H*W,E]
            K.permute(0,2,1)) # [B,E,Q]
        # softmax归一化，得到注意力权重
        y_norm = torch.softmax(y,dim=1)
        # 计算总结嵌入 summary_embedding
        # 1. 交换 y_norm 的维度，形状变为 [B, Q, H*W]
        # 2. 计算权重和输入特征的加权和，得到形状 [B, Q, E] 的总结嵌入
        summary_embedding = torch.matmul(
            y_norm.permute(0,2,1), # [B, Q, H*W]
            x.view(n,c,h*w).permute(0, 2, 1)  #  [B,H*W,E]
        )
        # 重新调整 y 的形状，使其匹配 [B, Q, H, W]
        y = y.permute(0,2,1).view(n,cout,h,w)

        return y,summary_embedding


class DepthDecoderQueryTr(nn.Module):
    """
    深度解码器-->使用SQLDepth的自查询层(Self Query Layer)来恢复深度信息
    """
    def __init__(self,in_channels,
                 embedding_dim=32,patch_size=32,
                 num_heads=4,query_nums=128,
                 dim_out=128,norm='linear',
                 min_val=0.001,max_val=80) -> None:
        super(DepthDecoderQueryTr, self).__init__()
        self.norm = norm
        self.query_nums = query_nums
        self.min_val = min_val
        self.max_val = max_val
        # Corase-grained queries Q
        # get a feature map F of shape C x h/p x h/p
        self.embedding_convPxP = nn.Conv2d(in_channels,embedding_dim,
                                           kernel_size=patch_size,stride=patch_size,padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500,embedding_dim), requires_grad=True)
        # mini-transformer of 4 layers to generate a set coarse-grained queries Q of shape R^{CxQ}
        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(
            embedding_dim,num_heads,dim_feedforward=1024)
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(encoder_layers,num_layers=4)
        # get self-cost volume V
        self.conv3x3 = nn.Conv2d(in_channels,embedding_dim,
                                 kernel_size=3,stride=1,padding=1)
        self.full_query_layer = FullQueryLayer()
        # a MLP to regress the depth bins b
        self.bins_regressor = nn.Sequential(
            nn.Linear(embedding_dim*query_nums,16*query_nums),
            nn.LeakyReLU(),
            nn.Linear(16*query_nums,16*16),
            nn.LeakyReLU(),
            nn.Linear(16*16,dim_out)
        )
        # probabilistic combination
        # a 1x1 convolution to the self volume V to obtain a D-planes volumes
        # a plane-wise softmax operation to convert the volume into plane-wise probabilistic map
        self.convert_to_prob = nn.Sequential(
            nn.Conv2d(query_nums,dim_out,
                      kernel_size=1,stride=1,padding=0),
            nn.Softmax(dim=1))

    def forward(self,S):
        """
        Args:
            S: high resolution immediate features S of shape (B, C, h, w)
        Returns:
        """
        # 1. 计算粗粒度查询Q
        # apply a convolution of kernel size pxp and stride=p to S get a feature map F of shape (B,C,h/p,w/p)
        F = self.embedding_convPxP(S.clone())
        # reshape F to (B,C,N)
        F = F.flatten(2)
        # add positional embeddings to F
        F = F + self.positional_encodings[:F.shape[2],:].T.unsqueeze(0)
        # reshape F to (N,B,C)方便后边计算
        F = F.permute(2,0,1)

        # feed these patch embeddings into a mini-transformer of 4 layers to generate as set of coarse-grained queries Q of shape R^{C,Q}
        total_queries = self.transformer_encoder(F)
        # get self-cost volume V = total_queries^T \cdot S
        S = self.conv3x3(S)
        queries = total_queries[:self.query_nums,...]
        queries = queries.permute(1,0,2)
        # summarys用于计算箱子宽度，energy_maps用于计算像素落在该箱子的概率
        energy_maps,summarys = self.full_query_layer(S,queries)

        #2. 预测深度bins
        B,Q,E = summarys.shape
        bins = self.bins_regressor(summarys.view(B,Q*E))

        #3. 归一化深度分箱
        if self.norm=='softmax':
            return torch.softmax(bins,dim=1),energy_maps
        elif self.norm=='linear':
            bins = torch.relu(bins)
            eps = 0.1
            bins = bins+eps
        else:
            bins = torch.sigmoid(bins)
        bins = bins / bins.sum(dim=1,keepdim=True)
        #4. 计算最终的深度估计
       # 转换自成本体积为概率
        out = self.convert_to_prob(energy_maps)
        # 深度区间
        bin_widths = (self.max_val - self.min_val)*bins
        bin_widths = nn.functional.pad(bin_widths,(1,0),
                                       mode='constant',value=self.min_val)
        bin_deges = torch.cumsum(bin_widths,dim=1)

        # 深度中心
        centers = 0.5*(bin_deges[:,:-1]+bin_deges[:,1:])
        n,dout = centers.size()
        centers = centers.view(n,dout,1,1)

        # 预测最终深度
        pred = torch.sum(out*centers,dim=1,keepdim=True)
        # 返回深度图
        return pred









