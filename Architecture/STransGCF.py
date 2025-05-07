import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from einops import rearrange
import numpy as np


class EGA_block(nn.Module):
    """
    Efficient Global Attention
    """
    def __init__(self, k_size, img_size, window_size, in_channel, out_channel):
        super(EGA_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=img_size//window_size)
        # self.conv_win = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                           kernel_size= window_size, stride=window_size)
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        num_windows = img_size//window_size * img_size//window_size
        # self.mlp = nn.Linear(in_features=num_windows, out_features=num_windows)
        self.sigmoid = nn.Sigmoid()
        self.img_size = img_size
        self.window_size = window_size

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        y = self.avg_pool(x)
        # y = self.conv_win(x)
        y = self.conv(y.view(b, c, -1))
        # y = self.mlp(y.view(b, c, -1))
        y = self.sigmoid(y)
        win_num_row = self.img_size//self.window_size
        x_loc = x.clone()
        for i in range(4):
            for j in range(4):
                window = x[:, :, self.window_size * i:self.window_size * (i + 1),
                self.window_size * j:self.window_size * (j + 1)]
                # print(window.shape)
                weight = y[:,:,i*win_num_row+j].unsqueeze(-1).unsqueeze(-1)
                # print(weight.shape)
                window = window * weight
                x_loc[:, :, self.window_size * i:self.window_size * (i + 1),
                self.window_size * j:self.window_size * (j + 1)] = window
        return x_loc
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, patch_size, embedded_dim=48, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        n_patches = (img_size//patch_size)*(img_size//patch_size)
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=embedded_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, embedded_dim))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # print(dim," ",hidden_dim)
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 6, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print(q.shape,k.shape,v.shape,self.heads)

        # all_relative = torch.index_select(self.relative, 1, self.relative_index)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# class Lo_block(nn.Module):
#     def __init__(self, img_size, window_size, in_channel, dropout = 0.):


class LA_block(nn.Module):
    def __init__(self, img_size, num_window, patch_size,
                 embedded_dim, in_channel, dropout = 0., mlp_dim=128):
        super(LA_block, self).__init__()
        # embedded_dim = patch_size * patch_size * in_channel
        mlp_dim = embedded_dim*4
        self.img_size = img_size
        self.window_size = img_size//int(np.sqrt(num_window))
        self.embedding = Embeddings(self.window_size, patch_size, embedded_dim, in_channel)
        self.attn = PreNorm(embedded_dim, Attention(embedded_dim, heads=8, dim_head=6, dropout=dropout))
        self.ff = PreNorm(embedded_dim, FeedForward(embedded_dim, mlp_dim, dropout))

    def forward(self, x):
        b,c,_,_ = x.shape
        x_loc = x.clone()
        for i in range(4):
            for j in range(4):
                x_p = x[:, :, self.window_size * i:self.window_size * (i + 1),
                      self.window_size * j:self.window_size * (j + 1)]
                # print(self.window_size)
                emb = self.embedding(x_p)
                emb = self.attn(emb) + emb;
                emb = self.ff(emb) + emb
                img_p = emb.view(b, c, self.window_size, self.window_size)
                x_loc[:, :, self.window_size * i:self.window_size * (i + 1),
                self.window_size * j:self.window_size * (j + 1)] = img_p
        return x_loc
class STrans(nn.Module):
    def __init__(self, img_size, num_window, patch_size, embedded_dim,
                 in_channel, out_channel=3, dropout = 0., mlp_dim=128):
        super(STrans, self).__init__()
        self.img_size = img_size
        self.window_size = img_size // int(np.sqrt(num_window))
        self.ega = EGA_block(k_size=5, img_size=img_size, window_size=self.window_size,
                             in_channel=in_channel, out_channel=in_channel)
        self.la = LA_block(img_size=img_size, num_window=num_window, patch_size=patch_size,
                           embedded_dim=embedded_dim, in_channel=in_channel, mlp_dim=mlp_dim, dropout=dropout)
        self.conv1 = Conv2dReLU(
            in_channels=in_channel*2,
            out_channels=in_channel,
            kernel_size=3,
            padding=1
        )
        self.conv2 = Conv2dReLU(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1
        )
        self.aciv = nn.ReLU()


    def forward(self, x):
        g_att = self.ega(x)
        # g_att = self.ga(x)
        l_att = self.la(x)

        res = torch.cat([g_att, l_att],dim=1)
        # res = g_att
        res = self.conv1(res)
        res = self.conv2(res)
        res = res + x
        return self.aciv(res)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class STransGCF_Eblock(nn.Module):
    def __init__(self, img_size, num_window, patch_size, in_channel, out_channel,
                 dropout, mlp_dim):
        super(STransGCF_Eblock, self).__init__()
        embeded_dim = patch_size*patch_size*in_channel
        self.lg = STrans(img_size, num_window=num_window, patch_size=patch_size,
                            embedded_dim=embeded_dim, in_channel=in_channel, out_channel=in_channel,
                            dropout=dropout, mlp_dim=mlp_dim)
        self.down_sample_conv = Conv2dReLU(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        lg_res = self.lg(x)#最终
        conv_res = self.down_sample_conv(lg_res) #最终
        return lg_res, conv_res

class GCA(nn.Module):
    def __init__(self, in_channel, k_size):
        super(GCA, self).__init__()
        self.indice = torch.tensor([i // 2 + in_channel * (i % 2) for i in range(in_channel*2)]).long()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gate1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.gate2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2d = Conv2dReLU(in_channels=in_channel*2, out_channels=in_channel,
                       kernel_size=3, padding=1, stride=1)

    def forward(self, x1, x2):
        x1 = x1*self.gate1
        x2 = x2*self.gate2

        cat_feature = torch.cat([x1, x2], dim=1)
        cat_feature = cat_feature[:, self.indice, :, :]
        avg = self.avg_pool(cat_feature)
        max = self.max_pool(cat_feature)
        y = (avg+max)/2
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        sig = self.sigmoid(y)
        res = cat_feature * sig.expand_as(cat_feature)
        ret = self.conv2d(res)
        return ret

class STransGCF(nn.Module):
    def __init__(self, config=None, img_size=512, num_window=16, patch_size=4,
                 in_channel=3, num_classes=2, dropout=0.1, mlp_dim=1024):
        super(STransGCF, self).__init__()
        self.num_classes = num_classes;
        outchannel_1 = in_channel*2
        outchannel_2 = in_channel*4
        outchannel_3 = in_channel*8
        outchannel_4 = in_channel*16
        self.encoder1 = STransGCF_Eblock(img_size=img_size, num_window=num_window, patch_size=patch_size,
                                    in_channel=in_channel, out_channel=outchannel_1,
                                    dropout=dropout, mlp_dim=mlp_dim)
        self.encoder2 = STransGCF_Eblock(img_size=img_size//2, num_window=num_window, patch_size=patch_size,
                                    in_channel=outchannel_1, out_channel=outchannel_2,
                                    dropout=dropout, mlp_dim=mlp_dim)

        self.encoder3 = STransGCF_Eblock(img_size=img_size//4, num_window=num_window, patch_size=patch_size,
                                    in_channel=outchannel_2, out_channel=outchannel_3,
                                    dropout=dropout, mlp_dim=mlp_dim)
        self.encoder4 = STransGCF_Eblock(img_size=img_size//8, num_window=num_window, patch_size=patch_size,
                                    in_channel=outchannel_3, out_channel=outchannel_4,
                                    dropout=dropout, mlp_dim=mlp_dim)

        self.mid = nn.Sequential(
            Conv2dReLU(in_channels=outchannel_4  , out_channels=outchannel_4,
                       kernel_size=3, padding=1, stride=1),
            Conv2dReLU(in_channels=outchannel_4, out_channels=outchannel_4,
                       kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(outchannel_4, outchannel_3, kernel_size=3,
                               stride=2, padding=1, output_padding=1)
        )


        self.up3 = nn.ConvTranspose2d(in_channels=outchannel_3, out_channels=outchannel_2, kernel_size=3,
                               stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=outchannel_2, out_channels=outchannel_1, kernel_size=3,
                               stride=2, padding=1, output_padding=1)
        self.up1 = nn.ConvTranspose2d(in_channels=outchannel_1, out_channels=in_channel, kernel_size=3,
                               stride=2, padding=1, output_padding=1)
        self.gca3 = GCA(in_channel=outchannel_3, k_size=7)
        self.gca2 = GCA(in_channel=outchannel_2, k_size=7)
        self.gca1 = GCA(in_channel=outchannel_1, k_size=5)
        self.gca0 = GCA(in_channel=in_channel, k_size=5)

        self.out = nn.Sequential(
            Conv2dReLU(in_channels=in_channel, out_channels=self.num_classes,
                       kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        lg_res1, en_res1 = self.encoder1(x)
        lg_res2, en_res2 = self.encoder2(en_res1)
        lg_res3, en_res3 = self.encoder3(en_res2)
        lg_res4, en_res4 = self.encoder4(en_res3)

        mid = self.mid(en_res4)

        f3 = self.up3(self.gca3(lg_res4, mid))
        f2 = self.up2(self.gca2(lg_res3, f3))
        f1 = self.up1(self.gca1(lg_res2, f2))
        f0 = self.out(self.gca0(lg_res1, f1))

        return f0