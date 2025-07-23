import mlx.core as mx
import mlx.nn as nn

class OverlapPatchEmbeddings(nn.Module):
    """This class turns a 2D image into a patch embedding."""
    def __init__(self, patch_size: int, stride: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(0, 2, 1)
        x = self.layer_norm(x)
        return x, H, W

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(0, 2, 1).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(0, 2, 1)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SegformerLayer(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, hidden_features=int(dim * mlp_ratio))

    def __call__(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class SegformerEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims, num_heads, mlp_ratios, sr_ratios, depths):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                patch_embed = OverlapPatchEmbeddings(7, 4, in_channels, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbeddings(3, 2, embed_dims[i-1], embed_dims[i])

            block = nn.ModuleList([SegformerLayer(embed_dims[i], num_heads[i], sr_ratios[i], mlp_ratios[i]) for _ in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            self.encoder_layers.append(nn.ModuleList([patch_embed, block, norm]))

    def __call__(self, x):
        features = []
        B, _, _, _ = x.shape
        for patch_embed, block, norm in self.encoder_layers:
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
            features.append(x)
        return features

class SegformerDecoder(nn.Module):
    def __init__(self, embed_dims, num_classes, decoder_dim):
        super().__init__()
        self.linear_c = nn.ModuleList([nn.Linear(embed_dims[i], decoder_dim) for i in range(len(embed_dims))])
        self.linear_fuse = nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, 1)
        self.batch_norm = nn.BatchNorm(decoder_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, 1)

    def __call__(self, features):
        B, _, H, W = features[0].shape
        all_features = []
        for i, feature in enumerate(features):
            feature = feature.flatten(2).transpose(0, 2, 1)
            feature = self.linear_c[i](feature)
            feature = feature.transpose(0, 2, 1).reshape(B, -1, H, W)
            feature = nn.Upsample(size=(H, W), scale_factor=None, mode='bilinear')(feature)
            all_features.append(feature)

        x = mx.concatenate(all_features, axis=1)
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x

class Segformer(nn.Module):
    def __init__(self, in_channels, embed_dims, num_heads, mlp_ratios, sr_ratios, depths, num_classes, decoder_dim):
        super().__init__()
        self.encoder = SegformerEncoder(in_channels, embed_dims, num_heads, mlp_ratios, sr_ratios, depths)
        self.decoder = SegformerDecoder(embed_dims, num_classes, decoder_dim)

    def __call__(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = nn.Upsample(scale_factor=4.0, mode='bilinear')(x)
        return x

def segformer_b0(num_classes):
    return Segformer(
        in_channels=3,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        depths=[2, 2, 2, 2],
        num_classes=num_classes,
        decoder_dim=256,
    )

def segformer_b1(num_classes):
    return Segformer(
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        depths=[2, 2, 2, 2],
        num_classes=num_classes,
        decoder_dim=256,
    )
