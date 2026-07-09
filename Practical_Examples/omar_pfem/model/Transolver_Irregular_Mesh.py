import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from omar_pfem.model.Embedding import timestep_embedding
import numpy as np
from omar_pfem.model.Physics_Attention import Physics_Attention_Irregular_Mesh, Physics_Attention_extraction, Physics_Attention_Irregular_Mesh_bc

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class Transolver_block_bc(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1_bc = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh_bc(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, key_bc, value_bc):
        fx = self.Attn(self.ln_1(fx), key_bc, value_bc) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class Transolver_block_extraction(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_extraction(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        out = self.Attn(self.ln_1(fx))
        return out


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False, B = False, bc_dim = 1
                 ):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_1D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
            if B:
                self.preprocess_bc = MLP(space_dim + bc_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
        if B is False:
            self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                        dropout=dropout,
                                                        act=act,
                                                        mlp_ratio=mlp_ratio,
                                                        out_dim=out_dim,
                                                        slice_num=slice_num,
                                                        last_layer=(_ == n_layers - 1))
                                        for _ in range(n_layers)])
        else:
            self.ln_bc = nn.LayerNorm(n_hidden)
            self.blocks_b = Physics_Attention_extraction(n_hidden, heads=n_head, dim_head=n_hidden // n_head, dropout=dropout, slice_num=slice_num)
            self.blocks = nn.ModuleList([Transolver_block_bc(num_heads=n_head, hidden_dim=n_hidden,
                                                        dropout=dropout,
                                                        act=act,
                                                        mlp_ratio=mlp_ratio,
                                                        out_dim=out_dim,
                                                        slice_num=slice_num,
                                                        last_layer=(_ == n_layers - 1))
                                        for _ in range(n_layers)])


        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.placeholder_bc = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        # x: B N 2
        # grid_ref
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda().reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, bc=None, bfx=None, T=None):
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :] # placeholder在模型的计算中添加额外的自由度

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        if bc is not None:
            bfx = torch.cat((bc, bfx), -1)
            bfx = self.preprocess_bc(bfx)
            bfx = bfx + self.placeholder_bc[None, None, :]
            key_bc, value_bc = self.blocks_b(self.ln_bc(bfx)) # 特征提取
            for block in self.blocks:
                fx = block(fx, key_bc, value_bc)
            return fx          
        else:
            for block in self.blocks:
                fx = block(fx)
            return fx


class BranchNet(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(BranchNet, self).__init__()
        self.__name__ = 'Transolver_1D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num)
                                     for _ in range(n_layers)] + [Transolver_block_extraction(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num)
                                     ])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        # x: B N 2
        # grid_ref
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda().reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :] 

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        # 把fx从32，128变成32
        out = torch.sum(fx, dim=-1)
        return out


class TrunkNet(nn.Module):
    def __init__(
        self,
        space_dim=1,
        hidden_dim=256,
        out_dim=256,
        n_layers=4,
        act='gelu',
        Time_Input=False
    ):
        super().__init__()
        self.Time_Input = Time_Input

        in_dim = space_dim
        if Time_Input:
            in_dim += 1  # 把 t 当成一个坐标

        self.net = MLP(
            n_input=in_dim,
            n_hidden=hidden_dim,
            n_output=out_dim,
            n_layers=n_layers,
            res=True,
            act=act
        )
        self.out_dim = out_dim
    def forward(self, x, T=None):
        """
        x: [B, N, space_dim]
        T: [B, 1] or [B]
        """
        if self.Time_Input and T is not None:
            if T.dim() == 1:
                T = T[:, None]
            T = T[:, None, :].repeat(1, x.shape[1], 1)  # [B, N, 1]
            x = torch.cat([x, T], dim=-1)

        return self.net(x)   # [B, N, p]



class DeepONet(nn.Module):
    def __init__(
        self,
        branch_net: nn.Module,
        trunk_net: nn.Module,
        out_dim=1,
        bias=True
    ):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.out_dim = out_dim

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
        self.out_linear = nn.Linear(trunk_net.out_dim, out_dim, bias=bias)
    def forward(self, inputs, trunk_x, T=None):
        branch_x, branch_fx = inputs
        """
        x  : [B, N, space_dim]
        fx : [B, N, fun_dim]  or None
        T  : [B] or [B, 1]
        """
        # -------- Trunk --------
        trunk_out = self.trunk_net(trunk_x, T)          # [B, N, p]

        # -------- Branch --------
        # 物理状态 / 几何 / 材料特征s
        branch_out = self.branch_net(branch_x, branch_fx, T)   # [B, N, p]

        branch_out = branch_out[:, None, :]   # [B, 1, p]



        # -------- DeepONet inner product --------
        u = self.out_linear(branch_out * trunk_out)

        if self.bias is not None:
            u = u + self.bias

        return u