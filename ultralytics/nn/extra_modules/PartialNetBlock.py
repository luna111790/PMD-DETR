import torch, math, timm
import torch.nn as nn
from torch import Tensor
from easydict import EasyDict as edict

RPEIndexFunction = None

@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    














    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        
        idx = idx.round().to(dtype)

    
    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    
























    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


@torch.no_grad()
def quantize_values(values):
    














    
    res = torch.empty_like(values)
    uq = values.unique()
    cnt = 0
    for (tid, v) in enumerate(uq):
        mask = (values == v)
        cnt += torch.count_nonzero(mask)
        res[mask] = tid
    assert cnt == values.numel()
    return res, uq.numel()


class METHOD:
    



    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42

'''
@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    """2D RPE with Euclidean method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff.square().sum(2).float().sqrt().round()
    size = torch.max(diff)
    dis = dis / size
    return piecewise_index(dis, **kwargs)
'''

@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    















    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)



@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    
















    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    















    
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    
    r = piecewise_index(diff[:, :, 0], **kwargs) + \
        beta_int  
    c = piecewise_index(diff[:, :, 1], **kwargs) + \
        beta_int  
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    















    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    
















    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)



_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


def get_num_buckets(method, alpha, beta, gamma):
    














    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        
        num_buckets = 2 * beta_int + 1
    return num_buckets



BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d_without_skip(method, height, width,
                                   alpha, beta, gamma,
                                   dtype=torch.long, device=torch.device('cpu')):
    



























    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)
        
        func = _METHOD_FUNC.get(method, None)
        if func is None:
            raise NotImplementedError(
                f"[Error] The method ID {method} does not exist.")
        pos = get_absolute_positions(max_height, max_width, dtype, device)

        
        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        
        diff = pos1 - pos2

        
        bucket_ids = func(diff, alpha=alpha, beta=beta,
                          gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(
            max_height, max_width, max_height, max_width)

        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]

    return bucket_ids, num_buckets, L


@torch.no_grad()
def get_bucket_ids_2d(method, height, width,
                      skip, alpha, beta, gamma,
                      dtype=torch.long, device=torch.device('cpu')):
    





























    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(method, height, width,
                                                                alpha, beta, gamma,
                                                                dtype, device)

    
    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))

        
        extra_bucket_id = num_buckets
        num_buckets += 1

        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids

        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets


class iRPE(nn.Module):
    

























    
    
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8,
                 mode=None, method=None,
                 transposed=True, num_buckets=None,
                 initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        
        assert mode in [None, 'bias', 'contextual']
        self.mode = mode

        assert method is not None, 'method should be a METHOD ID rather than None'
        self.method = method

        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

        self.rpe_config = rpe_config

    @torch.no_grad()
    def reset_parameters(self):
        
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'bias':
                raise NotImplementedError(
                    "[Error] Bias non-transposed RPE does not exist.")
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None):
        

















        rp_bucket, self._ctx_rp_bucket_flatten = \
            self._get_rp_bucket(x, height=height, width=width)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None):
        

























        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L))
            height = width = E
        key = (height, width, device)
        

        if self._rp_bucket_buf[0] == key:
            return self._rp_bucket_buf[1:3]

        skip = L - height * width
        config = self.rpe_config
        if RPEIndexFunction is not None and self.mode == 'contextual' and self.transposed:
            
            dtype = torch.int32
        else:
            dtype = torch.long
        rp_bucket, num_buckets = get_bucket_ids_2d(method=self.method,
                                                   height=height, width=width,
                                                   skip=skip, alpha=config.alpha,
                                                   beta=config.beta, gamma=config.gamma,
                                                   dtype=dtype, device=device)
        
        assert num_buckets == self.num_buckets

        
        _ctx_rp_bucket_flatten = None
        if self.mode == 'contextual' and self.transposed:
            if RPEIndexFunction is None:
                offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                      dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
                _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        

































        B = len(x)  
        L_query, L_key = rp_bucket.shape
        if self.mode == 'bias':
            return self.lookup_table_bias[:, rp_bucket.flatten()].\
                view(1, self.num_heads, L_query, L_key)

        elif self.mode == 'contextual':
            """
            ret[b, h, i, j] = lookup_table_weight[b, h, i, rp_bucket[i, j]]

            ret[b, h, i * L_key + j] = \
               lookup_table[b, h, i * num_buckets + rp_buckets[i, j]]

            computational cost
            ------------------
            matmul: B * H * L_query * head_dim * num_buckets
            index: L_query + L_query * L_key + B * H * L_query * L_key
            total: O(B * H * L_query * (head_dim * num_buckets + L_key))
            """
            lookup_table = torch.matmul(
                x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim),
                self.lookup_table_weight).\
                view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            if RPEIndexFunction is not None:
                return RPEIndexFunction.apply(lookup_table, rp_bucket)
            else:
                return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].\
                    view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        




























        B = len(x)  
        L_query, L_key = rp_bucket.shape
        assert self.mode == 'contextual', "Only support contextual \
version in non-transposed version"
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].\
            view(self.num_heads, L_query, L_key, self.head_dim)
        
        
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)

    def __repr__(self):
        return 'iRPE(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, \
mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, \
num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self)


class iRPE_Cross(nn.Module):
    


























    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        


























        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols

    def __repr__(self):
        return 'iRPE_Cross(head_dim={rpe.head_dim}, \
num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, \
transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, \
initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self.rp_rows)


def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    
























    config = edict()
    
    config.shared_head = shared_head
    
    config.mode = mode
    
    config.method = method
    
    config.alpha = 1 * ratio
    
    config.beta = 2 * ratio
    
    config.gamma = 8 * ratio

    
    config.num_buckets = get_num_buckets(method,
                                         config.alpha,
                                         config.beta,
                                         config.gamma)
    
    if skip > 0:
        config.num_buckets += 1
    return config


def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   skip=0,
                   rpe_on='k'):
    


































    
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = edict()
    
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
        skip=skip,
    )
    config.rpe_q = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    



















    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        rpe_cls = iRPE if rpe.method != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=transposed,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]

class RPEAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h*w).transpose(1,2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        
        if self.rpe_k is not None:
            
            attn += self.rpe_k(q, h, w)
        
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1,2).view(B, C, h, w)
        return x

class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1,2), bias=False)
        
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        
        mean = x.reshape(b, c, -1).mean(-1).view(b,c,1,1)
        std = x.reshape(b, c, -1).std(-1).view(b,c,1,1)
        
        u = torch.cat([mean, std], dim=-1)
        
        z = self.cfc1(u)
        
        
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

class PartialNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, version='v2', n_div=4):
        super().__init__()
        self.dim_conv3 = in_dim // n_div
        self.dim = in_dim
        self.n_div = n_div
        self.dim_untouched = in_dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.version = version

        if version == 'v2':
            self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
            rpe_config = get_rpe_config(
                    ratio=20,
                    method="euc",
                    mode='bias',
                    shared_head=False,
                    skip=0,
                    rpe_on='k',
                )
            num_heads = 4
            self.attn = RPEAttention(self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config)
            self.norm = timm.layers.LayerNorm2d(self.dim_untouched)
            self.forward = self.forward_atten
        else:
            self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
            self.attn = SRM(self.dim_untouched)
            self.norm = nn.BatchNorm2d(self.dim_untouched)
            self.forward = self.forward_atten
        
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
    
    def forward_atten(self, x: Tensor) -> Tensor:
        if self.version == 'v2':
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x2 = self.norm(x2)
            x2 = self.attn(x2)
            x = torch.cat((x1, x2), 1)
        else:
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x2 = self.attn(x2)
            x2 = self.norm(x2)
        x = self.conv(x)
        return x