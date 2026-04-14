
import random
class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        Recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def random_roll_dims(self,tensor, shift_range_n=(-5, 5), shift_range_c=(-7, 7)):
        '''
        对维度为 (B, num_heads, head_dim, N) 的 Tensor 的head_dim和N两个维度进行随机旋转。
        旋转大小从指定的范围内随机挑选。

        Args:
            tensor (torch.Tensor): 输入 Tensor，维度为 (B, num_heads,
            进一步的消融研究表明，通道滚动和序列滚动都有助于增益。
            shift_range_c/n (tuple): 包含最小和最大偏移量的范围 (min_shift, max_shift)。

        Returns:
            torch.Tensor: 旋转后的 Tensor。
        '''
        B, num_heads, head_dim, N = tensor.shape

        # 定义可选的旋转偏移量列表

        available_shifts = list(range(shift_range_n[0], shift_range_n[1] + 1))
        #available_shifts2 = list(range(-9, 10))
        #print('shift1',available_shifts)
        available_shifts2 = list(range(shift_range_c[0], shift_range_c[1] + 1))
        #print('shift2',available_shifts2)
        # 因为不能直接对整个批次应用不同的roll，我们需要一个列表来存储结果
        rolled_tensors = []

        # 循环处理每一个批次
        for i in range(B):
            # 随机选择一个偏移量
            # 注意：这里需要为每个批次（甚至在实际应用中可能为每个head）生成一个随机值
            # 这里的实现是为B中的每一个i单独roll，但随机值只生成一次，如果需要每次迭代都随机，需要放到循环内
            current_shift = random.choice(available_shifts)
            current_shift2 = random.choice(available_shifts2)

            # 对当前批次的 tensor 应用 torch.roll
            # torch.roll 的 shifts 和 dims 参数需要对应
            # shifts=[current_shift] 表示在 dims=[-1] 上旋转
            rolled_batch = torch.roll(tensor[i], shifts=[current_shift,current_shift2], dims=[-1,-2])
            rolled_tensors.append(rolled_batch)

        # 将列表中的 tensor 重新堆叠回原始形状
        result_tensor = torch.stack(rolled_tensors, dim=0)

        return result_tensor


    def forward(self, x):
        """ Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            if self.training:
                q = self.random_roll_dims(q)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            if self.training:
                k = self.random_roll_dims(k)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            if self.training:
                v = self.random_roll_dims(v)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    
