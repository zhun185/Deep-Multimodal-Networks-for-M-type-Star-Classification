import  torch
from torch import nn

def loss_sic(align_logits, fuse_logits, targets):
    bs = targets.shape[0]
    loss_fn = nn.CrossEntropyLoss()

    # 分类损失总是计算
    loss_f = loss_fn(fuse_logits, targets)

    # 只有当批次大小大于 1 时才计算对比损失
    if bs > 1:
        labels = torch.arange(bs, device=align_logits.device)
        loss_i = loss_fn(align_logits, labels)
        loss_s = loss_fn(align_logits.T, labels)
        loss = (loss_i + loss_s) / 2 + loss_f
    else:
        # 如果批次大小为 1 或 0，只使用分类损失
        loss = loss_f

    return loss

