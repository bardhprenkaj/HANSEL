import torch

class SupervisedContrastiveLoss(torch.nn.Module):
  
  def __init__(self, margin=2.0):
    super().__init__()
    self.margin = margin  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = torch.nn.functional.pairwise_distance(y1, y2)
    loss_contrastive = torch.mean((1 - d) * torch.pow(euc_dist, 2) +
                                          d * torch.pow(torch.clamp(self.margin - euc_dist, min=0.0, max=None), 2))
    return loss_contrastive