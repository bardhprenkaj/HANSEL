from torch_geometric.data import Dataset

class TorchGeometricDataset(Dataset):
  
  def __init__(self, instances):
    super(Dataset, self).__init__()
    self.instances = instances
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]