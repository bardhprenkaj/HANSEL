from typing import Tuple, List
import torch
import numpy as np

from src.dataset.data_instance_base import DataInstance


class Dataset(torch.utils.data.Dataset):
  
  def __init__(self, instances: List[DataInstance], transform=True):
    super(Dataset, self).__init__()    
    self.instances = []

    if transform:
      self._process(instances)
    else:
      self.instances = instances
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]
  
  def _process(self, instances: List[DataInstance]):
    arrays = [inst.to_numpy_array() for inst in instances]
    max_rows = max(arr.shape[0] for arr in arrays)
    max_rows = ((max_rows - 1) // 4 + 1) * 4
    # Pad each array to the maximum dimensions
    padded_arrays = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_rows - arr.shape[1])), mode='constant') for arr in arrays]
    for i, inst in enumerate(instances):
        #padded_arrays[i] = (padded_arrays[i] - np.min(padded_arrays[i])) / (np.max(padded_arrays[i]) - np.min(padded_arrays[i]))
        inst.from_numpy_array(padded_arrays[i], store=True)
        instances[i] = inst
    self.instances = [self.transform(inst) for inst in instances]
      
  def transform(self, instance: DataInstance) -> Tuple[torch.Tensor, torch.Tensor]:   
    adj = torch.from_numpy(instance.to_numpy_array()).double()
    label = torch.from_numpy(np.array([instance.graph_label]))
    return adj, label

  @property
  def num_nodes(self):
    return self.x.shape[0]