import torch
from torch_geometric.data import Data, Dataset

from src.dataset.data_instance_base import DataInstance


class TorchGeometricDataset(Dataset):
  
  def __init__(self, instances):
    super(TorchGeometricDataset, self).__init__()
    self.instances = instances
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]
  
  

class GraphPairDataset(Dataset):
  
  def __init__(self, anchors, targets, oracle=None, root=None, transform=None, pre_transform=None):
    super(GraphPairDataset, self).__init__(root, transform, pre_transform)
    self.anchors = anchors
    self.targets = targets
    
    self.oracle = oracle
    self.instances = []
    
    self._process()

  def _process(self):
    data_list = []
    for i in range(len(self.anchors)):
      data1 = self.anchors[i]
      for j in range(1, len(self.targets)):
        data2 = self.targets[j]
        
        if data1.id != data2.id:
        
          graph1 = self.to_geometric(data1)
          graph2 = self.to_geometric(data2)
          # if we're using the oracle, it means we're training
          # otherwise, we should consider all the targets as "different"
          if self.oracle:
            label = 1 if self.oracle.predict(data1) != self.oracle.predict(data2) else 0
          else:
            label = 1
          # Append the graph pair and label to the data_list
          data_list.append(PairData(x_s=graph1.x, edge_index_s=graph1.edge_index, edge_attr_s=graph1.edge_attr,
                                    x_t=graph2.x, edge_index_t=graph2.edge_index, edge_attr_t=graph2.edge_attr,
                                    label=label, 
                                    index_s=data1.id,
                                    index_t=data2.id,
                                    num_nodes=data1.graph.number_of_nodes() + data2.graph.number_of_nodes()))
        
    self.instances = data_list
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]
  
  def to_geometric(self, instance: DataInstance, label=0) -> Data:   
    adj = torch.from_numpy(instance.to_numpy_array()).double()
    x = torch.from_numpy(instance.features).double()

    a = torch.nonzero(torch.triu(adj))
    w = adj[a[:,0], a[:,1]]
    
    return Data(x=x, y=label, edge_index=a.T, edge_attr=w)
  
  
class PairData(Data):
  pass