import os
import shutil
import numpy as np

from src.dataset.dataset_adhd import ADHDDataset
from src.dataset.dataset_asd import ASDDataset
from src.dataset.dataset_base import Dataset
from src.dataset.dataset_bbbp import BBBPDataset
from src.dataset.dataset_hiv import HIVDataset
from src.dataset.dataset_imbd import IMDBDataset
from src.dataset.dataset_synthetic_generator import Synthetic_Data
from src.dataset.dataset_trisqr import TrianglesSquaresDataset
from src.dataset.dynamic_graphs.dataset_btc_alpha import BTCAlpha
from src.dataset.dynamic_graphs.dataset_coauthorship_dblp import \
    CoAuthorshipDBLP
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
from src.dataset.dynamic_graphs.dynamic_tree_cycles import DynTreeCycles
from src.dataset.dynamic_graphs.dataset_bonanaza import Bonanza


class DatasetFactory():

    def __init__(self, data_store_path) -> None:
        self._data_store_path = data_store_path
        self._dataset_id_counter = 0

    def get_dataset_by_name(self, dataset_dict) -> Dataset:

        dataset_name = dataset_dict['name']
        params_dict = dataset_dict['parameters']

        # Check if the graph is a tree-cycles graph
        if dataset_name == 'tree-cycles':
            if not 'n_inst' in params_dict:
                raise ValueError('''"n_inst" parameter containing the number of instances in the dataset
                 is mandatory for tree-cycles graph''')

            if not 'n_per_inst' in params_dict:
                raise ValueError('''"n_per_inst" parameter containing the number of nodes per instance
                 is mandatory for tree-cycles graph''')

            if not 'n_in_cycles' in params_dict:
                raise ValueError('''"n_in_cycles" parameter containing the number of nodes in cycles per instance
                 is mandatory for tree-cycles graph''')

            return self.get_tree_cycles_dataset(params_dict['n_inst'], params_dict['n_per_inst'], params_dict['n_in_cycles'], False, dataset_dict)

        # Check if the graph is a tree-cycles-balanced graph
        elif dataset_name == 'tree-cycles-balanced':
            if not 'n_inst_class' in params_dict:
                raise ValueError('''"n_inst_class" parameter containing the number of instances per class in the dataset
                 is mandatory for tree-cycles-balanced graph''')

            if not 'n_per_inst' in params_dict:
                raise ValueError('''"n_per_inst" parameter containing the number of nodes per instance
                 is mandatory for tree-cycles-balanced graph''')

            if not 'n_in_cycles' in params_dict:
                raise ValueError('''"n_in_cycles" parameter containing the number of nodes in cycles per instance
                 is mandatory for tree-cycles-balanced graph''')

            return self.get_tree_cycles_balanced_dataset(params_dict['n_inst_class'], params_dict['n_per_inst'], params_dict['n_in_cycles'], False, dataset_dict)

        # Check if the graph is a tree-cycles-dummy graph
        elif dataset_name == 'tree-cycles-dummy':
            if not 'n_inst_class' in params_dict:
                raise ValueError('''"n_inst_class" parameter containing the number of instances per class in the dataset
                 is mandatory for tree-cycles-dummy graph''')

            if not 'n_per_inst' in params_dict:
                raise ValueError('''"n_per_inst" parameter containing the number of nodes per instance
                 is mandatory for tree-cycles-dummy graph''')

            if not 'n_in_cycles' in params_dict:
                raise ValueError('''"n_in_cycles" parameter containing the number of nodes in cycles per instance
                 is mandatory for tree-cycles-dummy graph''')

            return self.get_tree_cycles_dummy_dataset(params_dict['n_inst_class'], params_dict['n_per_inst'],
                                                         params_dict['n_in_cycles'], False, dataset_dict)

        elif dataset_name == 'tree-infinity':
            if not 'n_inst' in params_dict:
                raise ValueError('''"n_inst" parameter containing the number of instances in the dataset
                 is mandatory for tree-infinity graph''')

            if not 'n_per_inst' in params_dict:
                raise ValueError('''"n_per_inst" parameter containing the number of nodes per instance in the dataset
                 is mandatory for tree-infinity graph''')

            if not 'n_infinities' in params_dict:
                raise ValueError('''"n_infinities" parameter containing the number of infinity-shaped subgraphs per instance
                 is mandatory for tree-infinity graph''')

            if not 'n_broken_infinities' in params_dict:
                raise ValueError('''"n_broken_infinities" parameter containing the number of broken infinity-shaped subgraphs per instance
                 is mandatory for tree-infinity graph''')

            return self.get_tree_infinity_dataset(params_dict['n_inst'], params_dict['n_per_inst'], params_dict['n_infinities'],
                                                         params_dict['n_broken_infinities'], False, dataset_dict)

        # Check if the dataset is the ASD-Dataset
        elif dataset_name == 'autism':
            return self.get_asd_dataset(False, dataset_dict)

        # Check if the dataset is the ADHD-Dataset
        elif dataset_name == 'adhd':
            return self.get_adhd_dataset(False, dataset_dict)

        # Check if the dataset is the human blood-brain barrier (BBB) penetration dataset
        elif dataset_name == 'bbbp':
            force_fixed_nodes = params_dict.get('force_fixed_nodes', False)

            return self.get_bbbp_dataset(False, force_fixed_nodes, dataset_dict)

        # Check if the dataset is the human blood-brain barrier (BBB) penetration dataset
        elif dataset_name == 'hiv':
            force_fixed_nodes = params_dict.get('force_fixed_nodes', False)
            return self.get_hiv_dataset(False, force_fixed_nodes, dataset_dict)
        
        # Check if the dataset is a triangles-squares dataset
        elif dataset_name == 'trisqr':
            if not 'n_inst' in params_dict:
                raise ValueError('''"n_inst" parameter containing the number of instances in the dataset
                 is mandatory for triangles-squares dataset''')

            return self.get_trisqr_dataset(params_dict['n_inst'], False, dataset_dict)
        
        elif dataset_name == 'syn1':
            return self.get_syn_dataset(False, "1")
        
        elif dataset_name == 'syn4':
            return self.get_syn_dataset(False, "4")
        
        elif dataset_name == 'syn5':
            return self.get_syn_dataset(False, "5")

        elif dataset_name == 'imdb':
            self_loops = params_dict.get('self_loops', False)
            return self.get_imdb(self_loops, dataset_dict)
        
        elif dataset_name == 'coauthorship_dblp':
            if not 'begin_time' in params_dict:
                raise ValueError('"begin_time" is mandatory for CoAuthorshipDBLP')
            if not 'end_time' in params_dict:
                raise ValueError('"end_time" is mandatory for CoAuthorshipDBLP')
            
            begin_time = params_dict['begin_time']
            end_time = params_dict['end_time']
            
            assert (begin_time <= end_time)
            
            percentile = params_dict.get('percentile', 75)
            sampling_ratio = params_dict.get('sampling_ratio', .05)
            min_nodes_per_egonet = params_dict.get('min_nodes_per_egonet', 3)
            features_dim = params_dict.get('features_dim', 8)
            padd = params_dict.get('padd', False)
        
            return self.get_coauthorship_dblp(begin_time,
                                              end_time,
                                              percentile,
                                              min_nodes_per_egonet,
                                              sampling_ratio,
                                              features_dim,
                                              dataset_dict,
                                              padd=padd)
            
        elif dataset_name == 'dynamic_tree_cycles':
            if not 'begin_time' in params_dict:
                raise ValueError('"begin_time" is mandatory for CoAuthorshipDBLP')
            if not 'end_time' in params_dict:
                raise ValueError('"end_time" is mandatory for CoAuthorshipDBLP')
            
            begin_time = params_dict['begin_time']
            end_time = params_dict['end_time']
            
            assert (begin_time <= end_time)
            
            num_instances_per_snapshot = params_dict.get('num_instances_per_snapshot', 300)
            n_nodes = params_dict.get('n_nodes', 300)
            nodes_in_cycle = params_dict.get('nodes_in_cycle', 200)
            
            return self.get_dynamic_tree_cycles(begin_time=begin_time,
                                                end_time=end_time,
                                                num_instances_per_snapshot=num_instances_per_snapshot,
                                                n_nodes=n_nodes,
                                                nodes_in_cycle=nodes_in_cycle,
                                                dataset_dict=dataset_dict)
            
        elif dataset_name in ['btc_alpha', 'btc_otc']:
            if not 'begin_time' in params_dict:
                raise ValueError('"begin_time" is mandatory for CoAuthorshipDBLP')
            if not 'end_time' in params_dict:
                raise ValueError('"end_time" is mandatory for CoAuthorshipDBLP')
            
            begin_time = params_dict['begin_time']
            end_time = params_dict['end_time']
            padd = params_dict.get('padd', False)

            
            assert (begin_time <= end_time)
            
            filter_min_graphs = params_dict.get('filter_min_graphs', 10)
            
            return self.get_btc(begin_time=begin_time,
                                      end_time=end_time,
                                      filter_min_graphs=filter_min_graphs,
                                      padd=padd, name=dataset_name)
        elif dataset_name == 'bonanza':
            if not 'begin_time' in params_dict:
                raise ValueError('"begin_time" is mandatory for CoAuthorshipDBLP')
            if not 'end_time' in params_dict:
                raise ValueError('"end_time" is mandatory for CoAuthorshipDBLP')
            
            begin_time = params_dict['begin_time']
            end_time = params_dict['end_time']
            pad = params_dict.get('pad', False)

            
            assert (begin_time <= end_time)
            
            filter_min_graphs = params_dict.get('filter_min_graphs', 10)
            
            return self.get_bonanza(begin_time=begin_time,
                                    end_time=end_time,
                                    filter_min_graphs=filter_min_graphs,
                                    padding=pad)
        # If the dataset name does not match any of the datasets provided by the factory
        else:
            raise ValueError('''The provided dataset name is not valid. Valid names include: tree-cycles,
             tree-cycles-balanced, tree-cycles-dummy''')
            
    def get_bonanza(self, begin_time, end_time, filter_min_graphs, dataset_dict=None, padding=False):
        result = Bonanza(id=self._dataset_id_counter,
                          begin_t=begin_time,
                          end_t=end_time,
                          filter_min_graphs=filter_min_graphs,
                          padding=padding,
                          config_dict=dataset_dict)
        
        self._dataset_id_counter += 1
        ds_uri = os.path.join(self._data_store_path, 'dynamic_graphs', 'bonanza', 'processed')
        ds_exists = os.path.exists(ds_uri)
        
        if ds_exists:
            result.read_datasets(ds_uri)
        else:
            result.read_csv_file(os.path.join(self._data_store_path, 'dynamic_graphs', 'bonanza'))
            result.build_temporal_graph()
            result.load_or_generate_splits(os.path.join(self._data_store_path, 'dynamic_graphs', 'bonanza'),
                                           n_splits=10, shuffle=True)
            result.write_datasets(ds_uri)
        if padding:
            max_nodes = max([instance.graph.number_of_nodes() for graph in result.dynamic_graph.values() for instance in graph.instances])
            result = self.pad_matrices(result, max_nodes)
            
        return result
    
    def get_btc(self, begin_time, end_time, filter_min_graphs, dataset_dict=None, padd=False, name='btc_alpha'):
        
        result = BTCAlpha(id=self._dataset_id_counter,
                          begin_t=begin_time,
                          end_t=end_time,
                          filter_min_graphs=filter_min_graphs,
                          config_dict=dataset_dict)
        self._dataset_id_counter += 1
        ds_uri = os.path.join(self._data_store_path, 'dynamic_graphs', name, 'processed')
        ds_exists = os.path.exists(ds_uri)
        
        if ds_exists:
            result.read_datasets(ds_uri)
        else:
            result.read_csv_file(os.path.join(self._data_store_path, 'dynamic_graphs', name))
            result.build_temporal_graph()
            result.load_or_generate_splits(os.path.join(self._data_store_path, 'dynamic_graphs', name),
                                           n_splits=10, shuffle=True)
            result.write_datasets(ds_uri)
            
        if padd:
            max_nodes = max([instance.graph.number_of_nodes() for graph in result.dynamic_graph.values() for instance in graph.instances ])
            result = self.pad_matrices(result, max_nodes)
            
        return result
    
    def get_dynamic_tree_cycles(self, begin_time, end_time,
                                num_instances_per_snapshot=300,
                                n_nodes=300,
                                nodes_in_cycle=200,
                                dataset_dict=None):
        
        result = DynTreeCycles(id=self._dataset_id_counter,
                               begin_t=begin_time,
                               end_t=end_time,
                               num_instances_per_snapshot=num_instances_per_snapshot,
                               n_nodes=n_nodes,
                               nodes_in_cycle=nodes_in_cycle,
                               config_dict=dataset_dict)
        
        self._dataset_id_counter += 1
        ds_name = 'dynamic_tree_cycles'
        ds_uri = os.path.join(self._data_store_path, 'dynamic_graphs', ds_name, 'processed')
        ds_exists = os.path.exists(ds_uri)
        
        if ds_exists:
            result.read_datasets(ds_uri)
        else:
            result.build_temporal_graph()
            result.load_or_generate_splits(os.path.join(self._data_store_path, 'dynamic_graphs', ds_name),
                                           n_splits=10, shuffle=True)
            result.write_datasets(ds_uri)
            
        return result
            
    def get_coauthorship_dblp(self, begin_time, end_time,
                              percentile=75,
                              min_nodes_per_egonet=3,
                              sampling_ratio=.05, 
                              features_dim=8,
                              dataset_dict=None,
                              padd=False):
        
        result = CoAuthorshipDBLP(self._dataset_id_counter,
                                  begin_time=begin_time,
                                  end_time=end_time,
                                  min_nodes_per_egonet=min_nodes_per_egonet,
                                  percentile=percentile,
                                  sampling_ratio=sampling_ratio,
                                  features_dim=features_dim,
                                  config_dict=dataset_dict)
        
        self._dataset_id_counter += 1
        ds_name = 'coauthorship_dblp'
        ds_uri = os.path.join(self._data_store_path, 'dynamic_graphs', ds_name, 'processed')
        ds_exists = os.path.exists(ds_uri)
        
        if ds_exists:
            result.read_datasets(ds_uri)
        else:
            result.read_csv_file(os.path.join(self._data_store_path, 'dynamic_graphs', ds_name))
            result.build_temporal_graph()
            result.load_or_generate_splits(os.path.join(self._data_store_path, 'dynamic_graphs', ds_name),
                                           n_splits=10, shuffle=True)
            result.write_datasets(ds_uri)
            
        if padd:
            max_nodes = max([instance.graph.number_of_nodes() for graph in result.dynamic_graph.values() for instance in graph.instances ])
            result = self.pad_matrices(result, max_nodes)

        return result
    
    
    def pad_matrices(self, dataset: DynamicDataset, dim: int):
        import copy
        new_dataset = copy.deepcopy(dataset)
        for key, graph in dataset.dynamic_graph.items():
            instances = []
            for instance in graph.instances:
                padded_matrix = np.zeros((dim, dim))
                adj_matrix = instance.to_numpy_array(store=False)
                padded_matrix[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
                instance.from_numpy_array(padded_matrix, store=True)
                instances.append(instance)
            graph.instances = instances
            new_dataset.dynamic_graph[key] = graph
        return new_dataset


    def get_imbd(self, self_loops=False, config_dict=None):
        result = IMDBDataset(self._dataset_id_counter, self_loops=self_loops, config_dict=config_dict)
        self._dataset_id_counter += 1
        
        ds_name = 'imdb-instances'
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)
        
        if ds_exists:
            result.read_data(ds_uri)
        else:
            result.read_adjacency_matrices(ds_uri)
            result.load_or_generate_splits(ds_uri)
            
        return result

    def get_tree_cycles_dataset(self, n_instances=300, n_total=300, n_in_cycles=200, regenerate=False, config_dict=None) -> Dataset:
        result = Synthetic_Data(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = ('tree-cycles_instances-'+ str(n_instances) + '_nodes_per_inst-' + str(n_total) +
                        '_nodes_in_cycles-' + str(n_in_cycles))
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_exists: 
            shutil.rmtree(ds_uri)

        # Check if the dataset already exists
        if(ds_exists):
            # load the dataset
            result.read_data(ds_uri)
        else:
            # Generate the dataset
            result.generate_tree_cycles_dataset(n_instances, n_total, n_in_cycles)
            result.generate_splits()
            result.write_data(self._data_store_path)
            
        return result


    def get_tree_cycles_balanced_dataset(self, n_instances_per_class=150, n_total=300, n_in_cycles=200, 
                                            regenerate=False, config_dict=None) -> Dataset:
        result = Synthetic_Data(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = ('tree-cycles-balanced_instances_per_class-'+ str(n_instances_per_class) + 
                        '_nodes_per_inst-' + str(n_total) + '_nodes_in_cycles-' + str(n_in_cycles))
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_exists: 
            shutil.rmtree(ds_uri)

        # Check if the dataset already exists
        if(ds_exists):
            # load the dataset
            result.read_data(ds_uri)
        else:
            # Generate the dataset
            result.generate_tree_cycles_dataset_balanced(n_instances_per_class=n_instances_per_class, 
                                                n_total=n_total, n_in_cycles=n_in_cycles)
            result.generate_splits()
            result.write_data(self._data_store_path)
            
        return result


    def get_tree_cycles_dummy_dataset(self, n_instances_per_class=150, n_total=300, n_in_cycles=200, 
                                        regenerate=False, config_dict=None) -> Dataset:
        result = Synthetic_Data(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = ('tree-cycles-dummy_instances_per_class-'+ str(n_instances_per_class) + 
                        '_nodes_per_inst-' + str(n_total) + '_nodes_in_cycles-' + str(n_in_cycles))
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_exists: 
            shutil.rmtree(ds_uri)

        # Check if the dataset already exists
        if(ds_exists):
            # load the dataset
            result.read_data(ds_uri)
        else:
            # Generate the dataset
            result.generate_dataset_dummy(n_instances_per_class=n_instances_per_class, 
                                                n_total=n_total, n_in_cycles=n_in_cycles)
            result.generate_splits()
            result.write_data(self._data_store_path)
            
        return result


    def get_asd_dataset(self, regenerate=False, config_dict=None) -> Dataset:
        result = ASDDataset(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = 'autism'
        ds_uri = os.path.join(self._data_store_path, ds_name)

        ds_formatted_uri = os.path.join(ds_uri, 'autism_dataset')
        ds_formatted_exists = os.path.exists(ds_formatted_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_formatted_exists: 
            shutil.rmtree(ds_formatted_uri)

        # TODO: Reactivate the saving and loading from our standard dataset format. Temporarly deactivated due to a bug
        # # Check if the dataset already exists
        # if(ds_formatted_exists):
        #     # load the dataset from our formatted data
        #     result.read_data(ds_formatted_uri, graph_format='adj_matrix')
        # else:
        #     # load the dataset from original
        #     result.read_adjacency_matrices(ds_uri)
        #     result.generate_splits()
        #     result.write_data(ds_uri, graph_format='adj_matrix')

        result.read_adjacency_matrices(ds_uri)
        result.load_or_generate_splits(ds_uri)
            
        return result

    
    def get_adhd_dataset(self, regenerate=False, config_dict=None) -> Dataset:
        result = ADHDDataset(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = 'adhd'
        ds_uri = os.path.join(self._data_store_path, ds_name, 'graphs')

        ds_formatted_uri = os.path.join(ds_uri, 'adhd_dataset')
        ds_formatted_exists = os.path.exists(ds_formatted_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_formatted_exists: 
            shutil.rmtree(ds_formatted_uri)

        # TODO: Reactivate the saving and loading from our standard dataset format. Temporarly deactivated due to a bug
        # Check if the dataset already exists
        # if(ds_formatted_exists):
        #     # load the dataset from our formatted data
        #     result.read_data(ds_formatted_uri, graph_format='adj_matrix')
        # else:
        #     # load the dataset from original
        #     result.read_adjacency_matrices(ds_uri)
        #     result.generate_splits()
        #     result.write_data(ds_uri, graph_format='adj_matrix')

        result.read_adjacency_matrices(ds_uri)
        result.load_or_generate_splits(ds_uri)
            
        return result


    def get_tree_infinity_dataset(self, n_instances=300, n_per_inst=300, n_infinity=10, n_brkn_infinity=10, regenerate=False, config_dict=None) -> Dataset:
        result = Synthetic_Data(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = ('tree-infinity_instances-'+ str(n_instances) + '_nodes_per_inst-' + str(n_per_inst) +
                        '_n_infinities-' + str(n_infinity) + '_n_broken_infinities-' 
                        + str(n_brkn_infinity))
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_exists: 
            shutil.rmtree(ds_uri)

        # Check if the dataset already exists
        if(ds_exists):
            # load the dataset
            result.read_data(ds_uri, graph_format='adj_matrix')
        else:
            # Generate the dataset
            result.generate_tree_infinity_dataset(n_instances, n_per_inst, n_infinity, n_brkn_infinity)
            result.generate_splits()
            result.write_data(self._data_store_path, graph_format='adj_matrix')
            
        return result


    def get_bbbp_dataset(self, regenerate=False, force_fixed_nodes=False, config_dict=None) -> Dataset:
        result = BBBPDataset(self._dataset_id_counter, config_dict, force_fixed_nodes)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = 'bbbp'
        ds_uri = os.path.join(self._data_store_path, ds_name)

        ds_formatted_uri = None

        if force_fixed_nodes:
            result.name = 'bbbp_fixed_nodes'
            ds_formatted_uri = os.path.join(ds_uri, 'bbbp_fixed_nodes')
        else:
            result.name = 'bbbp'
            ds_formatted_uri = os.path.join(ds_uri, 'bbbp')

        # ds_formatted_exists = os.path.exists(ds_formatted_uri)
        ds_formatted_exists = False

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_formatted_exists: 
            shutil.rmtree(ds_formatted_uri)

        # Check if the dataset already exists
        if(ds_formatted_exists):
            # load the dataset from our formatted data
            result.read_data(ds_formatted_uri, graph_format='edge_list')
            return result
        else:
            # load the dataset from original
            # result.read_molecules_file(ds_uri)
            result.read_csv_file(ds_uri)
            result.load_or_generate_splits(ds_uri)
            return result


    def get_hiv_dataset(self, regenerate=False, force_fixed_nodes=False, config_dict=None) -> Dataset:
        result = HIVDataset(self._dataset_id_counter, config_dict, force_fixed_nodes)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = 'hiv'
        ds_uri = os.path.join(self._data_store_path, ds_name)

        ds_formatted_uri = None

        if force_fixed_nodes:
            result.name = 'hiv_fixed_nodes'
            ds_formatted_uri = os.path.join(ds_uri, result.name)
        else:
            result.name = 'hiv'
            ds_formatted_uri = os.path.join(ds_uri, result.name)

        # ds_formatted_exists = os.path.exists(ds_formatted_uri)
        ds_formatted_exists = False

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_formatted_exists: 
            shutil.rmtree(ds_formatted_uri)

        # Check if the dataset already exists
        if(ds_formatted_exists):
            # load the dataset from our formatted data
            result.read_data(ds_formatted_uri, graph_format='edge_list')
            return result
        else:
            # load the dataset from original
            result.read_csv_file(ds_uri)
            result.load_or_generate_splits(ds_uri)
            return result


    def get_trisqr_dataset(self, n_instances=100, regenerate=False, config_dict=None) -> Dataset:
        result = TrianglesSquaresDataset(self._dataset_id_counter, config_dict)
        self._dataset_id_counter+=1

        # Create the name an uri of the dataset using the provided parameters
        ds_name = ('triangles-squares_instances-'+ str(n_instances))
        ds_uri = os.path.join(self._data_store_path, ds_name)
        ds_exists = os.path.exists(ds_uri)

        # If regenerate is true and the dataset exists then remove it an generate it again
        if regenerate and ds_exists: 
            shutil.rmtree(ds_uri)

        # Check if the dataset already exists
        if(ds_exists):
            # load the dataset
            result.read_data(ds_uri)
        else:
            # Generate the dataset
            result.generate_dataset(n_instances)
            result.generate_splits()
            result.write_data(self._data_store_path)
            
        return result