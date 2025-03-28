from abc import ABC
from src.oracle.oracle_custom_bonanza import BonanzaOracle
from src.dataset.converters.abstract_converter import ConverterAB
from src.dataset.converters.weights_converter import \
    DefaultFeatureAndWeightConverter
from src.dataset.dataset_base import Dataset
from src.oracle.dynamic_graphs.dynamic_oracle_base import DynamicOracle
from src.oracle.embedder_base import Embedder
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.embedder_graph2vec import Graph2vec
from src.oracle.oracle_asd_custom import ASDCustomOracle
from src.oracle.oracle_base import Oracle
from src.oracle.oracle_custom_btc_alpha import BTCAlphaCustomOracle
from src.oracle.oracle_custom_dblp import DBLPCoAuthorshipCustomOracle
from src.oracle.oracle_id import IDOracle
from src.oracle.oracle_knn import KnnOracle
from src.oracle.oracle_svm import SvmOracle
from src.oracle.oracle_tree_cycles_custom import TreeCyclesCustomOracle
from src.oracle.oracle_triangles_squares_custom import \
    TrianglesSquaresCustomOracle


class OracleFactory(ABC):

    def __init__(self, oracle_store_path) -> None:
        super().__init__()
        self._oracle_store_path = oracle_store_path
        self._oracle_id_counter = 0

    def get_oracle_by_name(self, oracle_dict, dataset: Dataset, emb_factory: EmbedderFactory) -> Oracle:

        oracle_name = oracle_dict['name']
        oracle_parameters = oracle_dict['parameters']

        # Check if the oracle is a KNN classifier
        if oracle_name == 'knn':
            if not 'k' in oracle_parameters:
                raise ValueError('''The parameter "k" is required for knn''')
            if not 'embedder' in oracle_parameters:
                raise ValueError('''knn oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_knn(dataset, emb, oracle_parameters['k'], -1, oracle_dict)

        # Check if the oracle is an SVM classifier
        elif oracle_name == 'svm':
            if not 'embedder' in oracle_parameters:
                raise ValueError('''svm oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_svm(dataset, emb, -1, oracle_dict)

        # Check if the oracle is an ASD Custom Classifier
        elif oracle_name == 'asd_custom_oracle':
            return self.get_asd_custom_oracle(oracle_dict)

        elif oracle_name == 'gcn_synthetic_pt':
            return self.get_pt_syn_oracle(dataset, -1, oracle_dict)
            
        # Check if the oracle is a Triangles-Squares Custom Classifier
        elif oracle_name == 'trisqr_custom_oracle':
            return self.get_trisqr_custom_oracle(oracle_dict)

        # Check if the oracle is a Tree-Cycles Custom Classifier
        elif oracle_name == 'tree_cycles_custom_oracle':
            return self.get_tree_cycles_custom_oracle(oracle_dict) 
        
        elif oracle_name == 'cf2':
            if not 'converter' in oracle_parameters:
                raise ValueError('''The parameter "converter" is required for cf2''')
            
            converter_name = oracle_parameters['converter'].get('name')
            if not converter_name:
                raise ValueError('''The parameter "name" for the converter is required for cf2''')
            
            converter = None
            feature_dim = oracle_parameters.get('feature_dim', 36)
            weight_dim = oracle_parameters.get('weight_dim', 28)
            if converter_name == 'tree_cycles':
                converter = CF2TreeCycleConverter(feature_dim=feature_dim)
            else:
                converter = DefaultFeatureAndWeightConverter(feature_dim=feature_dim,
                                                              weight_dim=weight_dim)
            lr = oracle_parameters.get('lr', 1e-3)
            batch_size_ratio = oracle_parameters.get('batch_size_ratio', .1)
            weight_decay = oracle_parameters.get('weight_decay', 5e-4)
            epochs = oracle_parameters.get('epochs', 100)
            fold_id = oracle_parameters.get('fold_id', 0)
            threshold = oracle_parameters.get('threshold', .5)
            
            return self.get_cf2(dataset, converter, feature_dim, weight_dim, lr,
                                      weight_decay, epochs, batch_size_ratio,
                                      threshold, fold_id, oracle_dict)
            
        elif oracle_name == 'dblp_coauthorship_custom_oracle':
            fold_id = oracle_parameters.get('fold_id', 0)                
            percentile = oracle_parameters.get('percentile', 75)
            first_train_timestamp = oracle_parameters.get('first_train_timestamp', 2000)
            
            return self.get_dblp_coauthorship_custom_oracle(dataset,
                                                            percentile=percentile,
                                                            fold_id=fold_id,
                                                            first_train_timestamp=first_train_timestamp,
                                                            config_dict=oracle_dict)
        elif oracle_name == 'dynamic_oracle':
            if 'base_oracle' not in oracle_dict['parameters']:
                raise ValueError('''The parameter "base_oracle" for the DynamicOracle is required''')
            
            if 'first_train_timestamp' not in oracle_dict['parameters']:
                raise ValueError('''The parameter "first_train_timestamp" for the DynamicOracle is required''')

            base_oracle = self.get_oracle_by_name(oracle_dict['parameters']['base_oracle'],
                                                  dataset,
                                                  emb_factory)
                        
            first_train_timestamp = oracle_dict['parameters']['first_train_timestamp'] 
            
            return self.get_dynamic_oracle(dataset,
                                           base_oracle,
                                           timestamp=first_train_timestamp,
                                           config_dict=oracle_dict)
            
        elif oracle_name == 'btc_alpha_oracle':
            return self.get_btc_alpha_oracle(dataset, config_dict=oracle_dict)
        
        elif oracle_name == 'bonanza_oracle':
            return self.get_bonanza_oracle(dataset, config_dict=oracle_dict)
        
        elif oracle_name == 'id_oracle':
            return self.get_id_oracle()
        # If the oracle name does not match any oracle in the factory
        else:
            raise ValueError('''The provided oracle name does not match any oracle provided by the factory''')
        
    def get_bonanza_oracle(self, dataset:Dataset, timestamp=-1, config_dict=None):
        clf = BonanzaOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter += 1
        return clf
    
    def get_btc_alpha_oracle(self, dataset:Dataset, timestamp=-1, config_dict=None):
        clf = BTCAlphaCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter += 1
        return clf
        
    def get_id_oracle(self):
        clf = IDOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=None)
        self._oracle_id_counter += 1
        return clf
    
    def get_dynamic_oracle(self, dataset: Dataset, base_oracle: Oracle, timestamp=-1, config_dict=None):
        clf = DynamicOracle(id=self._oracle_id_counter,
                            base_oracle=base_oracle,
                            oracle_store_path=self._oracle_store_path,
                            config_dict=config_dict)
        self._oracle_id_counter += 1
        clf.fit(dataset, timestamp=timestamp)
        return clf
        
    def get_dblp_coauthorship_custom_oracle(self,
                                            dataset: Dataset,
                                            percentile=75,
                                            fold_id=0,
                                            first_train_timestamp=0,
                                            config_dict=None):
        
        clf = DBLPCoAuthorshipCustomOracle(id=self._oracle_id_counter,
                                           percentile=percentile,
                                           fold_id=fold_id,
                                           first_train_timestamp=first_train_timestamp,
                                           oracle_store_path=self._oracle_store_path,
                                           config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf


    def get_knn(self, data: Dataset, embedder: Embedder, k, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = KnnOracle(id=self._oracle_id_counter,oracle_store_path=self._oracle_store_path,  emb=embedder, k=k, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_svm(self, data: Dataset, embedder: Embedder, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = SvmOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, emb=embedder, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_asd_custom_oracle(self, config_dict=None) -> Oracle:
        clf = ASDCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf

  
    
    def get_pt_syn_oracle(self, data: Dataset, split_index=-1, config_dict=None) -> Oracle:
        clf = SynNodeOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(data, split_index)
        return clf
        
    def get_trisqr_custom_oracle(self, config_dict=None) -> Oracle:
        clf = TrianglesSquaresCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf

    def get_tree_cycles_custom_oracle(self, config_dict=None) -> Oracle:
        clf = TreeCyclesCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf