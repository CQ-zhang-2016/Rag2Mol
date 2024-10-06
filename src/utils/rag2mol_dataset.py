import lmdb
import pickle
from torch.utils.data import Dataset
import os
import torch
import copy
import random


class RAGDataset(Dataset):

    def __init__(self, raw_path='./data/RAG', transform=None, transform_ret=None):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = os.path.join(self.raw_path, 'crossdocked_pocket10_processed.lmdb')
        self.name2id_path = os.path.join(self.raw_path, 'name2id.pt')
        self.keys = None
        self.transform = transform
        self.transform_ret = transform_ret
        if not os.path.exists(self.processed_path):
            raise Exception('Please processing the data first!')  
        self.name2id = torch.load(self.name2id_path)
    
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        self.db = lmdb.open(
            self.processed_path,
            map_size=48*(1024*1024*1024),   # 48 GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        
    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, idx):
        self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        assert data.protein_pos.size(0) > 0, 'preprocess failed!'

        ret_id = random.randint(0, len(data.retrieval_data)-1)

        if self.transform != None:
            data = self.transform(data)
        if self.transform_ret != None:    
            data.retrieval_data = self.transform_ret(data.retrieval_data[ret_id])        
            
        return data