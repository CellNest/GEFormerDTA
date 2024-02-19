import torch
from torch_geometric.data import Batch


def collate(data_list):
    proteinBach = Batch.from_data_list([data[1] for data in data_list])
    smileBach = Batch.from_data_list([data[0] for data in data_list])
    return smileBach, proteinBach
