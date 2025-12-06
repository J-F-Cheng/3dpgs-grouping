from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import os
import torch

class MixPartDataLoader(Dataset):
    def __init__(self, conf, data_path):
        self.conf = conf
        self.category = conf.category
        self.data_path = data_path

    def __len__(self):
        file_names = os.listdir(self.data_path)
        return len(file_names)

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.category + '_data_' + str(idx) + '.pt')
        return torch.load(f_name)

class MixPartDataLoader_for_del(Dataset):
    def __init__(self, conf, data_path):
        self.conf = conf
        self.category = conf.category
        self.data_path = data_path
        self.file_names = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return torch.load(self.file_names[idx])

def mix_collect_fn(batches):
    '''
    Input: a list of dicts which contains part repository and the corresponding poses

    Output: 1. the merged batches for the graph neural network
            2. merged batch code

    '''
    return_batches = {"all_parts": [], "all_poses": [], "all_euler_poses": [], "total_parts": [], "batch_code": [], "data_valid": []}

    batch_size = len(batches)

    for batch_idx in range(batch_size):
        return_batches["all_parts"].append(batches[batch_idx]["all_parts"])
        return_batches["all_poses"].append(batches[batch_idx]["all_poses"])
        return_batches["all_euler_poses"].append(batches[batch_idx]["all_euler_poses"])
        return_batches["total_parts"].append(batches[batch_idx]["total_parts"])
        return_batches["batch_code"].append(batch_idx * torch.ones(batches[batch_idx]["total_parts"], dtype=torch.int64))
        return_batches["data_valid"].append(batches[batch_idx]["data_valid"])

    return_batches["all_parts"] = torch.cat(return_batches["all_parts"], dim=0)
    return_batches["all_poses"] = torch.cat(return_batches["all_poses"], dim=1)
    return_batches["all_euler_poses"] = torch.cat(return_batches["all_euler_poses"], dim=1)
    return_batches["batch_code"] = torch.cat(return_batches["batch_code"], dim=0)
    return_batches["data_valid"] = torch.cat(return_batches["data_valid"], dim=1)

    return return_batches


def create_fully_connected_edge_index(num_nodes, loop=True):
    """
    生成一个全连接的边索引列表。
    
    参数:
        num_nodes (int): 图中节点的数量。
        loop (bool): 是否在每个节点上添加自环。
    
    返回:
        edge_index (Tensor): 2 x E的张量，其中E是边的数量。
    """
    # 创建一个num_nodes x num_nodes的矩阵，初始化为1
    adj = torch.ones((num_nodes, num_nodes))
    
    # 如果不包括自环，则将对角线元素设置为0
    if not loop:
        adj.fill_diagonal_(0)
    
    # 将邻接矩阵转换为稀疏格式的边索引
    edge_index, _ = dense_to_sparse(adj)
    
    return edge_index


def mix_collect_fn_data_list(batches):
    batch_size = len(batches)
    return_list = []
    for batch_idx in range(batch_size):
        # create an empty pyg data
        data = Data()
        data.x = batches[batch_idx]["all_parts"].float()
        # print("data.x.size(0): ", data.x.size(0))
        # print("total_parts: ", batches[batch_idx]["total_parts"])
        total_parts = data.x.size(0)
        data.all_poses = batches[batch_idx]["all_poses"].permute(1, 0, 2).float()
        # data.all_euler_poses = batches[batch_idx]["all_euler_poses"]
        # data.total_parts = batches[batch_idx]["total_parts"]
        data.edge_index = create_fully_connected_edge_index(total_parts, loop=True)
        data.data_valid = batches[batch_idx]["data_valid"].permute(1, 0)
        # print("data.data_valid dtype: ", data.data_valid.dtype)
        return_list.append(data)
    return return_list


def random_mix_collect_fn(batches):
    return_batches = {"all_parts": [], "all_poses": [], "all_euler_poses": [], "total_parts": [], "batch_code": []}

    batch_size = len(batches)

    for batch_idx in range(batch_size):
        return_batches["all_parts"].append(batches[batch_idx]["all_parts"])
        return_batches["all_poses"].append(batches[batch_idx]["all_poses"])
        return_batches["all_euler_poses"].append(batches[batch_idx]["all_euler_poses"])
        return_batches["total_parts"].append(batches[batch_idx]["total_parts"])
        return_batches["batch_code"].append(batch_idx * torch.ones(batches[batch_idx]["total_parts"], dtype=torch.int64))

    return_batches["all_parts"] = torch.cat(return_batches["all_parts"], dim=0)
    return_batches["all_poses"] = torch.cat(return_batches["all_poses"], dim=0)
    return_batches["all_euler_poses"] = torch.cat(return_batches["all_euler_poses"], dim=0)
    return_batches["batch_code"] = torch.cat(return_batches["batch_code"], dim=0)

    return return_batches
