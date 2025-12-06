import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import DataParallel
from .quaternion import euler_to_quaternion, qeuler

def create_directory(dir_path):
    if os.path.exists(dir_path):
        # If directory exists, ask the user if they want to delete the directory and its contents
        answer = input(f"The directory '{dir_path}' already exists. Do you want to delete the directory and its contents? [y/n]: ")
        if answer.lower() == "y":
            # Delete directory
            shutil.rmtree(dir_path)
        else:
            return False
    # Create directory
    os.makedirs(dir_path)
    return True


def save_network(network, dir):
    if isinstance(network, DataParallel):
        torch.save(network.module.state_dict(), dir)
    else:
        torch.save(network.state_dict(), dir)

def euler_to_quaternion_torch_data(e, order, device):
    """input: n * 6
        output: n * 7"""
    e_clone = e.clone()
    qua_data = torch.zeros(e_clone.size(0), 7)
    qua_data[:, :3] = e_clone[:, :3]
    qua_data[:, 3:] = torch.tensor(euler_to_quaternion(e_clone[:, 3:].cpu().numpy(), order), device=device)
    return qua_data.to(device)

def sel_euler_to_quaternion_torch_data(e, order, device, sel_first=False):
    '''
    Transform euler data with selection place to the quaternion data
    '''

def quaternion_to_euler_torch_data(qua, order, device):
    qua_clone = qua.clone()
    e_data = torch.zeros(qua_clone.size(0), 6)
    e_data[:, :3] = qua_clone[:, :3]
    e_data[:, 3:] = qeuler(qua_clone[:, 3:], order)
    return e_data.to(device)

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

def matrix_sns_plot(mat_matrix, mat_matrix_mask, gen_img_path, save_idx=0):
    # use sns to plot the match matrix
    for mat_idx in range(mat_matrix.size(0)):
        # save the match matrix
        valid_n = mat_matrix_mask[mat_idx, 0].sum().item()
        save_mat = mat_matrix[mat_idx][:valid_n, :valid_n].detach().cpu().numpy()
        mat_save_path = os.path.join(gen_img_path, f"mat_matrix_{mat_idx + save_idx}.png")
        plt.figure()
        sns.heatmap(save_mat).get_figure().savefig(mat_save_path)
        plt.close()

def calculate_and_apply_scale_and_center(points):
    """
    Calculate and apply a uniform scaling factor, and center the point clouds.
    :param points: A tensor of shape (K, N, 3), representing K point clouds with N points each.
    :return: The scaled and centered point cloud data.
    """
    centroids = torch.mean(points, dim=1, keepdim=True)
    centered_points = points - centroids
    
    min_vals, _ = torch.min(centered_points, dim=1, keepdim=True)
    max_vals, _ = torch.max(centered_points, dim=1, keepdim=True)
    scale_factors = 1.0 / (max_vals - min_vals)
    uniform_scale_factors = torch.min(scale_factors, dim=2, keepdim=True).values
    
    scaled_points = centered_points * uniform_scale_factors
    
    return scaled_points

def batch_calculate_and_apply_scale_and_center(batch_points, batch_code):
    batch_size = max(batch_code) + 1
    return_list = []
    for i in range(batch_size):
        return_list.append(calculate_and_apply_scale_and_center(batch_points[batch_code == i]))
    return torch.cat(return_list, dim=0)
