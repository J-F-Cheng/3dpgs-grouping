import torch
import torch.nn as nn
import torch.nn.functional as F
from .PointNet import PointNet
from .tools import pyg_batch_to_batch, seq_back_to_pyg_batch
from torch_geometric.nn import EdgeConv
from mix_part_tools.utils import batch_calculate_and_apply_scale_and_center


# CombNet is a general idea for 3D point cloud combinatorial optimization problems.
# It contains three parts:
# 1. Part feature extraction, it can be PointNet, PointNet++, etc.
# 2. Part Relation Modeling, it can be graph neural network, transformer, etc.
# 3. Part Combination Proposal Network, it could be a simple GNN, where given N per-part features, it outputs a N x N combination proposal matrix.

class RelationNetGNN(nn.Module):
    def __init__(self, feat_len, n_layers):
        super(RelationNetGNN, self).__init__()
        self.feat_len = feat_len
        self.n_layers = n_layers
        self.relation_net = nn.ModuleList()
        for i in range(n_layers):
            self.relation_net.append(self.create_one_layer())
            
    def create_one_layer(self):
        mlp = nn.Sequential(
            nn.Linear(self.feat_len * 2, self.feat_len),
            nn.ReLU(True),
            nn.Linear(self.feat_len, self.feat_len),
        )
        return EdgeConv(mlp)
    
    def forward(self, x, edge_index):
        # print(f'Inside model - num graphs: {data.num_graphs}, '
        #       f'device: {data.batch.device}, '
        #       f'x shape: {data.x.shape}, ')
        for layer in self.relation_net:
            x = layer(x, edge_index)
            x = torch.relu(x)
        return x
        

class MatchMatrixNet_V2(nn.Module):
    def __init__(self, feat_len):
        super().__init__()
        self.feat_proj = nn.Sequential(nn.Linear(feat_len, feat_len),
                                        nn.ReLU(True),
                                        nn.Linear(feat_len, feat_len))
    
    def forward(self, features):
        # print("You are using MatchMatrixNet_V2!")
        # proj the features
        features = self.feat_proj(features)
        # l2 normalize the features
        features = F.normalize(features, p=2, dim=-1)

        expanded_features1 = features.unsqueeze(2).expand(-1, -1, features.size(1), -1)
        expanded_features2 = features.unsqueeze(1).expand(-1, features.size(1), -1, -1)
        # perform dot product
        mat_matrix = torch.sum(expanded_features1 * expanded_features2, dim=-1)
        return mat_matrix

class Deep3DS_base(nn.Module):
    def __init__(self, feat_len, rela_layers, sel_first, **kwargs):
        super(Deep3DS_base, self).__init__()
        self.feat_len = feat_len
        self.rela_layers = rela_layers
        self.sel_first = sel_first
        self.feat_ext = PointNet(feat_len)
        self.rela_net = RelationNetGNN(feat_len, n_layers=rela_layers)
        self.matrix_net = MatchMatrixNet_V2(feat_len)
        # the output from the matrix net is a N x N combination proposal matrix
        # next, we use this information to calculate the selection matrix
        # a direct way is to use conv2d to generate the selection matrix

    def net_infer(self, data_x, data_edge_index, data_batch):
        emb_pcs = self.feat_ext(data_x) # get the per-part features
        emb_pcs = self.rela_net(emb_pcs, data_edge_index) # get relation features
        seq_emb_pcs, vec_mask = pyg_batch_to_batch(emb_pcs, data_batch, data_batch.max() + 1) # transform to sequences
        float_vec_mask = vec_mask.float() # transform to float
        mat_matrix_mask = torch.matmul(float_vec_mask.unsqueeze(-1), float_vec_mask.unsqueeze(-2)) # get the matrix mask
        float_mat_matrix_mask = mat_matrix_mask.float()
        mat_matrix = self.matrix_net(seq_emb_pcs) # get the matching matrix
        # change the range from [-1, 1] to [0, 1]
        mat_matrix = (mat_matrix + 1.0) / 2.0
        mat_matrix = mat_matrix * float_mat_matrix_mask
        vec_mask_bool = vec_mask > 0.5
        mat_matrix_mask_bool = mat_matrix_mask > 0.5
        return mat_matrix, vec_mask_bool, mat_matrix_mask_bool # for further consideration, we return the boolean version of the masks

    def forward(self, data):
        # forward the network to get the features and masks
        mat_matrix, vec_mask, mat_matrix_mask = self.net_infer(data.x, data.edge_index, data.batch)

        # get the matching matrix label
        if self.sel_first:
            selection = data.all_poses[:, :, 0]
        else:
            selection = data.all_poses[:, :, -1]
        valid_combo = selection * data.data_valid.float()
        seq_valid_combo, _ = pyg_batch_to_batch(valid_combo, data.batch, data.batch.max() + 1)
        seq_valid_combo = seq_valid_combo > 0.5
        label_matrix = self.create_match_matrix_label(seq_valid_combo.permute(0, 2, 1))
        float_label_matrix = label_matrix.float()
        # change range from [0, 1] to [-1, 1] in the training. In the graph search, we will change it back to [0, 1]
        # float_label_matrix = float_label_matrix * 2 - 1

        # get the selection matrix label
        mat_matrix_mask_float = mat_matrix_mask.float()

        # calculate the losses
        mat_loss = F.mse_loss(mat_matrix, float_label_matrix, reduction='none')
        # print("mat_loss size: ", mat_loss.size())
        mat_loss = mat_loss * mat_matrix_mask_float
        mat_loss = torch.mean(mat_loss, dim=(1, 2))
        return mat_loss

    def inference(self, data, threshold=0.5):
        # This function is only used in single GPU testing
        # inference the network to get the features and masks
        mat_matrix, vec_mask, mat_matrix_mask = self.net_infer(data.x, data.edge_index, data.batch)
        # we transform data from [-1, 1] to [0, 1]
        # mat_matrix = (mat_matrix + 1) / 2
        mat_matrix_pyg = seq_back_to_pyg_batch(mat_matrix, vec_mask)
        pred_sel_list, pred_all_sel_tensor = self.find_components_batch(mat_matrix_pyg, data.batch, threshold)
        return pred_sel_list, pred_all_sel_tensor, mat_matrix, mat_matrix_mask
    
    def find_components_batch(self, matrix_batch, batch_code, threshold):
        batch_size = batch_code.max() + 1
        components_batch = []
        max_sel = 0
        for i in range(batch_size):
            matrix = matrix_batch[batch_code == i]
            # we use mask to remove the paddings
            valid_n = matrix.size(0)
            matrix = matrix[:, :valid_n]
            components = self.find_components_one_graph(matrix, threshold)
            if len(components) > max_sel:
                max_sel = len(components)
            components = torch.stack(components, dim=0)
            components_batch.append(components)
        # padding components to the same size based on the max_sel
        padded_components_batch = []
        for i in range(batch_size):
            components = components_batch[i]
            pad_num = max_sel - components.size(0)
            if pad_num > 0:
                padding = torch.zeros(pad_num, components.size(1), dtype=torch.bool, device=components.device)
                components = torch.cat([components, padding], dim=0)
            padded_components_batch.append(components)
        padded_components_batch = torch.cat(padded_components_batch, dim=1)
        return components_batch, padded_components_batch

        
    def dfs(self, matrix, visited, start, component, threshold):
        stack = [start]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                # 获取所有连接的节点及其概率，使用tensor操作
                neighbors = torch.where(matrix[node] >= threshold)[0]
                # 按概率降序排序，需要提取相应的概率值
                probabilities = matrix[node][neighbors]
                sorted_neighbors = neighbors[torch.argsort(probabilities, descending=True)]
                # 仅添加满足阈值的邻居
                for adj in sorted_neighbors:
                    if not visited[adj.item()]:
                        stack.append(adj.item())

    def find_components_one_graph(self, matrix, threshold):
        print("matrix size: ", matrix.size())
        n = matrix.size(0)
        visited = [False] * n
        components = []
        
        for i in range(n):
            if not visited[i]:
                component = []
                self.dfs(matrix, visited, i, component, threshold)
                print(f"component {i}: ", component)
                print(f"component size: {len(component)}")
                component_one_hot = torch.zeros(n, dtype=torch.bool)
                component_one_hot[component] = True
                components.append(component_one_hot)
        
        return components

    # def post_process(self, results, batch_code):
    #     batch_size = max(batch_code) + 1
    #     for i in range(batch_size):
    #         sel_batch_results = results[batch_code == i]
    #         for j in range(sel_batch_results.shape[0]):
    #             # we check each j-th and set the largest one to 1, others to 0
    #             max_index = torch.argmax(sel_batch_results[j])
    #             sel_batch_results[j] = torch.zeros_like(sel_batch_results[j])
    #             sel_batch_results[j][max_index] = 1
    #             results[batch_code == i] = sel_batch_results
    #     return results
        

    def create_match_matrix_label(self, combinations):
        B, M, N = combinations.shape  # B是batch大小，M是每个batch中的组合数，N是元素数量
        match_matrix = torch.zeros((B, N, N), dtype=torch.bool, device=combinations.device)  # 创建一个B个N*N的False矩阵
        
        for b in range(B):  # 对每个batch进行处理
            for combo in combinations[b]:
                indices = torch.nonzero(combo, as_tuple=True)[0]  # 获取所有设为True的索引
                # 更新对应的batch的匹配矩阵
                match_matrix[b, indices[:, None], indices] = True

        return match_matrix

class Deep3DS_alpha(Deep3DS_base):
    def __init__(self, feat_len, rela_layers, sel_first, **kwargs):
        super().__init__(feat_len, rela_layers, sel_first)
        self.data_norm = kwargs.get('data_norm', False)
        print("Your data normalization: ", self.data_norm)
        
    def net_infer(self, data_x, data_edge_index, data_batch):
        if self.data_norm:
            input_data_x = batch_calculate_and_apply_scale_and_center(data_x, data_batch)
            # print("You are using data normalization!")
        else:
            input_data_x = data_x
        emb_pcs = self.feat_ext(input_data_x) # get the per-part features
        emb_pcs = self.rela_net(emb_pcs, data_edge_index) # get relation features
        seq_emb_pcs, vec_mask = pyg_batch_to_batch(emb_pcs, data_batch, data_batch.max() + 1) # transform to sequences
        float_vec_mask = vec_mask.float() # transform to float
        mat_matrix_mask = torch.matmul(float_vec_mask.unsqueeze(-1), float_vec_mask.unsqueeze(-2)) # get the matrix mask
        float_mat_matrix_mask = mat_matrix_mask.float()
        mat_matrix = self.matrix_net(seq_emb_pcs) # get the matching matrix
        # change the range from [-1, 1] to [0, 1]
        # mat_matrix = (mat_matrix + 1.0) / 2.0
        mat_matrix = mat_matrix * float_mat_matrix_mask
        vec_mask_bool = vec_mask > 0.5
        mat_matrix_mask_bool = mat_matrix_mask > 0.5
        return mat_matrix, vec_mask_bool, mat_matrix_mask_bool # for further consideration, we return the boolean version of the masks

    def forward(self, data):
        # forward the network to get the features and masks
        mat_matrix, vec_mask, mat_matrix_mask = self.net_infer(data.x, data.edge_index, data.batch)

        # get the matching matrix label
        if self.sel_first:
            selection = data.all_poses[:, :, 0]
        else:
            selection = data.all_poses[:, :, -1]
        valid_combo = selection * data.data_valid.float()
        seq_valid_combo, _ = pyg_batch_to_batch(valid_combo, data.batch, data.batch.max() + 1)
        seq_valid_combo = seq_valid_combo > 0.5
        label_matrix = self.create_match_matrix_label(seq_valid_combo.permute(0, 2, 1))
        float_label_matrix = label_matrix.float()
        # change range from [0, 1] to [-1, 1] in the training. In the graph search, we will change it back to [0, 1]
        float_label_matrix = float_label_matrix * 2.0 - 1.0

        # get the selection matrix label
        mat_matrix_mask_float = mat_matrix_mask.float()

        # calculate the losses
        mat_loss = F.mse_loss(mat_matrix, float_label_matrix, reduction='none')
        # print("mat_loss size: ", mat_loss.size())
        mat_loss = mat_loss * mat_matrix_mask_float
        mat_loss = torch.mean(mat_loss, dim=(1, 2))
        return mat_loss

    def inference(self, data, threshold=0.5, **kwargs):
        # This function is only used in single GPU testing
        # inference the network to get the features and masks
        mat_matrix, vec_mask, mat_matrix_mask = self.net_infer(data.x, data.edge_index, data.batch)
        # we transform data from [-1, 1] to [0, 1]
        mat_matrix = (mat_matrix + 1.0) / 2.0
        mat_matrix_pyg = seq_back_to_pyg_batch(mat_matrix, vec_mask)
        pred_sel_list, pred_all_sel_tensor = self.find_components_batch(mat_matrix_pyg, data.batch, threshold)
        return pred_sel_list, pred_all_sel_tensor, mat_matrix, mat_matrix_mask
