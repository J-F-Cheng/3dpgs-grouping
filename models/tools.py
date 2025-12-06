import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def combine_ass(pose: Tensor, sel: Tensor, sel_bool, sel_first=False):
    if sel_first:
        raise NotImplementedError
    else:
        return_tensor = torch.zeros(sel.size(0), 8, device=sel.device)
        return_tensor[sel_bool, :-1] = pose
        return_tensor[:, -1:] = sel
    return return_tensor


def pyg_batch_to_batch(pyg_batch, batch_code, batch_size, batch_first=True):
    batch_list = []
    for batch_idx in range(batch_size):
        batch_sel = batch_code == batch_idx # Select batch data
        batch_list.append(pyg_batch[batch_sel])

    # we pad the seq
    padded_data = pad_sequence(batch_list, batch_first=True, padding_value=0) # keep it true
    mask = torch.zeros(padded_data.size(0), padded_data.size(1), device=padded_data.device, dtype=torch.int)
    for batch_idx in range(batch_size):
        mask[batch_idx, :batch_list[batch_idx].size(0)] = 1
    if not batch_first:
        padded_data = padded_data.permute(1, 0, 2)
        mask = mask.permute(1, 0)
    return padded_data, mask

def seq_back_to_pyg_batch(padded_data, mask, batch_first=True):
    # 初始化一个空列表来存储原始序列
    original_data = []
    if not batch_first:
        padded_data = padded_data.permute(1, 0, 2)
        mask = mask.permute(1, 0)
    # 遍历填充后的数据和掩码
    for padded_seq, seq_mask in zip(padded_data, mask):
        # 使用掩码确定实际数据的长度
        actual_length = seq_mask.sum()

        # 截断序列以去除填充部分
        original_seq = padded_seq[:actual_length]

        # 将截断后的序列添加到列表中
        original_data.append(original_seq)
    return torch.cat(original_data, dim=0)


def pyg_pack_sequences(pyg_batch, batch_code, batch_first=True):
    batch_list = []
    seq_lengths = []
    batch_size = max(batch_code) + 1
    for batch_idx in range(batch_size):
        batch_sel = batch_code == batch_idx # Select batch data
        seq_lengths.append(int(batch_sel.sum()))
        batch_list.append(pyg_batch[batch_sel])
    # 对序列长度进行排序（降序）
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    batch_list = batch_list[perm_idx]

    # 打包序列
    packed_input = pack_padded_sequence(batch_list, seq_lengths, batch_first=batch_first)

    return packed_input, perm_idx

def pack_sequence_pyg(packed_input, perm_idx, batch_first=True):
    # 解包序列
    padded_data, _ = pad_packed_sequence(packed_input, batch_first=batch_first)

    # 恢复原始顺序
    _, unperm_idx = perm_idx.sort(0)
    padded_data = padded_data[unperm_idx]

    return torch.cat(padded_data, dim=0)
