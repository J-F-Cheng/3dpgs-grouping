import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import random
import importlib
from mix_part_data import MixPartDataLoader, mix_collect_fn_data_list
from mix_part_tools import utils
from mix_part_tools.assembly_tools import batch_assembly
from eval_tools import eval_selection_batch, reform_gt_sel_list, eval_map_multi_iou_thresh
from torch_geometric.data import Batch
from models.tools import pyg_batch_to_batch
from models import model_file_dict

def calculate_rec_prec(tp_list, fp_list, fn_list):
    # 存储精度和召回率
    precisions = []
    recalls = []

    # 对每个阈值计算精度和召回率
    for tp, fp, fn in zip(tp_list, fp_list, fn_list):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions


def calculate_from_rec_pre(recalls, precisions):
    ap_results = []
    f1_results = []
    for rec, pre in zip(recalls, precisions):
        ap_results.append(rec * pre) # a special AP calculation without confidence output
        f1_results.append(2 * rec * pre / (rec + pre + 1e-8))
    return ap_results, f1_results

def get_avg_ap_f1(ap_results, f1_results, scale=[0.5, 0.95], step=0.05):
    ap_results_choose = ap_results[int((scale[0] + 0.01) / step) : int((scale[1] + 0.01 + step) / step)]
    f1_results_choose = f1_results[int((scale[0] + 0.01) / step) : int((scale[1] + 0.01 + step) / step)]
    return np.mean(ap_results_choose), np.mean(f1_results_choose)
    
def get_single_ap_f1(ap_results, f1_results, single_thresh=0.5, step=0.05):
    ap_results_choose = ap_results[int((single_thresh + 0.01) / step)]
    f1_results_choose = f1_results[int((single_thresh + 0.01) / step)]
    return ap_results_choose, f1_results_choose

def main(conf):
    gen_img_path = os.path.join(conf.base_dir, conf.exp_name, "gen_img")
    if not os.path.exists(gen_img_path):
        os.makedirs(gen_img_path)
    ref_img_path = os.path.join(conf.base_dir, conf.exp_name, "ref_img")
    if not os.path.exists(ref_img_path):
        os.makedirs(ref_img_path)
    single_results_path = os.path.join(conf.base_dir, conf.exp_name, "test_results")
    if not os.path.exists(single_results_path):
        os.makedirs(single_results_path)
    map_batch_results_path = os.path.join(conf.base_dir, conf.exp_name, "map_batch_results")
    if not os.path.exists(map_batch_results_path):
        os.makedirs(map_batch_results_path)
    print('-'*70)
    print('testing dataset size:')
    test_data_path = os.path.join(conf.data_dir, conf.category, "test") # You must indicate whether it is training dataset or testing dataset!
    test_set = MixPartDataLoader(conf, test_data_path)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=conf.val_batch_size, shuffle=False, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn_data_list, worker_init_fn=utils.worker_init_fn)
    
    model_file = model_file_dict[conf.model_type]
    model_module = importlib.import_module(f"models.{model_file}")
    model_cls = getattr(model_module, conf.model_type)
    model = model_cls(conf.feat_len, conf.rela_layers, conf.sel_first, n_heads=conf.n_heads, dropout=conf.dropout, data_norm=conf.data_norm).to(conf.device)
    print("Your model is: ", model_cls)
    model.load_state_dict(torch.load(conf.model_path, map_location=conf.device))
    model.eval()
    # initialize the TP, FP, FN
    tp = 0
    fp = 0
    fn = 0
    # initialize the list of batch macro_f1
    macro_f1_list = []
    step = 0.05
    defined_scale = [0.5, 0.95]
    map_threshold = np.arange(0.0, 1.01, step)
    map_tp_list = np.zeros(len(map_threshold))
    map_fp_list = np.zeros(len(map_threshold))
    map_fn_list = np.zeros(len(map_threshold))
    with torch.no_grad():
        save_idx = 0
        save_results_idx = 0
        for _, val_batch in enumerate(test_loader):
            val_batch = Batch.from_data_list(val_batch).to(conf.device)
            test_batch = val_batch.x
            val_all_poses = val_batch.all_poses.permute(1, 0, 2).cpu()
            test_batch_code = val_batch.batch
            with torch.no_grad():
                pred_sel_list, pred_all_sel_tensor, mat_matrix, mat_matrix_mask = model.inference(val_batch, conf.sel_thre, s_threshold=conf.svd_thre)
                utils.matrix_sns_plot(mat_matrix, mat_matrix_mask, gen_img_path, save_idx)
            gt_sel_tensor = val_all_poses[:, :, -1]
            gt_sel_tensor = gt_sel_tensor > conf.sel_thre
            if conf.gt_matrix_plot:
                plot_gt_matrix, _ = pyg_batch_to_batch(gt_sel_tensor.permute(1, 0), val_batch.batch, val_batch.batch.max() + 1)
                plot_gt_matrix = model.create_match_matrix_label(plot_gt_matrix.permute(0, 2, 1))
                utils.matrix_sns_plot(plot_gt_matrix, mat_matrix_mask, ref_img_path, save_idx)
            # get gt_sel_list
            gt_sel_list = reform_gt_sel_list(test_batch_code, gt_sel_tensor)
            eval_dict = eval_selection_batch(pred_sel_list, gt_sel_list, single_results_path, save_results_idx)
            cur_tp_list, cur_fp_list, cur_fn_list = eval_map_multi_iou_thresh(pred_sel_list, gt_sel_list, map_threshold, map_batch_results_path, save_results_idx)
            map_tp_list += cur_tp_list
            map_fp_list += cur_fp_list
            map_fn_list += cur_fn_list

            save_results_idx += len(pred_sel_list)
            # print(eval_result_dict)
            tp += eval_dict["tp"]
            fp += eval_dict["fp"]
            fn += eval_dict["fn"]
            macro_f1_list += eval_dict["macro_f1_list"]
            gt_poses = val_all_poses.to(conf.device)[:, :, :-1].sum(dim=0)
            gt_poses = gt_poses.unsqueeze(0)
            gt_poses = gt_poses.repeat(pred_all_sel_tensor.size(0), 1, 1)
            gen_sel_poses = torch.cat([gt_poses, pred_all_sel_tensor.float().unsqueeze(-1).to(conf.device)], dim=-1)
            if not conf.no_render:
                for gen_idx in range(gen_sel_poses.size(0)):
                    batch_assembly(conf, test_batch, gen_sel_poses[gen_idx].to(conf.device), test_batch_code, pose_type=conf.render_pose_type,
                                                    data_idx_start=save_idx, gen_idx=gen_idx, save_fn=gen_img_path)
                for gen_idx in range(val_all_poses.size(0)):
                    batch_assembly(conf, test_batch, val_all_poses[gen_idx].to(conf.device), test_batch_code, pose_type=conf.render_pose_type,
                                data_idx_start=save_idx, gen_idx=gen_idx, save_fn=ref_img_path)
            save_idx += conf.val_batch_size
    # calculate f1 scores
    macro_f1 = np.mean(macro_f1_list)
    group_recs_at_all_thre, group_precs_at_all_thre = calculate_rec_prec(map_tp_list, map_fp_list, map_fn_list)
    group_aps_at_all_thre, group_f1s_at_all_thre = calculate_from_rec_pre(group_recs_at_all_thre, group_precs_at_all_thre)
    _, group_f1 = get_avg_ap_f1(group_aps_at_all_thre, group_f1s_at_all_thre, scale=defined_scale, step=step)
    _, f1_50 = get_single_ap_f1(group_aps_at_all_thre, group_f1s_at_all_thre, single_thresh=0.5, step=step)
    _, f1_75 = get_single_ap_f1(group_aps_at_all_thre, group_f1s_at_all_thre, single_thresh=0.75, step=step)
    print("f1_50: ", f1_50)
    print("f1_75: ", f1_75)
    print("group_f1_05_095: ", group_f1)

    # use tp, fp, fn to calculate micro_precision, micro_recall, micro_f1
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    eval_result_dict = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "f1_50": f1_50,
        "f1_75": f1_75,
        "group_f1": group_f1,
    }
    print(eval_result_dict)
    np.save(os.path.join(conf.base_dir, conf.exp_name, "eval_result_dict.npy"), eval_result_dict)
    # save it as txt file
    with open(os.path.join(conf.base_dir, conf.exp_name, "eval_result_dict.txt"), "w") as f:
        f.write(str(eval_result_dict))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # experimental setting
    parser.add_argument('--base_dir', type=str, default='logs', help='model def file')
    parser.add_argument('--exp_name', type=str, default='my_exp_1', help='Please set your exp name, all the output will be saved in the folder with this exp_name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

    # datasets parameters:
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--data_dir', type=str, default='./data_output', help='data directory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sel_first', action='store_true', default=False, help='selection at the first place')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--rela_layers', type=int, default=3, help='relation network layers')
    parser.add_argument('--n_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')

    parser.add_argument('--use_GGAM_oper', action='store_true', default=False)

    # model path
    parser.add_argument('--model_path', type=str, default=None, help='The model path for validation.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epoch_save', type=int, default=100)
    parser.add_argument('--val_every_epochs', type=int, default=100)
    parser.add_argument('--cont_train_start', type=int, default=0, help='If you want to continue training, please set this option greater than 0.')
    parser.add_argument('--cont_path', type=str, default=None, help='The model path for continue training.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--print_loss', type=int, default=100)

    # Validation options
    parser.add_argument('--val_batch_size', type=int, default=1)

    # Sampler options
    parser.add_argument('--sel_sampler', type=str, default='PC_origin', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--assem_sampler', type=str, default='PC_origin', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--sel_num_steps', type=int, default=500)
    parser.add_argument('--snr', type=float, default=0.16)
    parser.add_argument('--t0', type=float, default=1.0)
    parser.add_argument('--cor_steps', type=int, default=1)
    parser.add_argument('--cor_final_steps', type=int, default=1)
    parser.add_argument('--noise_decay_pow', type=int, default=1)
    parser.add_argument('--max_sampling', type=int, default=15)
    parser.add_argument('--model_type', type=str, default='comp_net', help='model type option')
    parser.add_argument('--data_norm', action='store_true', default=False, help='normalize the data')
    
    # assembly parameters
    parser.add_argument('--pose_type', type=str, default='euler', help='poses type option')
    parser.add_argument('--euler_type', type=str, default='xyz', help='what is the euler type: e.g. xyz')
    parser.add_argument('--sel_thre', type=float, default=0.5) # selection threshold
    parser.add_argument('--svd_thre', type=float, default=0.5) # svd threshold

    # rendering parameters
    parser.add_argument('--no_render', action='store_true', default=False, help='Whether to render the image')
    parser.add_argument('--obj_png', type=str, default='png', help='Generation options: obj, png, both and no')
    parser.add_argument('--render_img_size', type=int, default=512)
    parser.add_argument('--render_pose_type', type=str, default='qua')
    parser.add_argument('--gt_matrix_plot', action='store_true', default=False, help='Whether to plot the matrix')



    conf = parser.parse_args()

    # control randomness
    if conf.seed >= 0:
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)

    main(conf)
