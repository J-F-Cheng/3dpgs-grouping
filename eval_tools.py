import os
import torch
import numpy as np


def eval_selection_batch(pred_sel_list, gt_sel_list, save_single_path=None, start_idx=None):
    """
    We use this function to obtain the FP, FN, TP, TN
    We use FP, FN, FP, TN to calculate the precision, recall, F1 score
    The sequence of the sel list is unknown, we need to calculate the best matching
    """
    # get the number of batch
    batch_size = len(pred_sel_list)
    # initialize the TP, FP, FN, TN
    tp = 0
    fp = 0
    fn = 0
    # tn = 0
    # initialize the list of batch macro_precision, macro_recall, macro_f1
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    # calculate the TP, FP, FN, TN
    for i in range(batch_size):
        # for pred and gt, the first dimension is the number of samples, we use best matching to calculate the TP, FP, FN, TN
        # get the pred number of samples
        # check if this sample is None
        if pred_sel_list[i] == None:
            continue
        pred_num_samples = pred_sel_list[i].size(0)
        # get the gt number of samples
        gt_num_samples = gt_sel_list[i].size(0)
        # initialize batch tp, fp, fn, tn
        batch_tp = 0
        batch_fp = 0
        batch_fn = 0
        # batch_tn = 0
        # calculate the best matching
        for j in range(gt_num_samples):
            # max_tp, max_fp, max_fn, max_tn
            max_tp = 0
            max_fp = 0
            max_fn = 0
            # max_tn = 0
            max_cur_score = 0
            gt_sel_origin = gt_sel_list[i][j]
            for k in range(pred_num_samples):
                # check if this sample is None
                pred_sel_origin = pred_sel_list[i][k]
                if pred_sel_origin == None:
                    continue
                # We only need the true part of both gt_sel and pred_sel
                need_parts = pred_sel_origin | gt_sel_origin
                gt_sel = gt_sel_origin[need_parts]
                pred_sel = pred_sel_origin[need_parts]
                # calculate the score
                # calculate the TP
                cur_tp = torch.sum(pred_sel & gt_sel)
                cur_fp = torch.sum(pred_sel & ~gt_sel)
                cur_fn = torch.sum(~pred_sel & gt_sel)
                total_num = pred_sel.size(0)
                cur_score = cur_tp / (total_num + 1e-8)
                if cur_score > max_cur_score:
                    # update max
                    max_cur_score = cur_score
                    max_tp = cur_tp
                    max_fp = cur_fp
                    max_fn = cur_fn
                    # max_tn = cur_tn
            # update tp, fp, fn, tn
            tp += max_tp
            fp += max_fp
            fn += max_fn
            # tn += max_tn
            # update batch_tp, batch_fp, batch_fn, batch_tn
            batch_tp += max_tp
            batch_fp += max_fp
            batch_fn += max_fn
            # batch_tn += max_tn
        # for each batch, calcuate the macro_precision, macro_recall, macro_f1, and append them
        # use batch value to calculate
        single_data_eval = {"tp": batch_tp, "fp": batch_fp, "fn": batch_fn}
        macro_eval_dict = calc_prf(single_data_eval)
        if save_single_path != None:
            # we merge two dicts
            save_dict = {**single_data_eval, **macro_eval_dict}
            np.save(os.path.join(save_single_path, str(start_idx + i)), save_dict)
        macro_precision_list.append(macro_eval_dict["precision"])
        macro_recall_list.append(macro_eval_dict["recall"])
        macro_f1_list.append(macro_eval_dict["f1_score"])
    return {"tp": tp, "fp": fp, "fn": fn, "macro_precision_list": macro_precision_list, 
            "macro_recall_list": macro_recall_list, "macro_f1_list": macro_f1_list, "batch_size": batch_size}

def jcard_sim(pred_sel_origin, gt_sel_origin):
    # We only need the true part of both gt_sel and pred_sel
    need_parts = pred_sel_origin | gt_sel_origin
    gt_sel = gt_sel_origin[need_parts]
    pred_sel = pred_sel_origin[need_parts]
    # calculate the score
    # calculate the TP
    cur_tp = torch.sum(pred_sel & gt_sel)
    total_num = pred_sel.size(0)
    cur_score = cur_tp / (total_num + 1e-8)
    return cur_score


def eval_selection_map_one(pred_sel_list, gt_sel_list, iou_thresh):
    """
    This function is used to evalute the average precision (AP) for each batch
    """
    # 初始化统计值
    TP = 0
    FP = 0
    FN = 0

    # 创建一个标记数组来跟踪哪些真实边界框已经被匹配
    matched = [False] * len(gt_sel_list)

    # 遍历每个预测边界框
    for pred_box in pred_sel_list:
        # 记录是否找到匹配的真实边界框
        found_match = False

        # 遍历每个真实边界框
        for i, true_box in enumerate(gt_sel_list):
            # 计算IoU
            if jcard_sim(pred_box, true_box) >= iou_thresh:
                # 如果找到匹配且该真实边界框还未被匹配，则视为TP
                if not matched[i]:
                    matched[i] = True
                    TP += 1
                    found_match = True
                    break

        # 如果没有找到匹配，则视为FP
        if not found_match:
            FP += 1

    # 所有未匹配的真实边界框被视为FN
    FN = matched.count(False)

    return TP, FP, FN

def eval_selection_map_batch(batch_pred_sel_list, batch_gt_sel_list, iou_thresh):
    # get the number of batch
    batch_size = len(batch_pred_sel_list)
    all_TP = 0
    all_FP = 0
    all_FN = 0
    for i in range(batch_size):
        TP, FP, FN = eval_selection_map_one(batch_pred_sel_list[i], batch_gt_sel_list[i], iou_thresh)
        all_TP += TP
        all_FP += FP
        all_FN += FN
        # if save_single_path != None:
        #     # we merge two dicts
        #     save_dict = {"tp": TP, "fp": FP, "fn": FN}
        #     np.save(os.path.join(save_single_path, str(start_idx + i)), save_dict)
    return all_TP, all_FP, all_FN

def eval_map_multi_iou_thresh(batch_pred_sel_list, batch_gt_sel_list, iou_thresh_list, save_single_path=None, start_idx=None):
    # initialize TP, FP, FN list
    TP_list = []
    FP_list = []
    FN_list = []
    # calculate TP, FP, FN for each iou_thresh
    for iou_thresh in iou_thresh_list:
        TP, FP, FN = eval_selection_map_batch(batch_pred_sel_list, batch_gt_sel_list, iou_thresh)
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
    # save TP, FP, FN list
    if save_single_path != None:
        np.save(os.path.join(save_single_path, "batch_TP_{}".format(start_idx)), TP_list)
        np.save(os.path.join(save_single_path, "batch_FP_{}".format(start_idx)), FP_list)
        np.save(os.path.join(save_single_path, "batch_FN_{}".format(start_idx)), FN_list)    
    # return numpy version
    return np.array(TP_list), np.array(FP_list), np.array(FN_list)

# calculate precision, recall, F1 score
def calc_prf(eval_dict):
    """
    calculate the precision, recall, F1 score
    """
    tp = eval_dict["tp"]
    fp = eval_dict["fp"]
    fn = eval_dict["fn"]
    # calculate the precision
    precision = tp / (tp + fp + 1e-8)
    # calculate the recall
    recall = tp / (tp + fn + 1e-8)
    # calculate the F1 score
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def reform_gt_sel_list(batch_code, gt_sel_list):
    """
    gt_sel_list: N * total_num_parts
    We reform it as a batch_size list: N * batch_1; N * batch_2; ...; N * batch_batch_size
    """
    batch_size = max(batch_code) + 1
    # initialize the reform_gt_sel_list
    reform_gt_sel_list = []
    for i in range(batch_size):
        cur_batch_gt_sel_list = gt_sel_list[:, batch_code == i]
        after_remove_cur_batch_gt_sel_list = []
        for n in range(cur_batch_gt_sel_list.size(0)):
            # append only the current batch contains True
            if torch.sum(cur_batch_gt_sel_list[n]) > 0:
                after_remove_cur_batch_gt_sel_list.append(cur_batch_gt_sel_list[n])
        # stack the after_remove_cur_batch_gt_sel_list
        after_remove_cur_batch_gt_sel_list = torch.stack(after_remove_cur_batch_gt_sel_list, dim=0)
        reform_gt_sel_list.append(after_remove_cur_batch_gt_sel_list)
    return reform_gt_sel_list