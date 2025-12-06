import os
import torch
import torch.nn as nn
import random
import numpy as np
import importlib
from tqdm import tqdm
from models import deep3ds, model_file_dict
from argparse import ArgumentParser
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import imageio
from mix_part_data import MixPartDataLoader, mix_collect_fn_data_list
from mix_part_tools import utils
from mix_part_tools.assembly_tools import batch_assembly
from torch.utils.data import DataLoader
from define_dict import DATASET_POSE_TYPE, INPUT_DIM
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from eval_tools import eval_selection_batch, calc_prf, reform_gt_sel_list

# MULTI_GPU = False

def main(conf):
    # Create exp and log folder
    base_path = os.path.join(conf.base_dir, conf.exp_name)
    # Use wandb to log the experiment
    wandb.init(project='Deep3DS', name=conf.exp_name, config=conf)
    print('experiment start.')
    print('-'*70)
    print('training dataset size:')
    train_data_path = os.path.join(conf.data_dir, conf.category, "train") # You must indicate whether it is training dataset or testing dataset!
    train_set = MixPartDataLoader(conf, train_data_path)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, pin_memory=True, drop_last=True,
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn_data_list, worker_init_fn=utils.worker_init_fn)
    
    print('-'*70)
    print('testing dataset size:')
    test_data_path = os.path.join(conf.data_dir_test, conf.category, "test") # You must indicate whether it is training dataset or testing dataset!
    test_set = MixPartDataLoader(conf, test_data_path)
    print(len(test_set))
    

    # model = mml.RelationNetGNN(conf.feat_len, 3, conf.device).to(conf.device)
    # model = mml.PygTestingNet(conf.device).to(conf.device)
    model_file = model_file_dict[conf.model_type]
    model_module = importlib.import_module(f"models.{model_file}")
    model_cls = getattr(model_module, conf.model_type)
    model = model_cls(conf.feat_len, conf.rela_layers, conf.sel_first, n_heads=conf.n_heads, dropout=conf.dropout, data_norm=conf.data_norm).to(conf.device)
    print("Your model is: ", model_cls)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
        MULTI_GPU = True
    else:
        MULTI_GPU = False
        print("Single GPU training!")

    # model save path
    model_save_path = os.path.join(base_path, 'model_save')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    global_train_idx = 0
    for epoch in range(conf.epochs):
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            # print("before network parts: ", batch[1].x.size())
            # print("edge_index: ", batch[1].edge_index.size())
            # print("pose: ", batch[1].all_poses.size())
            # print("total_parts: ", batch[1].total_parts)
            # print("data_valid: ", batch[1].data_valid.size())
            if not MULTI_GPU:
                batch = Batch.from_data_list(batch).to(conf.device)
            mat_loss = model(batch) # only for testing
            mat_loss = torch.mean(mat_loss)
            loss = mat_loss
            wandb.log({"mat_loss": mat_loss.item()})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_train_idx += 1
            if global_train_idx % conf.print_loss == 0:
                tqdm.write(f'Epoch {epoch}, Iter {global_train_idx}, mat_loss: {mat_loss.item()}')
        # Save model
        if (epoch + 1) % conf.epoch_save == 0:
            utils.save_network(model, os.path.join(model_save_path, 'model_epoch_{}.pth'.format(epoch)))
            print('model saved.')
        # validation one batch
        if (epoch + 1) % conf.val_every_epochs == 0:
            # try:
            # create validation folder
            gen_folder = os.path.join(base_path, f'validation_{epoch}', "gen")
            ref_folder = os.path.join(base_path, f'validation_{epoch}', "ref")
            match_mat_folder = os.path.join(base_path, f'validation_{epoch}', "match_mat")
            os.makedirs(gen_folder)
            os.makedirs(ref_folder)
            os.makedirs(match_mat_folder)
            # we only run the inference on one gpu
            print('validation at epoch {}.'.format(epoch))
            model.eval()
            test_loader = DataLoader(test_set, batch_size=conf.val_batch_size, shuffle=True, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn_data_list, worker_init_fn=utils.worker_init_fn)
            _, val_batch = next(enumerate(test_loader))
            val_batch = Batch.from_data_list(val_batch).to(conf.device)
            test_batch = val_batch.x
            val_all_poses = val_batch.all_poses.permute(1, 0, 2).cpu()
            if not MULTI_GPU:
                val_model = model
            else:
                val_model = model.module
            with torch.no_grad():
                pred_sel_list, pred_all_sel_tensor, mat_matrix, mat_matrix_mask = val_model.inference(val_batch, conf.sel_thre)
                utils.matrix_sns_plot(mat_matrix, mat_matrix_mask, gen_folder, 0)
            if conf.sel_first:
                gt_sel_tensor = val_all_poses[:, :, 0]
                gt_poses = val_all_poses[:, :, 1:].sum(dim=0)
            else:
                gt_sel_tensor = val_all_poses[:, :, -1]
                gt_poses = val_all_poses[:, :, :-1].sum(dim=0)
            test_batch_code = val_batch.batch
            # use conf.sel_thre to get the final selection
            gt_sel_tensor = gt_sel_tensor > conf.sel_thre # 4 * N
            # get gt_sel_list
            gt_sel_list = reform_gt_sel_list(test_batch_code, gt_sel_tensor)
            # obtain tp, fp, fn, tn
            eval_dict = eval_selection_batch(pred_sel_list, gt_sel_list)
            # calculate precision, recall, f1
            eval_result_dict = calc_prf(eval_dict)
            print(eval_result_dict)
            # # record in tensorboard
            # writer.add_scalars('Validation result:', eval_result_dict, epoch)
            # record in wandb
            wandb.log(eval_result_dict)
            # gt_poses = val_batch[DATASET_POSE_TYPE["qua"]].to(conf.device)[:, :, :-1].sum(dim=0) # become a 2D tensor, artifact dataset may be change?
            # we repeat gt_poses at dim 0 for pred_all_sel_tensor.size(0) times
            # unsequeeze dim 0
            gt_poses = gt_poses.unsqueeze(0)
            gt_poses = gt_poses.repeat(pred_all_sel_tensor.size(0), 1, 1)
            # We concatenate gt_poses and pred_all_sel_tensor along dim -1
            # print(gt_poses.size())
            # print(pred_all_sel_tensor.size())
            gen_sel_poses = torch.cat([gt_poses, pred_all_sel_tensor.float().unsqueeze(-1)], dim=-1)

            


            # Gen show
            gen_list = []
            for gen_idx in range(gen_sel_poses.size(0)):
                batch_assemblies = batch_assembly(conf, test_batch, gen_sel_poses[gen_idx].to(conf.device), test_batch_code, pose_type=conf.render_pose_type)
                gen_list.append(torch.stack(batch_assemblies['render_imgs'], dim=0))
            gen_multi = torch.stack(gen_list, dim=0)
            # Show in tensorboard
            for gen_batch_idx in range(gen_multi.size(1)):
                # writer.add_images('epoch_{}_gen'.format(epoch), gen_multi[:,gen_batch_idx,...], gen_batch_idx, dataformats='NHWC')
                # our image is NHWC, we save them in the folder, and upload to wandb
                for gen_idx in range(gen_multi.size(0)):
                    # save the numpy image
                    img_np = gen_multi[gen_idx, gen_batch_idx].cpu().numpy()
                    # print("img max: ", gen_multi[gen_idx, gen_batch_idx].max())
                    img_np = (img_np * 255).astype(np.uint8)
                    img_path = os.path.join(gen_folder, f'gen_{gen_batch_idx}_{gen_idx}.png')
                    imageio.imsave(img_path, img_np)
                    # upload to wandb
                    # wandb.log({f'gen_{gen_batch_idx}_{gen_idx}': wandb.Image(img_path)})


            # Reference show
            val_gen_multi = []
            for gen_idx in range(val_all_poses.size(0)):
                batch_assemblies = batch_assembly(conf, test_batch, val_all_poses[gen_idx].to(conf.device), test_batch_code, pose_type=conf.render_pose_type)
                val_gen_multi.append(torch.stack(batch_assemblies['render_imgs'], dim=0))
            val_gen_multi = torch.stack(val_gen_multi, dim=0)
            # Show in tensorboard
            for gen_batch_idx in range(val_gen_multi.size(1)):
                # writer.add_images('epoch_{}_ref'.format(epoch), val_gen_multi[:,gen_batch_idx,...], gen_batch_idx, dataformats='NHWC')
                # our image is NHWC, we save them in the folder, and upload to wandb
                for gen_idx in range(val_gen_multi.size(0)):
                    # save the numpy image
                    img_np = val_gen_multi[gen_idx, gen_batch_idx].cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_path = os.path.join(ref_folder, f'ref_{gen_batch_idx}_{gen_idx}.png')
                    imageio.imsave(img_path, img_np)
                    # upload to wandb
                    # wandb.log({f'ref_{gen_batch_idx}_{gen_idx}': wandb.Image(img_path)})
            # except Exception as e:
            #     print(e)
            #     print('validation failed.')

    # save the last model
    utils.save_network(model, os.path.join(model_save_path, 'model_final.pth'))
    print('model saved, training completed.')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # experimental setting
    parser.add_argument('--description', type=str, default='mix_part', help='experiment description')
    parser.add_argument('--base_dir', type=str, default='logs', help='model def file')
    parser.add_argument('--exp_name', type=str, default='my_exp_1', help='Please set your exp name, all the output will be saved in the folder with this exp_name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

    # datasets parameters:
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--data_dir', type=str, default='./data_output', help='data directory')
    parser.add_argument('--data_dir_test', type=str, default='./data_output', help='data directory')
    #parser.add_argument('--how_many_fusion', type=int, default=3, help='You need to indicate how many data used to form one part repository.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sel_first', action='store_true', default=False, help='selection at the first place')
    parser.add_argument('--data_norm', action='store_true', default=False, help='normalize the data')

    # model parameters:
    parser.add_argument('--model_type', type=str, default='mml_base', help='which model to use')
    parser.add_argument('--feat_len', type=int, default=256, help='feature length')
    parser.add_argument('--n_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--rela_layers', type=int, default=3, help='relation network layers')
    parser.add_argument('--sel_thre', type=float, default=0.5, help='selection threshold')

    # loss parameters:
    parser.add_argument('--mat_loss_weight', type=float, default=0.5, help='weight for matching loss')
    parser.add_argument('--sel_loss_weight', type=float, default=0.5, help='weight for selection loss')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epoch_save', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--val_every_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--print_loss', type=int, default=100)
    
    # rendering parameters
    parser.add_argument('--obj_png', type=str, default='png', help='Generation options: obj, png, both and no')
    parser.add_argument('--render_img_size', type=int, default=512)
    parser.add_argument('--render_pose_type', type=str, default='qua')

    # parser.add_argument('--pose_type', type=str, default='euler', help='poses type option')

    conf = parser.parse_args()

    # control randomness
    if conf.seed >= 0:
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)

    main(conf)

