# Original code: https://github.com/taesupkim/vta/blob/master/utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def highlite_boundary(input_data):
    input_data[0, :, 0] = 1.0
    input_data[0, :, 1] = 0.0
    input_data[0, :, 2] = 0.0
    input_data[-1, :, 0] = 1.0
    input_data[-1, :, 1] = 0.0
    input_data[-1, :, 2] = 0.0

    input_data[:, 0, 0] = 1.0
    input_data[:, 0, 1] = 0.0
    input_data[:, 0, 2] = 0.0
    input_data[:, -1, 0] = 1.0
    input_data[:, -1, 1] = 0.0
    input_data[:, -1, 2] = 0.0
    return input_data


def tensor2numpy_img(input_tensor):
    return input_tensor.permute(1, 2, 0).data.cpu().numpy()


def plot_rec(init_data_list, org_data_list, rec_data_list, mask_data_list, prior_mask_list, post_mask_list):
    # get size
    batch_size, init_size, rgb_size,  row_size, col_size = init_data_list.size()
    seq_size = org_data_list.size(1)

    # init pad
    row_pad = np.zeros([1, (col_size + 2) * (seq_size + init_size), 3])
    col_pad = np.zeros([row_size, 1, 3])
    red_block = np.ones([row_size, col_size, 3])
    red_block[:, :, 1:] = 0.0
    blue_block = np.ones([row_size, col_size, 3])
    blue_block[:, :, :2] = 0.0

    # init out image
    output_img = []
    output_mask = []
    for img_idx in range(batch_size):
        org_img_list = []
        rec_img_list = []
        p_mask_list = []
        q_mask_list = []

        # for init image
        for i_idx in range(init_size):
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            org_img_list.append(highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx])))
            rec_img_list.append(highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx])))
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)

            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)
            p_mask_list.append(blue_block)
            q_mask_list.append(red_block)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

        # for roll out sequence
        for i_idx in range(seq_size):
            # padding
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

            # data
            if mask_data_list[img_idx, i_idx]:
                org_img_list.append(highlite_boundary(tensor2numpy_img(org_data_list[img_idx, i_idx])))
                rec_img_list.append(highlite_boundary(tensor2numpy_img(rec_data_list[img_idx, i_idx])))
            else:
                org_img_list.append(tensor2numpy_img(org_data_list[img_idx, i_idx]))
                rec_img_list.append(tensor2numpy_img(rec_data_list[img_idx, i_idx]))

            # mask
            p_mask_list.append(blue_block * prior_mask_list[img_idx, i_idx].item())
            q_mask_list.append(red_block * post_mask_list[img_idx, i_idx].item())

            # padding
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

        # stack
        org_img_list = np.concatenate(org_img_list, 1)
        rec_img_list = np.concatenate(rec_img_list, 1)
        p_mask_list = np.concatenate(p_mask_list, 1)
        q_mask_list = np.concatenate(q_mask_list, 1)
        output_img.append(np.concatenate([row_pad, org_img_list, row_pad, row_pad, rec_img_list, row_pad], 0))
        output_mask.append(np.concatenate([row_pad, p_mask_list, row_pad, row_pad, q_mask_list, row_pad], 0))
    output_img = np.clip(np.concatenate(output_img, 0), 0.0, 1.0)
    output_mask = np.clip(np.concatenate(output_mask, 0), 0.0, 1.0)
    return output_img, output_mask


def plot_gen(init_data_list, gen_data_list, mask_data_list=None):
    # get size
    batch_size, init_size, rgb_size, row_size, col_size = init_data_list.size()
    seq_size = gen_data_list.size(1)

    # init pad
    row_pad = np.zeros([1, (col_size + 2) * (seq_size + init_size), 3])
    col_pad = np.zeros([row_size, 1, 3])

    # init out image
    output_img = []
    for img_idx in range(batch_size):
        gen_img_list = []

        # for init image
        for i_idx in range(init_size):
            gen_img_list.append(col_pad)
            gen_img_list.append(highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx])))
            gen_img_list.append(col_pad)

        # for roll out sequence
        for i_idx in range(seq_size):
            # padding
            gen_img_list.append(col_pad)

            # data
            if mask_data_list is not None and mask_data_list[img_idx, i_idx]:
                gen_img_list.append(highlite_boundary(tensor2numpy_img(gen_data_list[img_idx, i_idx])))
            else:
                gen_img_list.append(tensor2numpy_img(gen_data_list[img_idx, i_idx]))

            # padding
            gen_img_list.append(col_pad)

        # stack
        gen_img_list = np.concatenate(gen_img_list, 1)
        output_img.append(np.concatenate([row_pad, gen_img_list, row_pad], 0))
    output_img = np.clip(np.concatenate(output_img, 0), 0.0, 1.0)
    return output_img


def log_train(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    train_obs_cost = results['obs_cost'].mean()
    train_kl_abs_cost = results['kl_abs_state'].mean()
    train_kl_obs_cost = results['kl_obs_state'].mean()
    train_kl_mask_cost = results['kl_mask'].mean()

    # log
    writer.add_scalar('train/full_cost', train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('train/obs_cost', train_obs_cost, global_step=b_idx)
    writer.add_scalar('train/kl_full_cost', train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('train/kl_abs_cost', train_kl_abs_cost, global_step=b_idx)
    writer.add_scalar('train/kl_obs_cost', train_kl_obs_cost, global_step=b_idx)
    writer.add_scalar('train/kl_mask_cost', train_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('train/q_ent', results['p_ent'].mean(), global_step=b_idx)
    writer.add_scalar('train/p_ent', results['q_ent'].mean(), global_step=b_idx)
    writer.add_scalar('train/read_ratio', results['mask_data'].sum(1).mean(), global_step=b_idx)
    writer.add_scalar('train/beta', results['beta'], global_step=b_idx)

    log_str = '[%08d] train=elbo:%7.3f, obs_nll:%7.3f, ' \
              'kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, ' \
              'num_reads:%3.1f, beta: %3.3f, ' \
              'p_ent: %3.2f, q_ent: %3.2f'
    log_data = [b_idx,
                - (train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost),
                train_obs_cost,
                train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
                train_kl_abs_cost,
                train_kl_obs_cost,
                train_kl_mask_cost,
                results['mask_data'].sum(1).mean(),
                results['beta'],
                results['p_ent'].mean(),
                results['q_ent'].mean()]
    return log_str, log_data


def log_test(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    test_obs_cost = results['obs_cost'].mean()
    test_kl_abs_cost = results['kl_abs_state'].mean()
    test_kl_obs_cost = results['kl_obs_state'].mean()
    test_kl_mask_cost = results['kl_mask'].mean()

    writer.add_scalar('valid/full_cost', test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('valid/obs_cost', test_obs_cost, global_step=b_idx)
    writer.add_scalar('valid/kl_full_cost', test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('valid/kl_abs_cost', test_kl_abs_cost, global_step=b_idx)
    writer.add_scalar('valid/kl_obs_cost', test_kl_obs_cost, b_idx)
    writer.add_scalar('valid/kl_mask_cost', test_kl_mask_cost, global_step=b_idx)
    writer.add_scalar('valid/read_ratio', results['mask_data'].sum(1).mean(), global_step=b_idx)

    log_str = '[%08d] valid=elbo:%7.3f, obs_nll:%7.3f, ' \
              'kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, ' \
              'num_reads:%3.1f'
    log_data = [b_idx,
                - (test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost),
                test_obs_cost,
                test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
                test_kl_abs_cost,
                test_kl_obs_cost,
                test_kl_mask_cost,
                results['mask_data'].sum(1).mean()]
    return log_str, log_data


def concat(*data_list):
    return torch.cat(data_list, 1)


def gumbel_sampling(log_alpha, temp, margin=1e-4):
    noise = log_alpha.new_empty(log_alpha.size()).uniform_(margin, 1 - margin)
    gumbel_sample = - torch.log(- torch.log(noise))
    return torch.div(log_alpha + gumbel_sample, temp)


def log_density_concrete(log_alpha, log_sample, temp):
    exp_term = log_alpha - temp * log_sample
    log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
    return log_prob
