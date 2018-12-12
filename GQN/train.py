import argparse
import datetime
import math
import os
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from gqn_dataset import GQNDataset, Scene, transform_viewpoint, sample_batch
from scheduler import AnnealingStepLR
from model import GQN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--dataset', type=str, default='Shepard-Metzler', help='dataset (dafault: Shepard-Mtzler)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    dir_name = str(datetime.datetime.now())
    log_dir = os.path.join(args.root_log_dir, dir_name)
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'models'))
    os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    # Dataset
    train_dataset = GQNDataset(root_dir=train_data_dir, target_transform=transform_viewpoint)
    test_dataset = GQNDataset(root_dir=test_data_dir, target_transform=transform_viewpoint)
    D = args.dataset

    # Pixel standard-deviation
    sigma_i, sigma_f = 2.0, 0.7
    sigma = sigma_i

    # Number of scenes over which each weight update is computed
    B = args.batch_size

    # Maximum number of training steps
    S_max = args.gradient_steps

    # Define model
    model = GQN().to(device)
    if args.data_parallel:
        model = nn.DataParallel(model, device_ids=[0,1,2])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True, **kwargs)

    train_iter = iter(train_loader)
    x_data_test, v_data_test = next(iter(test_loader))

    # Training Iterations
    for t in tqdm(range(S_max)):
        try:
            x_data, v_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_data, v_data = next(train_iter)

        x_data = x_data.to(device)
        v_data = v_data.to(device)
        x, v, x_q, v_q = sample_batch(x_data, v_data, D)
        elbo = model(x, v, v_q, x_q, sigma)
        
        # Logs
        writer.add_scalar('train_loss', -elbo.mean(), t)
             
        with torch.no_grad():
            # Write logs to TensorBoard
            if t % log_interval_num == 0:
                x_data_test = x_data_test.to(device)
                v_data_test = v_data_test.to(device)

                x_test, v_test, x_q_test, v_q_test = sample_batch(x_data_test, v_data_test, D, M=3, seed=0)
                elbo_test = model(x_test, v_test, v_q_test, x_q_test, sigma)
                
                if args.data_parallel:
                    kl_test = model.module.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                    x_q_rec_test = model.module.reconstruct(x_test, v_test, v_q_test, x_q_test)
                    x_q_hat_test = model.module.generate(x_test, v_test, v_q_test)
                else:
                    kl_test = model.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                    x_q_rec_test = model.reconstruct(x_test, v_test, v_q_test, x_q_test)
                    x_q_hat_test = model.generate(x_test, v_test, v_q_test)

                writer.add_scalar('test_loss', -elbo_test.mean(), t)
                writer.add_scalar('test_kl', kl_test.mean(), t)
                writer.add_image('test_ground_truth', make_grid(x_q_test, 6, pad_value=1), t)
                writer.add_image('test_reconstruction', make_grid(x_q_rec_test, 6, pad_value=1), t)
                writer.add_image('test_generation', make_grid(x_q_hat_test, 6, pad_value=1), t)

            if t % save_interval_num == 0:
                torch.save(model.state_dict(), log_dir + "/models/model-{}.pt".format(t))

        # Compute empirical ELBO gradients
        (-elbo.mean()).backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update optimizer state
        scheduler.step()

        # Pixel-variance annealing
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - t/(2e5)), sigma_f)
        
    torch.save(model.state_dict(), log_dir + "/models/model-final.pt")  
    writer.close()
