import os
import os.path as osp
import torch
from torch.nn import functional as F
from src.data_parser import GoalNetData
from torch.utils.data import DataLoader
from src import SAMP_models as models
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from src.cmd_parser import parse_config
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for batch_idx, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        cond = data['x'].to(device)
        y = data['y'].to(device)

        y_hat, mu, logvar = model(y, cond)

        recon_loss = F.mse_loss(y_hat, y, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + args.kl_w * kld

        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kld_loss += kld.item()
        total_loss += loss.item()

    total_recon_loss /= float(n_batches_train)
    total_kld_loss /= float(n_batches_train)
    total_loss /= float(n_batches_train)
    writer.add_scalar('total/train_total_loss', total_loss, epoch)
    print('====> Total_train_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(total_loss, total_recon_loss,
                                                                                        total_kld_loss))
    return total_loss


def test():
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for batch_idx, data in enumerate(test_data_loader):
        cond = data['x'].to(device)
        y = data['y'].to(device)

        y_hat, mu, logvar = model(y, cond)

        recon_loss = F.mse_loss(y_hat, y, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + args.kl_w * kld

        total_recon_loss += recon_loss.item()
        total_kld_loss += kld.item()
        total_loss += loss.item()

    total_recon_loss /= float(n_batches_test)
    total_kld_loss /= float(n_batches_test)
    total_loss /= float(n_batches_test)
    writer.add_scalar('total/total_test_loss', total_loss, epoch)
    print('====> Total_test_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(total_loss, total_recon_loss,
                                                                                       total_kld_loss))
    return total_loss


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        args, args_dict = parse_config()

    args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))
    args_dict['output_dir'] = osp.expandvars(args_dict.get('output_dir'))
    data_dir = args_dict.get('data_dir')
    output_dir = args_dict.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    conf_fn = osp.join(output_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        # delete all None arguments
        conf_data = vars(args)
        keys = [x for x in conf_data.keys() if conf_data[x] is None]
        for key in keys:
            del (conf_data[key])
        yaml.dump(conf_data, conf_file)

    rng = np.random.RandomState(23456)
    args_dict['rng'] = rng

    device = torch.device("cuda" if args.use_cuda else "cpu")
    dtype = torch.float32

    model = models.load_model(**args_dict).to(device)
    n_params = models.count_parameters(model)
    print('Num of trainable param = {:.2f}M, exactly = {}'.format(n_params / (10 ** 6), n_params))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 1
    if args.load_checkpoint > 0:
        print('loading stats of epoch {}'.format(args.load_checkpoint))
        checkpoint = torch.load(
            osp.join(osp.join(args_dict['output_dir'], 'checkpoints'), 'epoch_{:04d}.pt'.format(args.load_checkpoint)))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] + 1

    checkpoints_dir = osp.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = osp.join(output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_data_set = GoalNetData(device=torch.device("cpu"), train_data=True, **args_dict)
    train_data_loader = DataLoader(train_data_set, batch_size=args_dict.get('batch_size'),
                                   num_workers=args_dict.get('num_workers'), shuffle=args.shuffle,
                                   drop_last=True)
    train_set_size = len(train_data_set)
    n_batches_train = train_set_size // args.batch_size
    print('No of training example: {}, No of batches {}'.format(train_set_size, n_batches_train))
    if args.test:
        test_data_set = GoalNetData(device=torch.device("cpu"), train_data=False, **args_dict)
        test_data_loader = DataLoader(test_data_set, batch_size=args_dict.get('batch_size'),
                                      num_workers=args_dict.get('num_workers'), shuffle=False,
                                      drop_last=True)
        test_set_size = len(test_data_set)
        n_batches_test = test_set_size // args.batch_size
        print('Number of testing example: {}'.format(test_set_size))

    for epoch in range(epoch, args.epochs + 1):
        print('Training epoch {}'.format(epoch))

        start = time.time()
        total_train_loss = train()
        training_time = time.time() - start

        if args.test:
            print('Testing epoch {}'.format(epoch))
            start = time.time()
            total_test_loss = test()
            testing_time = time.time() - start

        writer.add_scalar('lr', get_lr(optimizer), epoch)

        print('training_time = {:.4f}'.format(training_time))
        writer.add_scalar('time/training_time', training_time, epoch)
        if args.test:
            print('test_time = {:.4f}'.format(testing_time))
            writer.add_scalar('time/testing_time', testing_time, epoch)

        if args.save_checkpoints and epoch % args.log_interval == 0:
            data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_train_loss': total_train_loss,
            }
            if args.test:
                data['total_test_loss'] = total_test_loss
            torch.save(data, osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(epoch)))
