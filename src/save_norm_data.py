import os
import os.path as osp
import numpy as np
from src.cmd_parser import parse_config
import torch
from src.data_parser import MotionNetData, GoalNetData

def split(x):
    return x[:args.state_dim], x[args.state_dim:]


if __name__ == '__main__':
    norm_data_folder = "~/SAMP_workspace/norm_data"

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        args, args_dict = parse_config()

    args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))

    args_dict['output_dir'] = osp.expandvars(args_dict.get('output_dir'))
    args_dict['save_norm_data'] = True


    norm_data_folder = osp.join(norm_data_folder, args.model_name)
    os.makedirs(norm_data_folder, exist_ok=True)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    if args_dict.get('float_dtype') == 'float64':
        dtype = torch.float64
    elif args_dict.get('float_dtype') == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(args_dict.get('float_dtype')))

    if args.model_name == 'MotionNet':
        data_set = MotionNetData(device=torch.device("cpu"), train_data=True, **args_dict)
        x1_mean, x2_mean = split(data_set.input_mean)
        x1_std, x2_std = split(data_set.input_std)
        x1_mean.tofile(osp.join(norm_data_folder, 'x1mean.bin'))
        x2_mean.tofile(osp.join(norm_data_folder, 'x2mean.bin'))
        x1_std.tofile(osp.join(norm_data_folder, 'x1std.bin'))
        x2_std.tofile(osp.join(norm_data_folder, 'x2std.bin'))

        data_set.output_mean.tofile(osp.join(norm_data_folder, 'Ymean.bin'))
        data_set.output_std.tofile(osp.join(norm_data_folder, 'Ystd.bin'))

    elif args.model_name == 'GoalNet':
        data_set = GoalNetData(device=torch.device("cpu"), train_data=True, **args_dict)
        data_set.input_mean.tofile(osp.join(norm_data_folder, 'Xmean.bin'))
        data_set.input_std.tofile(osp.join(norm_data_folder, 'Xstd.bin'))
        data_set.output_mean.tofile(osp.join(norm_data_folder, 'Ymean.bin'))
        data_set.output_std.tofile(osp.join(norm_data_folder, 'Ystd.bin'))


