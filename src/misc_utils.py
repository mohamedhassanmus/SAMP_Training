import os.path as osp
from src import SAMP_models as models
from src import sched_sampl_utils
import torch
import numpy as np


def Normalize(X, mean, std):
    return (X - mean) / std


def UnNormalize(X, mean, std):
    return std * X + mean


def transform_output(x1, x2, y, input_mean, input_std, output_mean, output_std, state_dim=None, interaction_dim=None,
                     num_actions=None, **kwargs):
    if x2 is not None:
        inputs = UnNormalize(
            torch.cat((x1, x2), dim=-1).reshape(-1, state_dim + interaction_dim),
            input_mean, input_std)
    else:
        inputs = UnNormalize(x1, input_mean, input_std)
    outputs = UnNormalize(y.reshape(-1, state_dim), output_mean, output_std)
    outputs_transformed, err = sched_sampl_utils.transform_data(inputs, outputs, state_dim=state_dim,
                                                                num_actions=num_actions)
    return Normalize(outputs_transformed, input_mean[:state_dim], input_std[:state_dim])


def split_features(data, start_pose=0, start_pose_inv=264, start_trajectory=330, start_contact=447, start_traj_inv=452,
                   start_goal=504, start_interaction=647):
    pose = data[:, start_pose: start_pose_inv]
    pose_inv = data[:, start_pose_inv: start_trajectory]
    traj = data[:, start_trajectory: start_contact]
    contact = data[:, start_contact:start_traj_inv]

    traj_inv = data[:, start_traj_inv:start_goal]
    goal = data[:, start_goal: start_interaction]
    if data.shape[1] > start_interaction:
        interaction = data[:, start_interaction:]
    else:
        interaction = None
    return pose, pose_inv, traj, contact, traj_inv, goal, interaction


def load_model_checkpoint(model_name, load_checkpoint, use_cuda, checkpoints_dir, checkpoint_path=None, **kwargs):
    model = models.load_model(model_name, use_cuda=use_cuda, **kwargs)
    if checkpoint_path is not None:
        print('loading stats of epoch from {}'.format(checkpoint_path))

    elif load_checkpoint > 0:
        checkpoint_path = osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(load_checkpoint))
        print('loading stats of epoch {} from {}'.format(load_checkpoint, checkpoints_dir))
    if checkpoint_path is not None:
        if not use_cuda:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('****** No checkpoint found ************')
        # sys.exit(0)
    return model
