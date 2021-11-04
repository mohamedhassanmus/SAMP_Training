import numpy as np
import torch
from src import misc_utils


def normalize_vector(v):
    # norm_v = v.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm_v


def split_pose_feat(data):
    pos = data[:, :, :3].clone()
    forward = data[:, :, 3:6].clone()
    up = data[:, :, 6:9].clone()
    velocity = data[:, :, 9:12].clone()
    return pos, forward, up, velocity


def project_root(pose_mat_world):
    bs = pose_mat_world.shape[0]
    pose_pos_world = pose_mat_world[:, :, :3, 3]

    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, 1, 1)), device=pose_mat_world.device, dtype=torch.float32)
    up_root = torch.tensor(np.tile(np.array([[0, 1, 0]]), (bs, 1)), device=pose_mat_world.device, dtype=torch.float32)

    pelvis_pos = pose_pos_world[:, 0, :3]
    pos_root = project_pos_on_plane(pelvis_pos)

    leftHip_pos = pose_pos_world[:, 1, :3]
    rightHip_pos = pose_pos_world[:, 5, :3]
    leftShoulder_pos = pose_pos_world[:, 13, :3]
    rightShoulder_pos = pose_pos_world[:, 19, :3]

    v1 = normalize_vector(project_pos_on_plane(rightHip_pos - leftHip_pos))
    v2 = normalize_vector(project_pos_on_plane(rightShoulder_pos - leftShoulder_pos))
    v = normalize_vector(v1 + v2)

    forward_root = normalize_vector(project_pos_on_plane(torch.cross(v, up_root)))

    right_root = torch.cross(up_root, forward_root)
    right_root = normalize_vector(right_root)

    mat_world_root = torch.stack(
        [right_root, up_root, forward_root, pos_root], dim=-1)
    mat_world_root = torch.cat([mat_world_root, lastrow], dim=1)
    return mat_world_root


def create_rootMat(input_FoR):
    bs = input_FoR.shape[0]
    v = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, 1, 1)), device=input_FoR.device, dtype=input_FoR.dtype)
    pos_root = input_FoR[:, :3]
    right_root = input_FoR[:, 3:6]
    up_root = input_FoR[:, 6:9]
    forward_root = input_FoR[:, 9:12]

    mat_world_root = torch.stack(
        [right_root, up_root, forward_root, pos_root], dim=-1)
    mat_world_root = torch.cat([mat_world_root, v], dim=1)
    return mat_world_root


def project_pos_on_plane(pos):
    pos_xz = pos.clone()
    pos_xz[:, 1] = 0.0
    return pos_xz


def transform_mat(mat, root_mat_in, root_mat_out_inv):
    mat_world = torch.matmul(root_mat_in[:, None], mat)
    # Convert to transforms wrt frame i+1
    mat_transformed = torch.matmul(root_mat_out_inv, mat_world)
    return mat_transformed


###############################################################################################################
#################### TRAJECTRORY
###############################################################################################################

def traj_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    traj = torch.zeros((bs, window_size, 4 + num_actions), device=mat.device, dtype=mat.dtype)
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    traj[:, :, 4:] = style
    return traj.reshape(bs, window_size * (4 + num_actions))


def traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)), device=data.device, dtype=torch.float32)
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]
    style = data.reshape(bs, window_size, -1)[:, :, 4:]

    # Convert to 3D data
    pos = torch.zeros((pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32)
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros((pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32)
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)), device=data.device, dtype=torch.float32)
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack(
        [right, up, forward, pos], dim=-1)
    mat = torch.cat(
        [mat, lastrow], dim=2)
    return mat, style


def transform_traj(traj_vec, root_mat_in, root_mat_out_inv, num_actions=5):
    traj_mat, traj_style = traj_vec_2_mat(traj_vec)
    traj_transformed = transform_mat(traj_mat, root_mat_in, root_mat_out_inv)
    traj_transformed = traj_mat_2_vec(traj_transformed, traj_style, num_actions=num_actions)
    return traj_transformed


###############################################################################################################
#################### GOAL Related
###############################################################################################################
def goal_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)), device=data.device, dtype=torch.float32)
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :6]
    style = data.reshape(bs, window_size, -1)[:, :, 6:]

    pos = pos_dir[:, :, :3]
    forward = pos_dir[:, :, 3:]
    forward = normalize_vector(forward)
    up = torch.tensor(np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)), device=data.device, dtype=torch.float32)
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack(
        [right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, style


def goal_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    traj = torch.zeros((bs, window_size, 6 + num_actions), dtype=torch.float32, device=mat.device)
    traj[:, :, :3] = mat[..., :3, 3]
    traj[:, :, 3:6] = mat[..., :3, 2]
    traj[:, :, 6:] = style
    return traj.reshape(bs, window_size * (6 + num_actions))


def transform_goal(goal_vec, root_mat_in, root_mat_out_inv, num_actions=5):
    goal_mat, goal_style = goal_vec_2_mat(goal_vec)
    goal_transformed = transform_mat(goal_mat, root_mat_in, root_mat_out_inv)
    goal_transformed = goal_mat_2_vec(goal_transformed, goal_style, num_actions=num_actions)
    return goal_transformed


###############################################################################################################
#################### Pose
###############################################################################################################


def pose_vec_2_mat(pose):
    bs = pose.shape[0]
    nj = pose.shape[1]
    lastrow = torch.tensor(
        np.tile(np.array([[[[0, 0, 0, 1]]]]), (bs, nj, 1, 1)), device=pose.device, dtype=pose.dtype)
    pos, forward, up, velocity = split_pose_feat(pose)
    # Get input joint transforms wrt frame i
    forward = normalize_vector(forward)
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack(
        [right, up, forward, pos], dim=-1)
    mat = torch.cat(
        [mat, lastrow], dim=2)
    return mat, velocity


def pose_mat_2_vec(mat, velocity):
    # Get velocities and convert them to velocities wrt frame i+1
    pose_transformed = [
        mat[..., :3, 3],
        mat[..., :3, 2],
        mat[..., :3, 1],
        velocity[..., :3]]
    pose_transformed = torch.cat(pose_transformed, dim=-1)
    return pose_transformed


def transform_pose(pose_vec_in, pose_vec_out, prev_root_transform=None, NJ=22, pose_feat_dim=12):
    pose_vec_in = pose_vec_in.reshape(-1, NJ, pose_feat_dim)
    pose_vec_out = pose_vec_out.reshape(-1, NJ, pose_feat_dim)
    bs = pose_vec_in.shape[0]
    nj = pose_vec_in.shape[1]

    pose_in_mat, _ = pose_vec_2_mat(pose_vec_in)
    # # Get root transforms at frame i
    # root_mat_in = create_rootMat(FoR_in)
    root_mat_in = project_root(pose_in_mat)
    # Accumulate root transforms
    if prev_root_transform is not None:
        root_mat_in = torch.matmul(prev_root_transform, root_mat_in)

    pose_mat_out, velocity_out = pose_vec_2_mat(pose_vec_out)

    pose_mat_world_out = torch.matmul(root_mat_in[:, None], pose_mat_out)
    # Get root transform for frame i+1
    root_mat_out = project_root(pose_mat_world_out)

    root_mat_out_inv = torch.inverse(root_mat_out[:, None])
    # Convert to transforms wrt frame i+1
    pose_mat_out_transformed = torch.matmul(root_mat_out_inv, pose_mat_world_out)

    zeros = torch.zeros((velocity_out.shape[0], nj, 1), dtype=torch.float32, device=pose_vec_in.device)
    velocity_out = torch.cat([velocity_out, zeros], dim=-1)
    velocity_out_world = torch.matmul(root_mat_in[:, None], velocity_out[..., None])[..., 0]
    velocity_out_transformed = torch.matmul(root_mat_out_inv, velocity_out_world[..., None])[..., 0]
    pose_vec_out_transformed = pose_mat_2_vec(pose_mat_out_transformed, velocity_out_transformed)

    return pose_vec_out_transformed.reshape(bs, NJ * pose_feat_dim), root_mat_in, root_mat_out_inv


###############################################################################################################
#################### Inv Pose
###############################################################################################################

def get_last_traj_world(traj_vec, root_mat_in):
    traj_mat, _ = traj_vec_2_mat(traj_vec)
    traj_mat_world = torch.matmul(root_mat_in[:, None], traj_mat)
    return traj_mat_world[:, -1, :, :]


def transform_pose_inv(pose_inv_out, root_mat_in, traj_in, traj_out, NJ=22):
    FoR_in = get_last_traj_world(traj_in, root_mat_in)
    FoR_out_inv = torch.inverse(get_last_traj_world(traj_out, root_mat_in))

    pose_inv_out = pose_inv_out.reshape(-1, NJ, 3)
    bs = pose_inv_out.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([1]), (bs, NJ, 1)), device=pose_inv_out.device, dtype=pose_inv_out.dtype)
    pose_inv_out = torch.cat(
        [pose_inv_out, lastrow], dim=2)
    # zeros = torch.zeros((pose_inv_out.shape[0], nj, 1), dtype=torch.float32, device=pose_vec_in.device)
    # velocity_out = torch.cat([velocity_out, zeros], dim=-1)
    pose_inv_out_world = torch.matmul(FoR_in[:, None], pose_inv_out.unsqueeze(-1))
    pose_inv_out_transformed = torch.matmul(FoR_out_inv[:, None], pose_inv_out_world)
    pose_inv_out_transformed = pose_inv_out_transformed[:, :, :3, 0].reshape(bs, NJ * 3)

    return pose_inv_out_transformed


###############################################################################################################
#################### Inv Traj
###############################################################################################################
def inv_traj_mat_2_vec(mat, window_size=13):
    bs = mat.shape[0]
    traj = torch.zeros((bs, window_size, 4), device=mat.device, dtype=mat.dtype)
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    return traj.reshape(bs, window_size * 4)


def inv_traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)), device=data.device, dtype=torch.float32)
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]

    # Convert to 3D data
    pos = torch.zeros((pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32)
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros((pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32)
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)), device=data.device, dtype=torch.float32)
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    # check normalization above
    mat = torch.stack(
        [right, up, forward, pos], dim=-1)
    mat = torch.cat(
        [mat, lastrow], dim=2)
    return mat


def get_pivot_goal_world(goal_vec, root_mat_in):
    goal_mat, _ = goal_vec_2_mat(goal_vec)
    goal_mat_world = torch.matmul(root_mat_in[:, None], goal_mat)
    return goal_mat_world[:, 6, :, :]


def transform_traj_inv(traj_inv_out, root_mat_in, goal_in, goal_out):
    FoR_in = get_pivot_goal_world(goal_in, root_mat_in)
    FoR_out_inv = torch.inverse(get_pivot_goal_world(goal_out, root_mat_in)[:, None])

    traj_inv_out_mat = inv_traj_vec_2_mat(traj_inv_out)

    traj_inv_transformed = transform_mat(traj_inv_out_mat, FoR_in, FoR_out_inv)
    traj_inv_transformed = inv_traj_mat_2_vec(traj_inv_transformed)

    return traj_inv_transformed


###############################################################################################################
####################  Main
###############################################################################################################


def transform_data(inputs, outputs, prev_root_transform=None, state_dim=524, num_actions=5, **kwargs):
    pose_in, pose_inv_in, traj_in, contact_in, traj_inv_in, goal_in, interaction_in = misc_utils.split_features(
        inputs)
    pose_out, pose_inv_out, traj_out, contact_out, traj_inv_out, goal_out, interaction_out = misc_utils.split_features(
        outputs)
    # Transform egocentric features
    pose_out_transformed, root_mat_in, root_mat_out_inv = transform_pose(pose_in, pose_out, prev_root_transform)
    traj_out_trasformed = transform_traj(traj_out, root_mat_in, root_mat_out_inv, num_actions=num_actions)
    goal_out_transformed = transform_goal(goal_out, root_mat_in, root_mat_out_inv, num_actions=num_actions)
    # inv
    pose_inv_out_transformed = transform_pose_inv(pose_inv_out, root_mat_in, traj_in, traj_out)
    traj_inv_out_transformed = transform_traj_inv(traj_inv_out, root_mat_in, goal_in, goal_out)

    outputs_transformed = torch.cat(
        [pose_out_transformed, pose_inv_out_transformed, traj_out_trasformed, contact_out, traj_inv_out_transformed,
         goal_out_transformed], dim=-1)
    if inputs.shape[0] > 1:
        error = ((outputs_transformed[:-1] - inputs[1:, :state_dim]) ** 2).max()
    else:
        error = None

    return outputs_transformed, error
