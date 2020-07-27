import numpy as np
import torch
from mocap.math.mirror_h36m import mirror_p3d
from mocap.math.quaternion import qrot, qmul, q_inv_batch_of_sequences


# DEFAULT h36m
# parent = [
#     -1,    # 0
#     0,     # 1
#     1,     # 2
#     2,     # 3
#     3,     # 4
#     4,     # 5
#     0,     # 6
#     6,     # 7
#     7,     # 8
#     8,     # 9
#     9,     # 10
#     0,     # 11
#     11,    # 12
#     12,    # 13
#     13,    # 14
#     14,    # 15
#     12,    # 16
#     16,    # 17
#     17,    # 18
#     18,    # 19
#     19,    # 20
#     20,    # 21
#     19,    # 22
#     22,    # 23
#     12,    # 24
#     24,    # 25
#     25,    # 26
#     26,    # 27
#     27,    # 28
#     28,    # 29
#     27,    # 30
#     30]    # 31

# simplified skeleton

#                      (18)
#                        |
# (13)-(12)-(11) --  (10, 9,14) -- (15)-(16)-(17)
#                        |
#                       ( 8)
#                        |
#             ( 1) -- ( 7, 0) -- ( 4)
#              |                   |
#             ( 2)               ( 5)
#              |                   |
#             ( 3)               ( 6)

map_large2simplified = np.array([
    0,  # 0
    1,  # 1
    2,  # 2
    3,  # 3
    6,  # 4
    7,  # 5
    8,  # 6
    11,  # 7
    12,  # 8
    13,  # 9
    24,  # 10
    25,  # 11
    26,  # 12
    27,  # 13
    16,  # 14
    17,  # 15
    18,  # 16
    19,  # 17
    14,  # 18
]).astype('int64')

parent_simplified = np.array([
    -1,  # 0
     0,  # 1
     1,  # 2
     2,  # 3
     0,  # 4
     4,  # 5
     5,  # 6
     0,  # 7
     7,  # 8
     8,  # 9
     9,  # 10
     10, # 11
     11, # 12
     12, # 13
     9,  # 14
     14, # 15
     15, # 16
     16, # 17
     9,  # 18
]).astype('int64')

# -- hardcoded data --
parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                   17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1
parent = parent.astype('int64')
bone_lengths = np.array(
    [0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000]
) / 1000.
bone_lengths = bone_lengths.reshape((-1, 3)).astype('float32')
assert len(parent) == len(bone_lengths)
n_joints = len(parent)


def calculate_chain(parent, n_joints):
    chain_per_joint = []
    for jid in range(n_joints):
        current = parent[jid]
        chain = [current]
        while current > -1:
            current = parent[current]
            chain.append(current)
        chain.reverse()
        chain.pop(0)
        chain_per_joint.append(chain)
    return chain_per_joint


chain_per_joint = calculate_chain(parent, n_joints=n_joints)


def quaternion_fk(rotations):
    """
    :param rotations: {n_batch x n_frames x J x 4}
    :return:
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    offsets = torch.from_numpy(bone_lengths).to(device)
    
    parent = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
               16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30])
    parent = torch.from_numpy(parent).to(device)
    is_numpy = False
    if isinstance(rotations, np.ndarray):
        is_numpy = True
        rotations = torch.from_numpy(rotations).to(device)

    n_batch = rotations.size(0)
    n_frames = rotations.size(1)
    if len(rotations.size()) == 3:
        rotations = rotations.reshape((n_batch, n_frames, -1, 4))
    rotations = q_inv_batch_of_sequences(rotations)

    positions_world = []
    rotations_world = []
    root_positions = torch.zeros((n_batch, n_frames, 3), device=device)
    expanded_offsets = offsets.expand(rotations.shape[0], rotations.shape[1],
                                      offsets.shape[0], offsets.shape[1])

    for i in range(offsets.shape[0]):
        if parent[i] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parent[i]], expanded_offsets[:, :, i]) \
                                   + positions_world[parent[i]])
            rotations_world.append(qmul(rotations_world[parent[i]], rotations[:, :, i]))

    result = torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    if is_numpy:
        result = result.cpu().numpy()

        n_batch = result.shape[0]
        n_frames = result.shape[1]
        result = result.reshape((n_batch * n_frames, 32, 3))
        result[:, :, (0, 1, 2)] = result[:, :, (0, 2, 1)]
        result = mirror_p3d(result)
        result = result.reshape((n_batch, n_frames, 32, 3))
    return result



def euler_fk_with_parameters(angles, n_joints, chain_per_joint, bone_lengths):
    """
    :param [n_batch x 3 * n_joints]
    """
    n_batch = np.shape(angles)[0]
    angles = np.reshape(angles, (-1, 3))
    Rs = batch_rot3d(angles)
    Rs = np.reshape(Rs, (n_batch, n_joints, 3, 3))
    Pts3d = []
    for jid in range(n_joints):
        bone = bone_lengths[jid]
        bone = np.tile(bone, n_batch)
        bone = np.reshape(bone, (n_batch, 3))
        bone = np.expand_dims(bone, axis=1)
        chain = chain_per_joint[jid]

        p_xyz = np.zeros((n_batch, 1, 3), dtype=np.float32)
        p_R = np.tile(np.eye(3, dtype=np.float32), (n_batch, 1))
        p_R = np.reshape(p_R, (n_batch, 3, 3))

        for jid2 in chain:
            cur_R = Rs[:, jid2]
            cur_bone = bone_lengths[jid2]
            cur_bone = np.tile(cur_bone, n_batch)
            cur_bone = np.reshape(cur_bone, (n_batch, 3))
            cur_bone = np.expand_dims(cur_bone, axis=1)
            p_xyz = p_xyz + np.matmul(cur_bone, p_R)
            p_R = np.matmul(cur_R, p_R)
        xyz = np.matmul(bone, p_R) + p_xyz
        xyz = np.reshape(xyz, (n_batch, 3))
        Pts3d.append(xyz)

    Pts3d = np.stack(Pts3d, axis=1).astype(np.float32)

    Pts3d[:, :, (0, 1, 2)] = Pts3d[:, :, (0, 2, 1)]

    # Pts3d = mirror_p3d(Pts3d)

    return Pts3d




def euler_fk(angles):
    """
    :param [n_batch x 3 * n_joints]
    """
    n_batch = np.shape(angles)[0]
    angles = np.reshape(angles, (-1, 3))
    Rs = batch_rot3d(angles)
    Rs = np.reshape(Rs, (n_batch, n_joints, 3, 3))
    Pts3d = []
    for jid in range(n_joints):
        bone = bone_lengths[jid]
        bone = np.tile(bone, n_batch)
        bone = np.reshape(bone, (n_batch, 3))
        bone = np.expand_dims(bone, axis=1)
        chain = chain_per_joint[jid]

        p_xyz = np.zeros((n_batch, 1, 3), dtype=np.float32)
        p_R = np.tile(np.eye(3, dtype=np.float32), (n_batch, 1))
        p_R = np.reshape(p_R, (n_batch, 3, 3))

        for jid2 in chain:
            cur_R = Rs[:, jid2]
            cur_bone = bone_lengths[jid2]
            cur_bone = np.tile(cur_bone, n_batch)
            cur_bone = np.reshape(cur_bone, (n_batch, 3))
            cur_bone = np.expand_dims(cur_bone, axis=1)
            p_xyz = p_xyz + np.matmul(cur_bone, p_R)
            p_R = np.matmul(cur_R, p_R)
        xyz = np.matmul(bone, p_R) + p_xyz
        xyz = np.reshape(xyz, (n_batch, 3))
        Pts3d.append(xyz)

    Pts3d = np.stack(Pts3d, axis=1).astype(np.float32)

    Pts3d[:, :, (0, 1, 2)] = Pts3d[:, :, (0, 2, 1)]

    Pts3d = mirror_p3d(Pts3d)

    return Pts3d


def batch_rot3d(r):
    n_batch = np.shape(r)[0]
    const0 = np.zeros((n_batch,))
    const1 = np.ones((n_batch,))
    X = r[:, 0]
    Y = r[:, 1]
    Z = r[:, 2]

    # X
    # 1       0       0
    # 0   cos(a) -sin(a)
    # 0   sin(a)  cos(a)
    X_cos = np.cos(X)
    X_sin = np.sin(X)
    r1 = np.stack([const1, const0, const0], axis=1)
    r2 = np.stack([const0,  X_cos, -X_sin], axis=1)  # pylint: disable=invalid-unary-operand-type
    r3 = np.stack([const0,  X_sin,  X_cos], axis=1)
    Rx = np.stack([r1, r2, r3], axis=1)

    # Y
    #  cos(b)  0  sin(b)
    #      0   1      0
    # -sin(b)  0  cos(b)
    Y_cos = np.cos(Y)
    Y_sin = np.sin(Y)
    r1 = np.stack([Y_cos,  const0,  Y_sin], axis=1)
    r2 = np.stack([const0, const1, -const0], axis=1)  # pylint: disable=invalid-unary-operand-type
    r3 = np.stack([-Y_sin, const0,  Y_cos], axis=1)  # pylint: disable=invalid-unary-operand-type
    Ry = np.stack([r1, r2, r3], axis=1)

    # Z
    # cos(c) -sin(c)  0
    # sin(c)  cos(c)  0
    #     0       0   1
    Z_cos = np.cos(Z)
    Z_sin = np.sin(Z)
    r1 = np.stack([ Z_cos, -Z_sin, const0], axis=1)  # pylint: disable=invalid-unary-operand-type
    r2 = np.stack([ Z_sin,  Z_cos, const0], axis=1)
    r3 = np.stack([const0, const0, const1], axis=1)
    Rz = np.stack([r1, r2, r3], axis=1)
    Rzy = np.matmul(Rz, Ry)
    R = np.matmul(Rzy, Rx)
    # R = np.transpose(R, [0, 2, 1])
    return R
