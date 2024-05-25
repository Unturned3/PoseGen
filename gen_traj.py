
import sys, time
from os.path import join as pjoin
from timer import Timer
import cv2
import numpy as np
from numpy.linalg import norm as vec_norm
import utils
import scipy.stats as ss
import yaml

cfg = None

def wrap_angles(a):
    """Wrap angles to be in range [-180, 180]"""
    a = np.atleast_1d(a.copy())
    a %= 360
    a[a > 180] -= 360
    return a

def angular_dist(a, b):
    """Angular distance from a to b (in range [-180,180])"""
    a, b = np.atleast_1d(a, b)
    d = (b - a) % 360 # Note that Python uses true math modulo, not remainder.
    d[d > 180] -= 360
    return d

def rand_pose(prev_pose=None, target_fov=None):

    global cfg

    if prev_pose is None:
        pan = np.random.uniform(-180, 180)
        tilt = np.random.uniform(cfg['min_init_tilt'], cfg['max_init_tilt'])
        roll = np.random.uniform(cfg['min_init_roll'], cfg['max_init_roll'])
    else:
        assert target_fov is not None
        pp, pt, pr = prev_pose
        # TODO: create next pan range as linear function of target_fov
        if target_fov < 40:
            pan = utils.multimodal(ss.uniform, [pp-18, pp+18], [1, 1])
            tilt = utils.multimodal(ss.uniform, [pt-5, pt+5], [1, 1])
            tilt = np.clip(tilt, -0.7 * target_fov + 15, 0)
        else:
            pan = utils.multimodal(ss.uniform, [pp-55, pp+55], [5, 5])
            tilt = utils.multimodal(ss.uniform, [pt-10, pt+10], [5, 5])
            tilt = np.clip(tilt, -0.7 * target_fov + 15, 5)

        roll = utils.multimodal(ss.norm, [pr-2, pr+2], [1, 1])
        roll = np.clip(roll, -5, 5)

    return wrap_angles([pan, tilt, roll])

def simulate():

    global cfg

    cur_fov = cfg['init_fov']
    cur_pose = rand_pose()
    angular_vel = np.array([0, 0, 0], np.float64)
    angular_accel = np.array([0, 0, 0], np.float64)

    frame_cnt = 0
    end_simulation = False

    fovs, poses = [], []

    while not end_simulation:

        if np.random.uniform(0, 1) < cfg['zoom_prob']:
            target_fov = utils.multimodal(
                ss.norm, [35, 55, 75], [0] * 3)
        else:
            target_fov = cur_fov

        target_pose = rand_pose(cur_pose, target_fov)

        while not end_simulation:

            cur_fov += (1 - 2 * (cur_fov > target_fov)) * min(1, abs(cur_fov - target_fov))

            # Higher zoom = slower rotation
            max_angular_vel = np.random.normal(30 - 0.6 * (cfg['max_fov'] - cur_fov), 2)

            angular_err = angular_dist(cur_pose, target_pose)

            if vec_norm(angular_err) < cfg['angular_err_tolerance']:
                break

            angular_err *= 50 / vec_norm(angular_err)

            noise_vector = np.random.normal([0,0,0], cfg['noise_scale'])

            angular_accel = cfg['kp'] * angular_err + noise_vector
            damping = cfg['drag'] * angular_vel

            angular_vel += (angular_accel - damping) * cfg['dt']

            if vec_norm(angular_vel) > max_angular_vel:
                angular_vel *= max_angular_vel / vec_norm(angular_vel)

            cur_pose = (cur_pose + angular_vel * cfg['dt']) % 360
            cur_pose[cur_pose > 180] -= 360

            fovs.append(cur_fov)
            poses.append(cur_pose.copy())

            frame_cnt += 1

            if frame_cnt >= cfg['n_frames']:
                end_simulation = True

    return poses, fovs


def save_trajectory(name, poses, fovs):
    arr = np.array([[*p, f] for p, f in zip(poses, fovs)])
    np.save(name, arr)


def main():

    global cfg

    np.set_printoptions(precision=3, suppress=True)

    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f)

    n_trajs = int(sys.argv[2])
    out_dir = sys.argv[3]

    for i in range(n_trajs):
        poses, fovs = simulate()
        save_trajectory(pjoin(out_dir, f'{i:03d}.npy'), poses, fovs)

if __name__ == '__main__':
    main()
