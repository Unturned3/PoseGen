
import cv2
import numpy as np
from numpy.linalg import norm as vec_norm
import utils
from timer import Timer
import scipy.stats as ss
import time

show_histogram = False

if show_histogram:
    import matplotlib.pyplot as plt

# pyequilib uses a right-handed coordinate system: X forward, Y left, Z up.
# However, this has the unintuitive effect that negative tilt (pitch) angles
# will make the camera look up. So, we set z_down=True to fix this behavior.

n_frames = 900
dt = 1 / 30
kp = 2
drag = 1
angular_err_tolerance = 5

max_fov = 75
min_fov = 15

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
    if prev_pose is None:
        pan = np.random.uniform(-180, 180)
        tilt = np.random.uniform(-5, 0)
        roll = np.random.uniform(-5, 5)
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
    cur_fov = 75
    cur_pose = rand_pose()
    angular_vel = np.array([0, 0, 0], np.float64)
    angular_accel = np.array([0, 0, 0], np.float64)

    frame_cnt = 0
    end_simulation = False

    fovs, poses = [], []

    while not end_simulation:

        if np.random.uniform(0, 1) < 0.75:
            target_fov = utils.multimodal(
                ss.norm, [35, 55, 75], [0] * 3)
        else:
            target_fov = cur_fov

        target_pose = rand_pose(cur_pose, target_fov)

        while not end_simulation:

            cur_fov += (1 - 2 * (cur_fov > target_fov)) * min(1, abs(cur_fov - target_fov))

            # Higher zoom = slower rotation
            max_angular_vel = np.random.normal(30 - 0.6 * (max_fov - cur_fov), 2)

            angular_err = angular_dist(cur_pose, target_pose)

            if vec_norm(angular_err) < angular_err_tolerance:
                break

            angular_err *= 50 / vec_norm(angular_err)

            noise_vector = np.random.normal([0,0,0], [50,50,50])

            angular_accel = kp * angular_err + noise_vector
            damping = drag * angular_vel

            angular_vel += (angular_accel - damping) * dt

            if vec_norm(angular_vel) > max_angular_vel:
                angular_vel *= max_angular_vel / vec_norm(angular_vel)

            cur_pose = (cur_pose + angular_vel * dt) % 360
            cur_pose[cur_pose > 180] -= 360

            fovs.append(cur_fov)
            poses.append(cur_pose.copy())

            frame_cnt += 1

            if frame_cnt >= n_frames:
                end_simulation = True

    if show_histogram:
        pan_angles = [e[0] for e in poses]
        tilt_angles = [e[1] for e in poses]
        plt.hist(pan_angles)
        plt.hist(tilt_angles)
        plt.hist(fovs)
        plt.show()

    return poses, fovs


def save_trajectory(name, poses, fovs):
    arr = np.array([[*p, f] for p, f in zip(poses, fovs)])
    np.save(name, arr)


def main():
    np.set_printoptions(precision=3, suppress=True)

    for i in range(25):
        poses, fovs = simulate()
        save_trajectory(f'/Users/richard/Desktop/Trajs/t{i:03d}.npy', poses, fovs)

if __name__ == '__main__':
    main()
