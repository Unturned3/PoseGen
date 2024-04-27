
import cv2
from PIL import Image
from equilib import equi2pers
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm as vec_norm
import utils
from timer import Timer
import scipy.stats as ss

import os, time

print("Starting...")

#image_path = '/Users/richard/Desktop/test.jpg'
image_path = 'pano-1024.jpg'
equi_img = cv2.imread(image_path)
equi_img = np.transpose(equi_img, (2, 0, 1))

# pyequilib uses a right-handed coordinate system: X forward, Y left, Z up.
# However, this has the unintuitive effect that negative tilt (pitch) angles
# will make the camera look up. So, we set z_down=True to fix this behavior.

n_keyframes = 20
dt = 1 / 30
kp = 2
drag = 1

angular_err_tolerance = 10

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

def rand_pan_tilt(prev_pose=None, target_fov=None):
    if prev_pose is None:
        pan = np.random.uniform(-180, 180)
        tilt = np.random.uniform(-40, 10)
        roll = np.random.uniform(-5, 5)
    else:
        assert target_fov is not None
        pp, pt, pr = prev_pose
        """ TODO: draw from more realistic distributions. A simple uniform
            range around the current pose can create unnaturally small
            "jittery" movements (when the next pose is very close to the
            current pose). Do something like a bimodal distribution with peaks
            at ±X degrees from the current pan/tilt.
        """
        #min_tilt, max_tilt = -50, 40
        #pan = np.random.uniform(pp-30, pp+30)
        #tilt = np.random.uniform(max(min_tilt, pt-20), min(max_tilt, pt+20))
        pan = utils.multimodal(ss.uniform, [pp-65, pp+55], [10, 10])
        tilt = utils.multimodal(ss.uniform, [pt-15, pt+5], [10, 10])
        tilt = np.clip(tilt, -0.5 * target_fov - 2.5, 10)
        roll = utils.multimodal(ss.norm, [pr-2, pr+2], [1, 1])
        roll = np.clip(roll, -5, 5)
    return wrap_angles([pan, tilt, roll])

def main():
    np.set_printoptions(precision=3, suppress=True)

    max_fov = 75
    min_fov = 15

    cur_fov = 75    # TODO: make this random too?
    cur_pose = rand_pan_tilt()
    angular_vel = np.array([0, 0, 0], np.float64)
    angular_accel = np.array([0, 0, 0], np.float64)

    n_frames = 300
    frame_cnt = 0
    end_simulation = False

    fovs, poses = [], []

    while not end_simulation:

        # 50% chance to attempt zoom change
        if np.random.uniform(0, 1) < 0.5:
            target_fov = utils.multimodal(
                ss.norm, [15, 35, 55, 75], [1.5] * 4)
        else:
            target_fov = cur_fov

        target_pose = rand_pan_tilt(cur_pose, target_fov)

        while not end_simulation:

            cur_fov += (1 - 2 * (cur_fov > target_fov)) * min(1, abs(cur_fov - target_fov))

            # Higher zoom = slower rotation
            max_angular_vel = np.random.normal(50 - 0.7 * (max_fov - cur_fov), 2)

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

    pan_angles = [e[0] for e in poses]
    tilt_angles = [e[1] for e in poses]
    plt.hist(pan_angles)
    plt.hist(tilt_angles)
    plt.hist(fovs)
    plt.show()

    #print('Press return to proceed...', end='')
    #input()

    with Timer():
        imgs = []
        for p, fov in zip(poses, fovs):
            start_time = time.time()
            rots = {
                'yaw': np.radians(p[0]),
                'pitch': np.radians(p[1]),
                'roll': np.radians(p[2]),
            }
            img = equi2pers(
                equi=equi_img,
                height=120,
                width=160,
                fov_x=fov,
                rots=rots,
                mode='bilinear',
                z_down=True,
            )
            img = np.transpose(img, (1, 2, 0))
            t = time.time() - start_time
            cv2.imshow('img', img)
            wait_time = max(int((dt-t)*1000), 1)
            cv2.waitKey(wait_time)
            #cv2.waitKey(1)
            imgs.append(img)

    cv2.destroyAllWindows()
    cv2.waitKey(1) # Without this the window won't actually close

    print('Save trajectory (y/n)? ', end='')
    if input() == 'y':
        arr = np.array([[*p, f] for p, f in zip(poses, fovs)])
        print(f'arr.shape: {arr.shape}')
        np.save('out.npy', arr)

if __name__ == '__main__':
    main()


#print('Save video (y/n)? ', end='')
#if input() == 'y':
#    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
#    for i in range(0, frames):
#        writer.write(imgs[i])
#    writer.release()

