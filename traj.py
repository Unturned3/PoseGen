
import cv2
import Equirec2Perspec as ep
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm as vec_norm
import utils

import os, time

equ = ep.Equirectangular('t2.png')

n_keyframes = 10
dt = 1 / 25
kp = 1
drag = 2

max_angular_vel = 60
angular_err_tolerance = 5

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

def rand_pan_tilt(prev_pose=None):
    min_tilt, max_tilt = -50, 40
    if prev_pose is None:
        pan = np.random.uniform(-180, 180)
        tilt = np.random.uniform(-50, 30)
    else:
        pp, pt = prev_pose
        """ TODO: draw from more realistic distributions. A simple uniform
            range around the current pose can create unnaturally small
            "jittery" movements (when the next pose is very close to the
            current pose). Do something like a bimodal distribution with peaks
            at ±X degrees from the current pan/tilt.
        """
        #pan = np.random.uniform(pp-30, pp+30)
        #tilt = np.random.uniform(max(min_tilt, pt-20), min(max_tilt, pt+20))
        pan = utils.bimodal_normal([0.5, 0.5], [pp-60, pp+60], [1, 1])
        tilt = utils.bimodal_normal([0.5, 0.5], [pt-20, pt+20], [1, 1])
        tilt = np.clip(tilt, -50, 40)
    return wrap_angles([pan, tilt])

def main():
    np.set_printoptions(precision=3, suppress=True)

    """ TODO: instead of moving to a new pose in each keyframe, we can
        duplicate certain keyframes a number of times to "pause" the rotation.

        Vary FoV, angular velocity, roll, add jitter, etc.
    """

    keyframes = [rand_pan_tilt()]
    for _ in range(n_keyframes - 1):
        keyframes.append(rand_pan_tilt(keyframes[-1]))

    pose = keyframes[0]
    angular_vel = np.array([0, 0], np.float64)
    angular_accel = np.array([0, 0], np.float64)

    poses = []

    for keyframe in keyframes[1:]:

        #max_angular_vel = np.clip(np.random.normal(50, 2), 0, 90)
        max_angular_vel = np.random.uniform(10, 90)

        while True:

            angular_err = angular_dist(pose, keyframe)

            if vec_norm(angular_err) < angular_err_tolerance:
                break

            angular_err *= 50 / vec_norm(angular_err)

            angular_accel = kp * angular_err
            damping = drag * angular_vel

            angular_vel += (angular_accel - damping) * dt

            if vec_norm(angular_vel) > max_angular_vel:
                angular_vel *= max_angular_vel / vec_norm(angular_vel)

            pose = (pose + angular_vel * dt) % 360
            pose[pose > 180] -= 360

            poses.append(pose.copy())

    pan_angles = [e[0] for e in poses]
    tilt_angles = [e[1] for e in poses]

    plt.hist(pan_angles)
    plt.hist(tilt_angles)
    plt.show()

    #print('Press return to proceed...', end='')
    #input()

    imgs = []
    for p in poses:
        start_time = time.time()
        img = equ.GetPerspective(75, p[0], p[1], 240, 320)
        t = time.time() - start_time
        cv2.imshow('img', img)
        cv2.waitKey(max(int(1/dt- t*1000), 1))
        #cv2.waitKey(1)
        imgs.append(img)

    cv2.destroyAllWindows()
    cv2.waitKey(1) # Without this the window won't actually close


if __name__ == '__main__':
    main()


#print('Save video (y/n)? ', end='')
#if input() == 'y':
#    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
#    for i in range(0, frames):
#        writer.write(imgs[i])
#    writer.release()

