
import argparse

import joblib
import tensorflow as tf

from sandbox.michael.rllab.sampler.utils import rollout
from rllab.misc.ext import set_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Fixed random seed')
    parser.add_argument('--best_action', type=bool, default=1,
                        help='Best action or sampling action')
    args = parser.parse_args()

    policy = None
    env = None

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    # while True:
    if args.seed >= 0:
        set_seed(args.seed)
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            policy = data["algo"].policy
            env = data["algo"].env
        else:
            policy = data['policy']
            env = data['env']
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup)
            print("Final reward: {}".format(path["rewards"][-1]))