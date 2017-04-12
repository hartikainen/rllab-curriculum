import argparse
import os
import os.path as osp
import random
import sys
from collections import OrderedDict

import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab import config
from rllab.misc.instrument import VariantGenerator

from sandbox.young_clgan.envs.init_sampler.base import UniformInitGenerator
from sandbox.young_clgan.envs.init_sampler.base import InitExplorationEnv
from sandbox.carlos_snn.autoclone import autoclone

from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy  # this used for both init and goal
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples
from sandbox.young_clgan.envs.base import FixedGoalGenerator

from sandbox.young_clgan.envs.init_sampler.base import update_env_init_generator, UniformListInitGenerator, FixedInitGenerator
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.envs.init_sampler.evaluator import label_inits, convert_label
from rllab.misc import logger

# from sandbox.young_clgan.utils import initialize_parallel_sampler
# initialize_parallel_sampler()


EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)

    # setup ec2
    subnets = [
        'us-east-2b', 'us-east-1a', 'us-east-1d', 'us-east-1b', 'us-east-1e', 'ap-south-1b', 'ap-south-1a', 'us-west-1a'
    ]
    ec2_instance = args.type if args.type else 'c4.2xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = 4
    else:
        mode = 'local'
        n_parallel = 4
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    exp_prefix = 'init-maze-trpo2'
    vg = VariantGenerator()
    # algorithm params
    vg.add('seed', range(20, 70, 7))
    vg.add('n_itr', [300])
    vg.add('inner_itr', [5])
    vg.add('outer_itr', lambda n_itr, inner_itr: [int(n_itr/inner_itr)])
    vg.add('batch_size', [20000])
    vg.add('max_path_length', [400])
    # environemnt params
    vg.add('init_generator', [UniformInitGenerator])
    vg.add('init_center', [(2,2)])
    vg.add('init_range', lambda init_generator: [4] if init_generator == UniformInitGenerator else [None])
    vg.add('angle_idxs', lambda init_generator: [(None,)])
    vg.add('goal', [(0, 4), ])
    vg.add('final_goal', lambda goal: [goal])
    vg.add('goal_reward', ['NegativeDistance'])
    vg.add('goal_weight', [0])  # this makes the task spars
    vg.add("inner_weight", [1])
    vg.add('terminal_bonus', [1])
    vg.add('reward_dist_threshold', [0.3])
    vg.add('terminal_eps', lambda reward_dist_threshold: [reward_dist_threshold])
    vg.add('indicator_reward', [True])
    vg.add('max_reward', [270])
    vg.add('min_reward', [10])
    # policy hypers
    vg.add('learn_std', [True])
    vg.add('policy_init_std', [1])
    vg.add('output_gain', [1])


    def run_task(v):
        random.seed(v['seed'])
        np.random.seed(v['seed'])

        # tf_session = tf.Session()

        # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
        logger.log("Initializing report and plot_policy_reward...")
        log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
        report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=2)
        report.add_header("{}".format(EXPERIMENT_TYPE))
        report.add_text(format_dict(v))

        inner_env = normalize(PointMazeEnv(
            goal_generator=FixedGoalGenerator(v['final_goal']),
            reward_dist_threshold=v['reward_dist_threshold'],
            indicator_reward=v['indicator_reward'],
            terminal_eps=v['terminal_eps'],
        ))

        init_generator_class = v['init_generator']
        if init_generator_class == UniformInitGenerator:
            init_generator = init_generator_class(init_size=np.size(v['goal']), bound=v['init_range'], center=v['init_center'])
        else:
            assert init_generator_class == FixedInitGenerator, 'Init generator not recognized!'
            init_generator = init_generator_class(goal=v['goal'])

        env = InitExplorationEnv(env=inner_env, goal=v['goal'], init_generator=init_generator, goal_reward=v['goal_reward'],
                                 goal_weight=v['goal_weight'], terminal_bonus=v['terminal_bonus'], angle_idxs=v['angle_idxs'],
                                 inner_weight=v['inner_weight'], terminal_eps=v['terminal_eps'])

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            # Fix the variance since different goals will require different variances, making thio cs parameter hard to learn.
            learn_std=v['learn_std'],
            output_gain=v['output_gain'],
            init_std=v['policy_init_std'],
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_traj = 3
        sampling_res = 2
        report.save()
        report.new_row()

        all_mean_rewards = []
        all_success = []

        for outer_itr in range(v['outer_itr']):

            init_states = np.array(v['init_center']) + \
                          np.random.uniform(-v['init_range'], v['init_range'], size=(300, np.size(v['goal'])))

            # with ExperimentLogger(log_dir, outer_itr, snapshot_mode='last', hold_outter_log=True):
            with ExperimentLogger(log_dir, 'inner', snapshot_mode='last', hold_outter_log=True):
                logger.log("Updating the environment init generator")
                update_env_init_generator(
                    env,
                    UniformListInitGenerator(init_states.tolist())
                )

                logger.log('Training the algorithm')
                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=v['batch_size'],
                    max_path_length=v['max_path_length'],
                    n_itr=v['inner_itr'],
                    discount=0.99,
                    step_size=0.01,
                    plot=False,
                )

                algo.train()

            logger.log('Generating the Heatmap...')
            avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env,
                                                                     sampling_res=sampling_res, n_traj=n_traj)
            reward_img = save_image(fig=heatmap)

            mean_rewards = np.mean(avg_rewards)
            success = np.mean(avg_success)

            all_mean_rewards.append(mean_rewards)
            all_success.append(success)

            with logger.tabular_prefix('Outer_'):
                logger.record_tabular('MeanRewards', mean_rewards)
                logger.record_tabular('Success', success)
            # logger.dump_tabular(with_prefix=False)

            report.add_image(
                reward_img,
                'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
                    outer_itr, all_mean_rewards[-1], all_success[-1],
                )
            )

            report.save()

            logger.log("Labeling the goals")
            labels = label_inits(
                init_states, env, policy, v['max_path_length'],
                min_reward=v['min_reward'],
                max_reward=v['max_reward'],
                old_rewards=None,
                n_traj=n_traj)

            logger.log("Converting the labels")
            init_classes, text_labels = convert_label(labels)

            logger.log("Plotting the labeled samples")
            total_inits = labels.shape[0]
            init_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
            for k in text_labels.keys():
                frac = np.sum(init_classes == k) / total_inits
                logger.record_tabular('GenInit_frac_' + text_labels[k], frac)
                init_class_frac[text_labels[k]] = frac

            img = plot_labeled_samples(
                samples=init_states, sample_classes=init_classes, text_labels=text_labels,
                limit=v['init_range'], center=v['init_center']
                # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
            )
            summary_string = ''
            for key, value in init_class_frac.items():
                summary_string += key + ' frac: ' + str(value) + '\n'
            report.add_image(img, 'itr: {}\nLabels of generated inits:\n{}'.format(outer_itr, summary_string), width=500)

            logger.dump_tabular(with_prefix=False)
            report.save()
            report.new_row()


    for vv in vg.variants(randomized=False):

        if mode in ['ec2', 'local_docker']:
            # # choose subnet
            # subnet = random.choice(subnets)
            # config.AWS_REGION_NAME = subnet[:-1]
            # config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            #     config.AWS_REGION_NAME]
            # config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            #     config.AWS_REGION_NAME]
            # config.AWS_SECURITY_GROUP_IDS = \
            #     config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
            #         config.AWS_REGION_NAME]
            # config.AWS_NETWORK_INTERFACES = [
            #     dict(
            #         SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
            #         Groups=config.AWS_SECURITY_GROUP_IDS,
            #         DeviceIndex=0,
            #         AssociatePublicIpAddress=True,
            #     )
            # ]

            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode=mode,
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                # plot=True,
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
                sync_s3_pkl=True,
                # for sync the pkl file also during the training
                sync_s3_png=True,
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    'export MPLBACKEND=Agg',
                    'pip install --upgrade pip',
                    'pip install --upgrade -I tensorflow',
                    'pip install git+https://github.com/tflearn/tflearn.git',
                    'pip install dominate',
                    'pip install scikit-image',
                    'conda install numpy -n rllab3 -y',
                ],
            )
            if mode == 'local_docker':
                sys.exit()
        else:
            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode='local',
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
            )
