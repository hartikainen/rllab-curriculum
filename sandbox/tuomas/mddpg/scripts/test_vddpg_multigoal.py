import tensorflow as tf
import numpy as np
import pickle

from rllab.envs.proxy_env import ProxyEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy


from sandbox.tuomas.mddpg.misc.rollout import rollout
from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.critics.gaussian_critic import MixtureGaussian2DCritic

from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('actor_lr', 0.01, 'Base learning rate for actor.')
flags.DEFINE_float('critic_lr', 0.0001, 'Base learning rate for critic.')
flags.DEFINE_integer('path_length', 100, 'Maximum path length.')


class VDDPGTest(VDDPG):
    lim = 2
    plots_initialized = False
    eval_counter = 0


    def eval_critic(self, o):
        xx = np.arange(-self.lim, self.lim, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = {
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: all_actions
        }
        Q = self.sess.run(self.qf.output, feed).reshape(X.shape)
        return X, Y, Q

    @overrides
    def evaluate(self, epoch, es_path_returns):
        self.eval_counter += 1
        n_paths = 10


        if not self.plots_initialized:
            env = self.env
            while isinstance(env, ProxyEnv):
                env = env.wrapped_env

            #fig = plt.figure(figsize=(7, 12))
            fig = plt.figure(figsize=(7, 7))
            self.ax_paths = fig.add_subplot(111)
            plt.axis('equal')

            # Set up all critic plots.
            fig = plt.figure(figsize=(20, 7))
            self.ax_critics = []
            for i in range(3):
                ax = fig.add_subplot(130 + i + 1)
                self.ax_critics.append(ax)
                plt.axis('equal')
                ax.set_xlim((-self.lim, self.lim))
                ax.set_ylim((-self.lim, self.lim))
            #self.ax_critic = fig.add_subplot(131)


            # Plot cost only once.
            env.plot_position_cost(self.ax_paths)

            # Create path objects only once.
            self.h_paths = []
            for i in range(n_paths):
                self.h_paths += self.ax_paths.plot([], [])

            self.plots_initialized = True

        obss = np.array([[-2.5, 0.0],
                         [0.0, 0.0],
                         [2.5, 2.5]])

        for ax_critic, obs in zip(self.ax_critics, obss):

            X, Y, Q = self.eval_critic(obs)

            ax_critic.clear()
            cs = ax_critic.contour(X, Y, Q, 20)
            ax_critic.clabel(cs, inline=1, fontsize=10, fmt='%.0f')

            # sample and plot actions
            N = 250
            all_obs = np.array([obs] * N)
            all_actions = self.policy.get_actions(all_obs)[0]


            x = all_actions[:, 0]
            y = all_actions[:, 1]
            ax_critic.plot(x, y, '*')

        # Do rollouts.
        for line in self.h_paths:
            path = rollout(self.env, self.policy, FLAGS.path_length)
            line.set_data(np.transpose(path['observations']))

        plt.draw()
        plt.pause(0.001)






        ## Switch to the actual q:
        #if self.eval_counter == 100:
        #    self.qf = self.q_actual
        #    self._init_ops()

        #    # Dump the current actor first.
        #    params = self.dummy_policy.get_param_values()
        #    self.sess.run(tf.initialize_all_variables())
        #    self.dummy_policy.set_param_values(params)

        if self.eval_counter % 25 == 0:
            self.alpha /= 3



        return

# -------------------------------------------------------------------
def test():
    from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.tuomas.mddpg.policies.stochastic_policy \
        import StochasticNNPolicy
    from sandbox.tuomas.mddpg.envs.multi_goal_env import MultiGoalEnv
    from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic

    K = 16  # number of particles

    ddpg_kwargs = dict(
        epoch_length=25,  # evaluate / plot per SVGD step
        min_pool_size=2,  # must be at least 2
        #replay_pool_size=2,  # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples=1,  # doesn't matter since we override evaluate()
        n_epochs=100000,  # number of SVGD steps
        policy_learning_rate=FLAGS.actor_lr,  # note: this is higher than DDPG's 1e-4
        qf_learning_rate=FLAGS.critic_lr,
        #soft_target_tau=1.0,
        max_path_length=FLAGS.path_length,
        batch_size=64,  # only need recent samples, though it's slow
        alpha=3,#0.1,  # 1 is the SVGD default
        train_actor=True,
        train_critic=True,
        q_target_type='mean'
    )
    # ----------------------------------------------------------------------
    env = TfEnv(MultiGoalEnv())

    es = DummyExplorationStrategy()

    #es = OUStrategy(env_spec=env.spec, mu=0, theta=0.15, sigma=0.03)

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )

    q_init = MixtureGaussian2DCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        observation_input=None,
        weights=[1.],
        mus=[np.array((0., 0.))],
        sigmas=[1.0]
    )

    policy = StochasticNNPolicy(
        'actor',
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        hidden_dims=(100, 100),
        output_nonlinearity= None, #tf.tanh,
        sample_dim=2,
    )

    kernel = SimpleAdaptiveDiagonalGaussianKernel(
        "kernel",
        dim=env.action_space.flat_dim,
    )

    algorithm = VDDPGTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        kernel=kernel,
        qf=qf,
        q_init=qf,#_init,
        K=K,
        **ddpg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
