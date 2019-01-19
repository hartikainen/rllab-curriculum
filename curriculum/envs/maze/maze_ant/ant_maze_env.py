import numpy as np

from curriculum.envs.maze.maze_env import MazeEnv
# from curriculum.envs.maze.maze_ant.ant_env import AntEnv
from curriculum.envs.maze.maze_ant.ant_target_env import AntEnv


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0
    GOAL_INDEX = slice(-2, None)

    # MANUAL_COLLISION = True  # this is in point_mass
    # MAZE_MAKE_CONTACTS = True  # this is in rllab

    def __init__(self, *args, **kwargs):
        self.fixed_goal = None
        super(AntMazeEnv, self).__init__(*args, **kwargs)

    def update_goal_position(self):
        if self.fixed_goal is not None:
            goal_position = self.fixed_goal[:2]
        else:
            y_length = (
                len(self.MAZE_STRUCTURE)
                * self.MAZE_SIZE_SCALING
                - self._init_torso_y)
            x_length = (
                len(self.MAZE_STRUCTURE[0])
                * self.MAZE_SIZE_SCALING
                - self._init_torso_x)

            goal_position = np.random.uniform([0, 0], [x_length, y_length])
            while not self.is_feasible(goal_position):
                goal_position = np.random.uniform([0, 0], [x_length, y_length])

        goal_velocity = np.array([0, 0])

        goal_qpos = self.init_qpos.flat.copy()
        goal_qvel = self.init_qvel.flat[:-2].copy()

        goal_qpos[:2] = goal_position
        goal_qvel[:2] = goal_velocity

        goal_observation = np.concatenate([
            goal_qpos,
            goal_qvel,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            np.eye(3).flat,
            [0, 0, 0.47967345],
        ]).reshape(-1)

        self.goal = goal_observation

        # Update the goal body position

        qpos = self.model.data.qpos.copy()
        qvel = self.model.data.qvel.copy()

        qpos[-2:] = goal_position[:, None]
        qvel[-2:] = goal_velocity[:, None]

        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
