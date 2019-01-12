import numpy as np

from curriculum.envs.maze.maze_env import MazeEnv
from curriculum.envs.maze.point_env import PointEnv


class PointMazeEnv(MazeEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
    GOAL_INDEX = slice(0, 2)

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True

    def reset_goal(self):
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
        self.goal = np.concatenate([goal_position, goal_velocity])

        qpos = self.model.data.qpos.copy()
        qvel = self.model.data.qvel.copy()

        qpos[2:4] = goal_position[:, None]
        qvel[2:4] = goal_velocity[:, None]

        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
