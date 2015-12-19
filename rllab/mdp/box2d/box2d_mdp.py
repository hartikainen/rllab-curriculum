from contextlib import contextmanager
import numpy as np
import os.path as osp
from rllab.mdp.box2d.parser.xml_box2d import world_from_xml, find_body, \
    find_joint
from rllab.mdp.box2d.box2d_viewer import Box2DViewer
from rllab.mdp.base import ControlMDP
from rllab.misc.overrides import overrides


class Box2DMDP(ControlMDP):

    def __init__(self, model_path):
        with open(model_path, "r") as f:
            s = f.read()
        world, extra_data = world_from_xml(s)
        self.world = world
        self.extra_data = extra_data
        self.initial_state = self.get_state()
        self.current_state = self.initial_state
        self.viewer = None
        self._action_bounds = None
        self._observation_shape = None

    def model_path(self, file_name):
        return osp.abspath(osp.join(osp.dirname(__file__),
                                    'models/%s' % file_name))

    def set_state(self, state):
        splitted = np.array(state).reshape((-1, 6))
        for body, body_state in zip(self.world.bodies, splitted):
            xpos, ypos, apos, xvel, yvel, avel = body_state
            body.position = (xpos, ypos)
            body.angle = apos
            body.linearVelocity = (xvel, yvel)
            body.angularVelocity = avel

    @property
    @overrides
    def state_shape(self):
        return (len(self.world.bodies) * 6,)

    @overrides
    def reset(self):
        self.set_state(self.initial_state)
        return self.get_state(), self.get_current_obs()

    def get_state(self):
        s = []
        for body in self.world.bodies:
            s.append(np.concatenate([
                list(body.position),
                [body.angle],
                list(body.linearVelocity),
                [body.angularVelocity]
            ]))
        return np.concatenate(s)

    @property
    @overrides
    def action_dim(self):
        return len(self.extra_data.controls)

    @property
    @overrides
    def action_dtype(self):
        return 'float32'

    @property
    @overrides
    def observation_dtype(self):
        return 'float32'

    @property
    @overrides
    def observation_shape(self):
        if not self._observation_shape:
            self._observation_shape = self.get_current_obs().shape
        return self._observation_shape

    @property
    @overrides
    def action_bounds(self):
        if not self._action_bounds:
            lb = [control.ctrllimit[0] for control in self.extra_data.controls]
            ub = [control.ctrllimit[1] for control in self.extra_data.controls]
            self._action_bounds = (np.array(lb), np.array(ub))
        return self._action_bounds

    @contextmanager
    def set_state_tmp(self, state, restore=True):
        if np.array_equal(state, self.current_state) and not restore:
            yield
        else:
            prev_state = self.current_state
            self.set_state(state)
            yield
            if restore:
                self.set_state(prev_state)
            else:
                self.current_state = self.get_state()

    @overrides
    def forward_dynamics(self, state, action, restore=True):
        if len(action) != self.action_dim:
            raise ValueError('incorrect action dimension: expected %d but got '
                             '%d' % (self.action_dim, len(action)))
        with self.set_state_tmp(state, restore):
            lb, ub = self.action_bounds
            action = np.clip(action, lb, ub)
            for ctrl, act in zip(self.extra_data.controls, action):
                if ctrl.typ == "force":
                    assert ctrl.body
                    body = find_body(self.world, ctrl.body)
                    direction = np.array(ctrl.direction)
                    direction = direction / np.linalg.norm(direction)
                    world_force = body.GetWorldVector(direction * act)
                    world_point = body.GetWorldPoint(ctrl.anchor)
                    body.ApplyForce(world_force, world_point, wake=True)
                elif ctrl.typ == "torque":
                    assert ctrl.joint
                    joint = find_joint(self.world, ctrl.joint)
                    joint.motorEnabled = True
                    # forces the maximum allowed torque to be taken
                    if act > 0:
                        joint.motorSpeed = 1e5
                    else:
                        joint.motorSpeed = -1e5
                    joint.maxMotorTorque = abs(act)
                else:
                    raise NotImplementedError
            self.world.Step(
                self.extra_data.timeStep,
                self.extra_data.velocityIterations,
                self.extra_data.positionIterations
            )
            return self.get_state()

    @overrides
    def step(self, state, action):
        reward = self.get_current_reward(action)
        next_state = self.forward_dynamics(state, action,
                                           restore=False)
        done = self.is_current_done()
        next_obs = self.get_current_obs()
        return next_state, next_obs, reward, done

    def get_current_reward(self, action):
        raise NotImplementedError

    def is_current_done(self):
        raise NotImplementedError

    def get_current_obs(self):
        obs = []
        for state in self.extra_data.states:
            body = find_body(self.world, state.body)
            if state.typ == "xpos":
                obs.append(body.position[0])
            elif state.typ == "ypos":
                obs.append(body.position[1])
            elif state.typ == "xvel":
                obs.append(body.linearVelocity[0])
            elif state.typ == "yvel":
                obs.append(body.linearVelocity[1])
            elif state.typ == "apos":
                obs.append(body.angle)
            elif state.typ == "avel":
                obs.append(body.angularVelocity)
            else:
                raise NotImplementedError
        return np.array(obs)

    @overrides
    def start_viewer(self):
        if not self.viewer:
            self.viewer = Box2DViewer(self.world)

    @overrides
    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()
        self.viewer = None

    @overrides
    def plot(self, states=None, actions=None, pause=False):
        if states or actions or pause:
            raise NotImplementedError
        if self.viewer:
            self.viewer.loop_once()