import pickle
from agent.Base_Agent import Base_Agent
from behaviors.custom.Kick.Env import Env
from math_ops.Neural_Network import run_mlp
from math_ops.Math_Ops import Math_Ops as M

class Kick:

    def __init__(self, base_agent : Base_Agent) -> None:
        self.world = base_agent.world
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.description = "Kick"
        self.auto_head = False
        self.env = Env(base_agent)
        with open(M.get_active_directory('/behaviors/custom/Kick/Kick_R1_25.6Msteps.pkl'), "rb") as f:
            self.model = pickle.load(f)
        r_type = self.world.robot.type
        self.ball_x_limits = ((0.19,0.215), (0.2,0.22), (0.19,0.22), (0.2,0.215), (0.2,0.215))[r_type]
        self.ball_y_limits = ((-0.115,-0.1), (-0.125,-0.095), (-0.12,-0.1), (-0.13,-0.105), (-0.09,-0.06))[r_type]
        self.ball_x_center = (self.ball_x_limits[0] + self.ball_x_limits[1])/2
        self.ball_y_center = (self.ball_y_limits[0] + self.ball_y_limits[1])/2

    def execute(self, reset, direction):
        w = self.world
        r = self.world.robot
        b = w.ball_rel_torso_cart_pos
        t = w.time_local_ms
        
        Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator

        if reset:
            self.phase = 0
            self.reset_time = t
        if self.phase == 0:
            ang_diff = abs(M.normalize_deg(direction - r.loc_torso_orientation))# the reset was learned with loc, not IMU
            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=direction,
                x_dev=(-0.2),   # 和训练环境保持一致
                y_dev=0.05,     # 和训练环境保持一致
                torso_ori=direction)
            if (dist_to_final_target < 0.02 # 和训练环境保持一致
                and ang_diff < 5            # 和训练环境保持一致
                and (w.time_local_ms - w.ball_abs_pos_last_update) < 100 # 和训练环境保持一致
                and (w.time_local_ms - self.reset_time) > 500    # 和训练环境保持一致
                and not Step_Generator.state_is_left_active and Step_Generator.state_current_ts == 2 # 和训练环境保持一致
                ):
                self.phase += 1
                self.env.kick_ori = direction
                obs = self.env.observe(True)
                action = run_mlp(obs, self.model)
                return self.env.execute(action)
            else:
                dist = max(0.07, dist_to_final_target * 0.9) # 和训练环境保持一致
                reset_walk = reset and self.behavior.previous_behavior != "Walk" # reset walk if it wasn't the previous behavior
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)
                return False
        else: # define kick parameters and execute 
            self.env.kick_ori = direction
            obs = self.env.observe(False)
            action = run_mlp(obs, self.model)
            return self.env.execute(action)

    def is_ready(self) -> any: # You can add more arguments 
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True
