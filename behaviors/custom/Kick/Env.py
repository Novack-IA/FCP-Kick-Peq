import numpy as np
from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M

class Env:
    def __init__(self, base_agent : Base_Agent) -> None:
        self.world = base_agent.world 
        self.ik = base_agent.inv_kinematics
        self.obs = np.zeros(63, np.float32) 
        self.kick_ori = None

    def observe(self, init=False): # 和训练环境保持一致
        # 获取输入参数
        w = self.world
        r = self.world.robot
        if init:
            self.step_counter = 0
            self.act = np.zeros(16, np.float32)
        self.obs[0] = self.step_counter / 20
        self.obs[1] = r.loc_head_z * 3
        self.obs[2] = r.loc_head_z_vel / 2
        self.obs[3] = r.imu_torso_roll / 15
        self.obs[4] = r.imu_torso_pitch / 15
        self.obs[5:8] = r.gyro / 100
        self.obs[8:11] = r.acc / 10
        self.obs[11:17] = r.frp.get("lf", np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        self.obs[17:23] = r.frp.get("rf", np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        self.obs[23:39] = r.joints_position[2:18] / 100
        self.obs[39:55] = r.joints_speed[2:18] / 6.1395
        ball_rel_hip_center = self.ik.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        if init:
            self.obs[55:58] = (0, 0, 0)
        elif w.ball_is_visible:
            self.obs[55:58] = (ball_rel_hip_center - self.obs[58:61]) * 10
        self.obs[58:61] = ball_rel_hip_center
        self.obs[61] = np.linalg.norm(ball_rel_hip_center) * 2
        self.obs[62] = M.normalize_deg(self.kick_ori - r.imu_torso_orientation) / 30
        return self.obs

    def execute(self, action):
        r = self.world.robot
        # 线性拟合输出范围，提高训练速度，和训练环境保持一致
        self.kick_ready_step_time = 6                       # 准备踢球的时间，和训练环境保持一致
        if self.step_counter < self.kick_ready_step_time:   # 准备踢球
          action *= [6, 6, 3, 2, 1, 1, 1, 1, 4, 2, 3, 4, 4, 3, 2, 6]
          action += [1.5, -1.5, 0, -0.5, 2.5, 0, 2.5, -5, 0, -8, 2, 0, -4, 0, 0, 0.5]
        else:                                               # 出脚
          action *= [12, 15, 8, 8, 15, 15, 22, 30, 14, 28, 5, 12, 6, 7, 3, 11]
          action += [5.5, 1, 0.5, 1, -15, 1.5, -8, 19, -8, -17.5, 2, -3.5, -2.5, -1, -0.5, 3.5]
        r.joints_target_speed[2:18] = action
        r.set_joints_target_position_direct([0, 1], np.array([0, -44], float), False)
        self.step_counter += 1
        return self.step_counter >= 16 # 踢球步数时长，和训练环境保持一致
