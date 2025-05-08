import math, os, gym, numpy as np
from agent.Base_Agent import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M
from scripts.commons.Server     import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3          import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

class Kick(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, ip, server_p, monitor_p, robot_type,
                 enable_draw=False):
        '''---重要--- 训练的踢球距离，踢球步数时长（动作时间越长一般效果越好，但踢得更慢）'''
        self.kick_dist = 18
        self.kick_whole_step_time = 16
        self.kick_ready_step_time = 6
        # 球起始的位置，一般无需调整
        self.ball_start_pos = (-12, 0, 0.042)
        
        self.player = Agent(ip, server_p, monitor_p, 1, robot_type, "Gym", False, enable_draw)
        self.server_p = server_p
        self.step_counter = 0
        self.ik     = self.player.inv_kinematics
        self.kick_ori = 0

        # 输入参数，63维，一般无需调整
        obs_size = 63
        self.obs = np.zeros(obs_size, np.float32) 
        self.observation_space = gym.spaces.Box(low=np.full(obs_size, -np.inf, np.float32), 
                                                high=np.full(obs_size, np.inf, np.float32), 
                                                dtype=np.float32)
        # 输出参数，16维，一般无需调整
        MAX = np.finfo(np.float32).max  # 输出域：负无穷到正无穷，一般无需调整
        self.no_of_actions = act_size = 16 
        self.action_space = gym.spaces.Box(low=np.full(act_size, -MAX, np.float32), 
                                           high=np.full(act_size, MAX, np.float32), 
                                           dtype=np.float32)
        # 确保服务端开启Cheats，以便各种命令（传送，获取球速等）
        assert np.any(self.player.world.robot.cheat_abs_pos), "Cheats are not enabled! Run_Utils.py -> Server -> Cheats"
        # 设置合理的游戏模式
        self.player.scom.unofficial_set_play_mode("GameOver")
        self.player.scom.unofficial_set_play_mode("PlayOn")
        self.player.scom.unofficial_move_ball(self.ball_start_pos, [0, 0, -0.001])
        self.player.scom.unofficial_set_game_time(0)

    def observe(self, init=False):
        # 获取输入参数
        w = self.player.world
        r = self.player.world.robot
        if init:
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

    def reset(self):
        p, w, r = self.player, self.player.world, self.player.world.robot
        
        # 绕着球随机放置机器人
        # 随机半圆上的点，一般无需调整
        radius = 0.5
        # theta = np.random.uniform(0, 2 * math.pi)
        theta = np.random.uniform(0.5 * math.pi, 1.5 * math.pi)
        robot_x = radius * math.cos(theta) + self.ball_start_pos[0]
        robot_y = radius * math.sin(theta)
        # 随机朝向，一般无需调整
        # theta = np.random.uniform(0, 360)
        theta = np.random.uniform(-90, 90)
        # 放置
        for _ in range(25):
            p.scom.unofficial_beam((robot_x, robot_y, 0.50), theta)
            p.behavior.execute("Zero_Bent_Knees") 
            self.sync()
        p.scom.unofficial_beam((robot_x, robot_y, r.beam_height), theta)
        r.joints_target_speed[0] = 0.01
        self.sync()
        for _ in range(7):
            p.behavior.execute("Zero_Bent_Knees")
            self.sync()
            
        # 让机器人走过去，模拟各种真实的随机的踢球情况
        step_gen = p.behavior.get_custom_behavior_object("Walk").env.step_generator
        reset_walk = False
        reset_time = w.time_local_ms
        while(True):
            self.player.scom.unofficial_move_ball(self.ball_start_pos, [0, 0, -0.001]) # 保证球在固定位置
            w.ball_abs_pos = np.array(self.ball_start_pos, np.float32)
            next_pos, next_ori, dist_to_final_target = p.path_manager.get_path_to_ball(
                x_ori = 0,      # 一般无需调整
                x_dev = (-0.2), # 决定站在球后面的距离，结合需求调整
                y_dev = 0.05,   # 决定站在球的左右偏差，结合需求调整
                torso_ori = 0)  # 一般无需调整
            ang_diff = abs(M.normalize_deg(0 - r.loc_torso_orientation))
            if (dist_to_final_target < 0.02 # 决定多近开始踢，结合需求调整（也可以自行设置球离机器人的范围，参考FCP的Basic_Kick）
                and ang_diff < 5            # 决定角度多准开始踢，这里是可以接受机器人面朝方向误差5度，结合需求调整
                and (w.time_local_ms - w.ball_abs_pos_last_update) < 100 # 丢失球视野的最长时间，结合需求调整
                and (w.time_local_ms - reset_time) > 500    # 最短准备时间，结合需求调整
                and not step_gen.state_is_left_active and step_gen.state_current_ts == 2 # 左脚站地上
                ):
                break
            else:
                # 摔了就站起来
                if p.behavior.is_ready('Get_Up'):
                    while(not p.behavior.execute('Get_Up')):
                        self.sync()
                # 走路参数准备
                index = 0.9 # 决定靠近球的速度，越大越快，适当调低可以提高精度，但准备时间会延长
                dist = max(0.07, dist_to_final_target * index)
                reset_walk = not reset_walk and p.behavior.previous_behavior != "Walk"
                p.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)
                self.sync()
        
        # 重置参数
        self.target = (self.kick_dist + self.ball_start_pos[0], 0)
        self.ball_was_kicked = False
        self.step_counter = 0
        return self.observe(True)

    def step(self, action):
        # 获取需要的数值
        p, w, r = self.player, self.player.world, self.player.world.robot
        # 防止比赛超时
        self.player.scom.unofficial_set_game_time(0)
        # 执行动作
        self.execute(action)
        self.sync()
        
        # 判断是否踢了球
        ball_speed = np.linalg.norm(w.ball_cheat_abs_vel[:2])
        if(ball_speed > 0.1 and not self.ball_was_kicked):
            self.ball_was_kicked = True
            
        reward = 0
        terminal = self.step_counter >= self.kick_whole_step_time
        if (terminal and self.ball_was_kicked):
            # 踢球后发呆，等球停下（长踢适用，不需要考虑摔倒）
            while(np.linalg.norm(w.ball_cheat_abs_vel[:2]) >= 0.1):
                p.behavior.execute("Zero_Bent_Knees")
                self.sync()
            # 踢球后走路，等球停下（短踢适用），结合需求决定是“发呆等球”还是“走路等球”
            # reset_walk = True
            # while(np.linalg.norm(w.ball_cheat_abs_vel[:2]) >= 0.1):
            #     next_pos, next_ori, distance_to_final_target = p.path_manager.get_path_to_target((-15, 0))
            #     p.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, distance_to_final_target)
            #     self.sync()
            #     if(reset_walk): reset_walk = False
            
            # 利用远离比例计算准度奖励
            ball_target_dist = np.linalg.norm(w.ball_cheat_abs_pos[:2] - self.target)
            goal_rate = 1 - (ball_target_dist / self.kick_dist)
            PRECISE_INDEX = 2 # 准度系数，较小的数值（1-2）学得快，较大的数值（2-3）精度高，结合需求调整
            reward += 30 * (goal_rate ** PRECISE_INDEX)   # 最高30分，结合需求调整
            # 踢完走路没摔给额外奖励（短踢适用），结合需求调整分数
            # robot_pos = r.cheat_abs_pos
            # reward += robot_pos[2]
                
            if(self.server_p == 3100):
                print("球到目标落点的距离:", round(ball_target_dist, 2),
                      "准度:", round(goal_rate, 2),
                      "奖励:", round(reward, 2))

        return self.observe(), reward, terminal, {}
    
    def execute(self, action):
        r = self.player.world.robot
        # 利用PPO的特性，把输出线性拟合一个范围，提高训练效率，一般无需调整
        # -------------------------------------------------
        if self.step_counter < self.kick_ready_step_time:   # 准备踢球
          action *= [6, 6, 3, 2, 1, 1, 1, 1, 4, 2, 3, 4, 4, 3, 2, 6]
          action += [1.5, -1.5, 0, -0.5, 2.5, 0, 2.5, -5, 0, -8, 2, 0, -4, 0, 0, 0.5]
        else:                                               # 出脚
          action *= [12, 15, 8, 8, 15, 15, 22, 30, 14, 28, 5, 12, 6, 7, 3, 11]
          action += [5.5, 1, 0.5, 1, -15, 1.5, -8, 19, -8, -17.5, 2, -3.5, -2.5, -1, -0.5, 3.5]
        # -------------------------------------------------
        r.joints_target_speed[2:18] = action
        r.set_joints_target_position_direct([0, 1], np.array([0, -44], float), False)
        self.step_counter += 1
    
    def sync(self):
        self.player.scom.commit_and_send(self.player.world.robot.get_command())
        self.player.scom.receive()

    def render(self, mode='human', close=False):pass

    def close(self):self.player.terminate()

class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(4, os.cpu_count()) # 训练线程数，调试时建议1线程，训练时需要结合实际情况（CPU性能）调整，一般越大训练得越快
        n_steps_per_env = 128   # 训练初期建议128加快学习，后期建议1024（甚至2048）提高稳定性，需要结合训练过程调整
        minibatch_size = 64     # 训练初期建议64加快学习，后期建议256（甚至512）提高稳定性，需要结合训练过程调整
        total_steps = 1000000   # 训练一次的步数，需要结合训练过程调整
        learning_rate = 3e-4    # 学习率，开始建议3e-4，后期建议1e-4（甚至1e-5）提高稳定性，需要结合训练过程调整

        folder_name = f'Kick_R{self.robot_type}'    # 保存文件夹，结合需求调整
        model_path = f'./scripts/gyms/logs/{folder_name}/'  # 保存模型路径，结合需求调整

        print("Model path:", model_path)
        # 环境对象创建函数
        def init_env(i_env):    
            def thunk():
                return Kick(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk
        # 测试用的服务端
        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)    
        # 创建环境
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])   
        eval_env = SubprocVecEnv([init_env(n_envs)])
        '''
        PPO参数：
        必要：
        "MlpPolicy"                 多层感知机策略（CPU友好，在CPU下训练和运算得更快）
                                    注意只有MlpPolicy可以用run_mlp运算
                                    如果你用了别的策略（比如CNN），要自行写一个run_cnn
        env=env                     训练环境
        verbose=1                   日志级别
        n_steps=n_steps_per_env     每个环境的步数
        batch_size=minibatch_size   批量大小
        learning_rate=learning_rate 学习率
        device="cpu"                训练设备（CPU或GPU）
        
        模型架构，结合需求调整
        policy_kwargs=dict(net_arch=[64])       （FCP用的这个，这里代表“输入-64维运算-输出”架构）
        policy_kwargs=dict(net_arch=[64,64])    （这里代表“输入-64维运算-64维运算-输出”架构）
        理论上来说：运算层越多，模型越精细，但是训练越慢
        '''
        # 决定是继续训练还是从头训练
        try:
            if "model_file" in args:
                model = PPO.load(args["model_file"],
                                 env=env,
                                 n_envs=n_envs,
                                 n_steps=n_steps_per_env,
                                 batch_size=minibatch_size,
                                 learning_rate=learning_rate,
                                 policy_kwargs=dict(net_arch=[64]),
                                 device="cpu"
                                 )
            else:
                model = PPO("MlpPolicy",
                            env=env, verbose=1,
                            n_steps=n_steps_per_env,
                            batch_size=minibatch_size,
                            learning_rate=learning_rate,
                            policy_kwargs=dict(net_arch=[64]),
                            device="cpu"
                            )

            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env * 10, save_freq=n_steps_per_env * 20, backup_env_file=__file__)
        except KeyboardInterrupt:   # Ctrl+C关闭
            from time import sleep
            sleep(1) 
            print("\nctrl+c pressed, aborting...\n")
            servers.kill() 
            return
        # 正常关闭
        env.close()
        eval_env.close()
        servers.kill()


    def test(self, args):
        # 测试用的服务端
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Kick(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)
        try:    # 保存权重文件以便调用
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False) 
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()
        env.close()
        server.kill()
