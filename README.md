# 基于FC Portugal Codebase的踢球训练环境  

![](https://github.com/GrayJimars/FCP-Kick/blob/main/Kick.gif?raw=true)

## 准备
1. 配置训练环境（列表仅供参考，因为包随时会更新）：  
```
conda               25.1.1  
python              3.12.9  
```
```
numpy               2.2.5  
pybind11            2.13.6  
psutil              7.0.0  
stable-baselines3   2.6.0  
gym                 0.26.2  
shimmy              2.0.0   
```
  
2. **仔细阅读**以下文件中的**代码和注释**   
[scripts/gyms/Kick.py](https://github.com/GrayJimars/FCP-Kick/tree/main/scripts/gyms/Kick.py)  

3. 由于**参数并非最优**，请尝试修改参数提高上限，见**期望**

## 训练
注意机器人类型
```
python Run_Utils.py -r 1  
```
```
Kick  
```
```
Train——从头训练  
Test——导出.pkl  
Retrain——继续训练  
```
## 使用
1. 导出.pkl文件  

2. **仔细阅读**以下文件中的**代码和注释**   
[behaviors/custom/Kick/Env.py](https://github.com/GrayJimars/FCP-Kick/tree/main/behaviors/custom/Kick/Env.py)  
[behaviors/custom/Kick/Kick.py](https://github.com/GrayJimars/FCP-Kick/tree/main/behaviors/custom/Kick/Kick.py)   

3. 编写Your_Kick.py和Your_Env.py  
4. 在[behaviors/Behavior.py](https://github.com/GrayJimars/FCP-Kick/tree/main/behaviors/Behavior.py)中添加行为  
5. 参考Run_Kick_Test.py进行测试  
## 预期  
目标距离为18米，机器人类型为1
```
精度 = (奖励/最高奖励)**(1/PRECISE_INDEX)  
误差 = 目标距离*(1-精度)  
```
> PRECISE_INDEX=2，n_steps_per_env=128，minibatch_size=64，learning_rate=3e-4  

| 步数 | 最大奖励 | 最大精度 | 最小误差 | 
| --- | --- | --- | --- | 
| 1024000 | 9.76 | 57% | 7.73m | 
| 2048000 | 13.16 | 66% | 6.07m | 
| 5120000 | 19.77 | 81% | 3.39m | 
| 10240000 | 22.72 | 87% | 2.34m |   

> PRECISE_INDEX=3，n_steps_per_env=128，minibatch_size=64，learning_rate=3e-4  

| 步数 | 最大奖励 | 最大精度 | 最小误差 | 
| --- | --- | --- | --- | 
| 15360000 | 20.94 | 89% | 1.99m |

> PRECISE_INDEX=3，n_steps_per_env=256，minibatch_size=128，learning_rate=1e-4  

| 步数 | 最大奖励 | 最大精度 | 最小误差 | 
| --- | --- | --- | --- | 
| 20480000 | 24.22 | 93% | 1.24m |  
| 25600000 | 25.55 | 95% | 0.94m |  

25M步后，Test平均奖励为20（能踢15米以上），对于15米内的射门足够用了  
由于**参数并非最优**，请尝试修改参数提高上限  

# FC Portugal Codebase <br> for RoboCup 3D Soccer Simulation League

![](https://s5.gifyu.com/images/Siov6.gif)

## About

The FC Portugal Codebase was mainly written in Python, with some C++ modules. It was created to simplify and speed up the development of a team for participating in the RoboCup 3D Soccer Simulation League. We hope this release helps existing teams transition to Python more easily, and provides new teams with a robust and modern foundation upon which they can build new features.


## Documentation

The documentation is available [here](https://docs.google.com/document/d/1aJhwK2iJtU-ri_2JOB8iYvxzbPskJ8kbk_4rb3IK3yc/edit)

## Features

- The team is ready to play!
    - Sample Agent - the active agent attempts to score with a kick, while the others maintain a basic formation
        - Launch team with: **start.sh**
    - Sample Agent supports [Fat Proxy](https://github.com/magmaOffenburg/magmaFatProxy) 
        - Launch team with: **start_fat_proxy.sh**
    - Sample Agent Penalty - a striker performs a basic kick and a goalkeeper dives to defend
        - Launch team with: **start_penalty.sh**
- Skills
    - Get Ups (latest version)
    - Walk (latest version)
    - Dribble v1 (version used in RoboCup 2022)
    - Step (skill-set-primitive used by Walk and Dribble)
    - Basic kick
    - Basic goalkeeper dive
- Features
    - Accurate localization based on probabilistic 6D pose estimation [algorithm](https://doi.org/10.1007/s10846-021-01385-3) and IMU
    - Automatic head orientation
    - Automatic communication with teammates to share location of all visible players and ball
    - Basics: common math ops, server communication, RoboViz drawings (with arrows and preset colors)
    - Behavior manager that internally resets skills when necessary
    - Bundle script to generate a binary and the corresponding start/kill scripts
    - C++ modules are automatically built into shared libraries when changes are detected
    - Central arguments specification for all scripts
    - Custom A* pathfinding implementation in C++, optimized for the soccer environment
    - Easy integration of neural-network-based behaviors
    - Integration with Open AI Gym to train models with reinforcement learning
        - User interface to train, retrain, test & export trained models
        - Common features from Stable Baselines were automated, added evaluation graphs in the terminal
        - Interactive FPS control during model testing, along with logging of statistics
    - Interactive demonstrations, tests and utilities showcasing key features of the team/agents
    - Inverse Kinematics
    - Multiple agents can be launched on a single thread, or one agent per thread
    - Predictor for rolling ball position and velocity
    - Relative/absolute position & orientation of every body part & joint through forward kinematics and vision
    - Sample train environments
    - User-friendly interface to check active arguments and launch utilities & gyms

## Citing the Project

```
@article{abreu2023designing,
  title={Designing a Skilled Soccer Team for RoboCup: Exploring Skill-Set-Primitives through Reinforcement Learning},
  author={Abreu, Miguel and Reis, Luis Paulo and Lau, Nuno},
  journal={arXiv preprint arXiv:2312.14360},
  year={2023}
}
```
