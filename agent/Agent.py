from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # =================================================================================================
        # LÓGICA DE DEFINIÇÃO DE TIPO DE ROBÔ (vFinal - Ajustada)
        # Goleiro (1) = tipo 0
        # Zagueiro (2) = tipo 3
        # Meio-campistas (3-9) = tipo 1 (total de 7)
        # Atacantes (10-11) = tipo 2
        # =================================================================================================
        if unum == 1:
            robot_type = 0
        elif unum == 2:
            robot_type = 3
        elif 3 <= unum <= 9:
            robot_type = 1
        else: # unum 10, 11
            robot_type = 2
        # =================================================================================================

        # Inicialização do agente base
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        # Formação Inicial Tática
        self.init_pos = ([-14,0],[-10,-4],[-10,0],[-10,4],[-6,-5],[-6,0],[-6,5],[-2,-3],[-2,3],[2, 5],[2, -5])[unum-1]

        # =================================================================================================
        # ATRIBUIÇÃO DE PAPÉIS TÁTICOS (AJUSTADA)
        # =================================================================================================
        if unum == 1:
            self.role = "GOALKEEPER"
        elif 2 <= unum <= 4:
            self.role = "DEFENDER"
        elif 5 <= unum <= 9:
            self.role = "MIDFIELDER"
        else: # 10 e 11
            self.role = "ATTACKER"
        # =================================================================================================

    # =================================================================================================
    # FUNÇÕES DE APOIO À ESTRATÉGIA
    # =================================================================================================
    def is_kick_viable(self, goal_direction, ball_position_2d):
        """Verifica se um chute direto ao gol é uma boa opção."""
        angle_to_goal = abs(M.normalize_deg(goal_direction - self.world.robot.imu_torso_orientation))
        if angle_to_goal > 60:
            return False

        goal_center = (15.0, 0.0)
        distance_to_goal = np.linalg.norm(goal_center - ball_position_2d)
        if distance_to_goal > 22:
            return False

        if self.min_opponent_ball_dist < 3.0:
            return False

        return True

    def find_best_pass_target(self):
        """Encontra o melhor atacante livre para um passe longo."""
        best_teammate_pos = None
        max_x = 5

        # Itera sobre os ATACANTES (unums 10 e 11)
        for unum_atacante in range(10, 12):
            teammate = self.world.teammates[unum_atacante - 1]

            if teammate.state_last_update > 0 and not teammate.state_fallen:
                teammate_pos = teammate.state_abs_pos[:2]
                
                if teammate_pos[0] > max_x:
                    is_marked = False
                    for opponent in self.world.opponents:
                        if opponent.state_last_update > 0:
                            dist_to_opponent = np.linalg.norm(teammate_pos - opponent.state_abs_pos[:2])
                            if dist_to_opponent < 3.5:
                                is_marked = True
                                break
                    
                    if not is_marked:
                        max_x = teammate_pos[0]
                        best_teammate_pos = teammate_pos
        
        return best_teammate_pos
    # =================================================================================================
    
    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] 
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            
            # --- CORREÇÃO DEFINITIVA (Baseada no Bug Report) ---
            # Calcula a rotação ideal para o robô olhar para o centro do campo
            target_rotation = M.vector_angle((-pos[0], -pos[1]))
            
            # Arredonda o ângulo para 5 casas decimais para uma comparação mais estável
            rounded_rotation = round(target_rotation, 5)

            # Verifica se o ângulo arredondado é um múltiplo de 90 graus.
            # Esta é a causa documentada do crash no motor de física ODE.
            if rounded_rotation % 90.0 == 0.0:
                target_rotation += 0.001
            
            # Executa o beam com a rotação segura
            self.scom.commit_beam(pos, target_rotation)
            # --- FIM DA CORREÇÃO ---

        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)
    
    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", True, target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)

    def kick(self, kick_direction=None, kick_distance=18.0, abort=False, enable_pass_command=False):
        """
        Executa o comportamento de chute apropriado com base no tipo de robô.
        """
        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance
        
        if self.fat_proxy_cmd is not None:
            return self.fat_proxy_kick()

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        # Robôs TIPO 1 (meio-campistas) usam o chute treinado
        if self.world.robot.type == 1:
            return self.behavior.execute("Kick", True, self.kick_direction)
        # Todos os outros tipos usam o chute básico
        else:
            return self.behavior.execute("Basic_Kick", True, self.kick_direction, abort)


    def think_and_send(self):
        w = self.world
        r = self.world.robot  
        my_head_pos_2d = r.loc_head_position[:2]
        ball_2d = w.ball_abs_pos[:2]
        goal_dir = M.target_abs_angle(ball_2d,(15.05,0))
        PM = w.play_mode
        PM_GROUP = w.play_mode_group

        slow_ball_pos = w.get_predicted_ball_pos(0.5)
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2) if p.state_last_update > 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen else 1000 for p in w.teammates]
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2) if p.state_last_update > 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen else 1000 for p in w.opponents]
        
        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist))

        team_has_possession = min_teammate_ball_sq_dist < (self.min_opponent_ball_dist**2 - 0.5)
        active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1

        if PM == w.M_GAME_OVER:
            pass
        elif PM_GROUP == w.MG_ACTIVE_BEAM or PM_GROUP == w.MG_PASSIVE_BEAM:
            self.beam()
        elif self.state == 1 or (self.behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if self.behavior.execute("Get_Up") else 1
            
        elif self.role == "GOALKEEPER":
            self.move(self.init_pos, orientation=M.vector_angle(ball_2d - my_head_pos_2d))

        elif self.role == "DEFENDER":
            if active_player_unum == r.unum:
                 self.kick(goal_dir)
            else:
                new_x = max(-14, 0.6 * (ball_2d[0] + 15) + self.init_pos[0] * 0.4 - 9)
                self.move((new_x, self.init_pos[1]), orientation=M.vector_angle(ball_2d - my_head_pos_2d))
        
        elif self.role == "MIDFIELDER":
            if not team_has_possession and ball_2d[0] < 12:
                self.move(slow_ball_pos, is_aggressive=True)
            else:
                if active_player_unum == r.unum:
                    if self.is_kick_viable(goal_dir, ball_2d):
                        self.kick(goal_dir)
                    else:
                        pass_target = self.find_best_pass_target()
                        if pass_target is not None:
                            pass_direction = M.vector_angle(pass_target - ball_2d)
                            self.kick(pass_direction)
                        else:
                            self.behavior.execute("Dribble", True, None, None)
                else:
                    new_x = max(-7, 0.7 * (ball_2d[0] + 15) + self.init_pos[0] * 0.3 - 6)
                    self.move((new_x, self.init_pos[1]), orientation=M.vector_angle(ball_2d - my_head_pos_2d))

        elif self.role == "ATTACKER":
            if team_has_possession and active_player_unum == r.unum:
                self.kick(goal_dir)
            else:
                pos_x = max(self.init_pos[0], ball_2d[0] - 1.5)
                self.move((pos_x, self.init_pos[1]), orientation=M.vector_angle(ball_2d - my_head_pos_2d))

        self.radio.broadcast()
        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send( r.get_command() )
        else:
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() )
            self.fat_proxy_cmd = ""

    # =================================================================================================
    # MÉTODOS AUXILIARES PARA O FAT PROXY
    # =================================================================================================
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot
        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += "(proxy dash 100 0 0)"
            return

        if target_dist < 0.1:
            if is_orientation_absolute and orientation is not None:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += f"(proxy dash 0 0 {target_dir:.1f})"
        else:
            self.fat_proxy_cmd += f"(proxy dash 20 0 {target_dir:.1f})"
    # =================================================================================================