from scripts.commons.Script import Script
script = Script(cpp_builder_unum=1) # Initialize: load config file, parse arguments, build cpp modules
a = script.args

from agent.Agent import Agent

player = Agent(a.i, a.p, None, a.u, a.t, False, False, False, a.F)

while True:
    if(player.behavior.is_ready('Get_Up')):
        while(not player.behavior.execute('Get_Up')):
            player.scom.commit_and_send(player.world.robot.get_command())
            player.scom.receive()
    
    player.behavior.execute('Kick', 0)
    player.scom.commit_and_send(player.world.robot.get_command())
    player.scom.receive()
