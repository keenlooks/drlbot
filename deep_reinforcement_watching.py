# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------
import sys

if len(sys.argv) < 2:
    print "please provide path to model"
    quit()

import MalmoPython
import random
import time
import logging
import struct
import socket
import os
import json
import numpy as np
from DRLBot import DRLBot
from time import sleep
from tqdm import *

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)#DEBUG) # set to INFO if you want fewer messages

# create a file handler
#handler = logging.FileHandler('depthmaprunner.log')
#handler.setLevel(logging.DEBUG)

# create a logging format
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)

# add the handlers to the logger
#logger.addHandler(handler)

#-------------------------------------------------------------------------------------------------------------------------------------

    
#----------------------------------------------------------------------------------------------------------------------------------

video_width = 192
video_height = 192
channels = 3

action_strings = ["move","strafe","pitch","turn","jump","crouch","attack","use"]    
actions = [[1,0,0,0,0,0,0,0]   # 0 - move forward
          ,[-1,0,0,0,0,0,0,0]  # 1 - move backward
          ,[1,0,0,0,1,0,0,0]   # 2 - move forward and jump
          ,[0,0,0,1,0,0,0,0]   # 3 - turn right
          ,[0,0,0,-1,0,0,0,0]  # 4 - turn left
          ,[0,0,0,0,0,0,1,0]   # 5 - attack
          ,[0,0,0,0,0,0,0,1]   # 6 - use
          ,[0,0,0,0,1,0,0,0]   # 7 - jump
          ,[0,0,1,0,0,0,0,0]   # 8 - look down
          ,[0,0,-1,0,0,0,0,0]  # 9 - look up
          ,[0,0,0,0,0,0,0,0]   # 10 - do nothing
           ]
missionXML = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    
      <About>
        <Summary>Survive!</Summary>
      </About>
     <ModSettings>
        <MsPerTick>50</MsPerTick>
        <PrioritiseOffscreenRendering>false</PrioritiseOffscreenRendering>
     </ModSettings>
     <ServerSection>
        <ServerInitialConditions>
          <Time>
            <StartTime>6000</StartTime>
            <AllowPassageOfTime>false</AllowPassageOfTime>
          </Time>
          <Weather>clear</Weather>
        </ServerInitialConditions>
        <ServerHandlers>
            <DefaultWorldGenerator/>
            <ServerQuitFromTimeUp timeLimitMs="600000"/>
            <ServerQuitWhenAnyAgentFinishes />
        </ServerHandlers>
    </ServerSection>

    <AgentSection mode="Survival">
        <Name>Jason Bourne</Name>
        <AgentStart>
            <Placement x="0" y="80" z="0"/>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <VideoProducer want_depth="false">
                <Width>''' + str(video_width) + '''</Width>
                <Height>''' + str(video_height) + '''</Height>
            </VideoProducer>
            <ContinuousMovementCommands turnSpeedDegs="90" />
            <RewardForMissionEnd rewardForDeath="-100" dimension="0">
                 <Reward reward="-100" description="death" />
            </RewardForMissionEnd>
        </AgentHandlers>
    </AgentSection>
  </Mission>'''

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

validate = True
my_mission = MalmoPython.MissionSpec( missionXML, validate )

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.KEEP_ALL_FRAMES)

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000

my_mission_record_spec = MalmoPython.MissionRecordSpec()

where_i_have_been_before = []

#function that gets agents x,y,z cooordinates from JSON observations
def get_agent_location():
    world_state = agent_host.peekWorldState()
    if len(world_state.observations) < 1:
        return None
    obs_text = world_state.observations[0].text
    obs = json.loads(obs_text) # most recent observation
    logger.debug(obs)
    if not 'XPos' in obs or not 'YPos' or not 'ZPos' in obs:
        logger.error("Incomplete observation received: %s" % obs_text)
        return None
    logger.debug("State: %s (x = %.2f, z = %.2f)" % ("%d:%d" % (int(obs['XPos']), int(obs['ZPos'])), float(obs[u'XPos']), float(obs[u'ZPos'])))
    return (float(obs['XPos']), float(obs['YPos']), float(obs['ZPos']))

def is_agent_in_new_area():
    curr_loc = get_agent_location()
    if curr_loc is None:
        return False
    
    for loc in where_i_have_been_before:
        #the [::2] makes it only use the XPos and ZPos variables so movement in height (falling or jumping) does not get it a reward
        if np.linalg.norm(np.array(curr_loc[::2])-np.array(loc[::2])) < 5:
            return False
    where_i_have_been_before.append(curr_loc)
    return True
    
def get_minecraft_frame():
    world_state = agent_host.peekWorldState()
    while len(world_state.video_frames) < 1 and world_state.is_mission_running:
        logger.info("Waiting for frames...")
        time.sleep(0.001)
        world_state = agent_host.peekWorldState()
    if not world_state.is_mission_running:
        return None
    logger.info("Got frame!")
    pixels = agent_host.peekWorldState().video_frames[0].pixels
    #red,green,blue = pixels[0::4],pixels[1::4],pixels[2::4]
    #pixels_averaged = [int(np.mean([re,ge,be])) for re,ge,be in zip(red,green,blue)]
    #green.extend(blue)
    #red.extend(green)
    #pixels_no_depth = red
    return np.reshape(pixels,(channels,video_width,video_height),order='F')
    #return np.reshape(np.mat(green,dtype=np.float32),(video_width,video_height),order='C')

    
def make_minecraft_action(action_i):
    for action_tuple in zip(action_strings,actions[action_i]):
        agent_host.sendCommand(action_tuple[0]+" "+str(action_tuple[1]))
        logger.info("doing "+action_tuple[0]+" "+str(action_tuple[1]))
    total_reward = 0
    rewards = agent_host.peekWorldState().rewards
    for reward in rewards:
        total_reward += reward.getValue()
    if is_agent_in_new_area():
        total_reward += 5
   #print "reward for this move: "+str(total_reward)
    return total_reward
 
drlbot = DRLBot(get_state_f=get_minecraft_frame,make_action_f=make_minecraft_action,available_actions_num=len(actions))
drlbot.load_model(sys.argv[1])


start = time.time()
agent_host.startMission( my_mission, my_mission_record_spec )
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
print
my_mission.setTimeOfDay(6000, False)
total_reward = 0
while world_state.is_mission_running:
    world_state = agent_host.getWorldState()
    if world_state.is_mission_running:
        reward_for_move = drlbot.perform_play_step_no_storage()
        if reward_for_move != 0:
            print "Received reward: "+str(reward_for_move)
            total_reward += reward_for_move
print "Time to Die: ",time.time()-start, "Total Reward: ",total_reward