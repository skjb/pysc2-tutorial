from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app

class ZergAgent(base_agent.BaseAgent):
  def step(self, obs):
    super(ZergAgent, self).step(obs)
    
    return actions.FUNCTIONS.no_op()
