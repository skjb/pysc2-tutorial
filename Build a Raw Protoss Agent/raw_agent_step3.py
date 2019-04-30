import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class RawAgent(base_agent.BaseAgent):
  def __init__(self):
    super(RawAgent, self).__init__()
    self.base_top_left = None

  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
    
  def step(self, obs):
    super(RawAgent, self).step(obs)
  
    if obs.first():
      nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)[0]
      self.base_top_left = (nexus.x < 32)
    
    return actions.RAW_FUNCTIONS.no_op()


def main(unused_argv):
  agent = RawAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="Simple64",
          players=[sc2_env.Agent(sc2_env.Race.protoss), 
                   sc2_env.Bot(sc2_env.Race.protoss, 
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              action_space=actions.ActionSpace.RAW,
              use_raw_units=True,
              raw_resolution=64,
          ),
      ) as env:
        run_loop.run_loop([agent], env)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
  