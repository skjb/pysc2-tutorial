import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class RawAgent(base_agent.BaseAgent):
  def step(self, obs):
    super(RawAgent, self).step(obs)
    
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
