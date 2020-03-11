from absl import app
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
from terran_agent import TerranAgent
import random
import csv


class BattleManager:
  predictor_marines = 0
  enemy_marines = 0
  predictor_ready = False
  enemy_ready = False
  
  
class EnemyAgent(TerranAgent):
  def __init__(self, bm):
    super(EnemyAgent, self).__init__()
    self.bm = bm

  def step(self, obs):
    super(EnemyAgent, self).step(obs)

    if obs.first():
      self.bm.enemy_ready = False
      self.bm.enemy_marines = random.randint(1, 10)
      print("enemy", self.bm.enemy_marines)

    if len(self.marines) == self.bm.enemy_marines and not self.bm.enemy_ready:
      print("enemy ready")
      self.bm.enemy_ready = True
          
    if self.bm.predictor_ready and self.bm.enemy_ready:
      return self.attack()
    if self.supply_depot is None:
      return self.build_supply_depot()
    if self.barracks is None:
      return self.build_barracks()
    if len(self.marines) + self.queued_marine_count < self.bm.enemy_marines:
      return self.train_marine()
    return actions.RAW_FUNCTIONS.no_op()


class PredictorAgent(TerranAgent):
  def __init__(self, bm):
    super(PredictorAgent, self).__init__()
    self.bm = bm

  def step(self, obs):
    super(PredictorAgent, self).step(obs)

    if obs.first():
      self.bm.predictor_ready = False
      self.bm.predictor_marines = random.randint(1, 10)
      print("predictor", self.bm.predictor_marines)

    if obs.last():
      print(self.bm.predictor_marines, self.bm.enemy_marines, obs.reward)
      with open("tvt.csv", "a", newline="\n") as myfile:
        csvwriter = csv.writer(myfile)
        csvwriter.writerow([self.bm.predictor_marines,
                            self.bm.enemy_marines,
                            obs.reward])

    if (len(self.marines) == self.bm.predictor_marines and
        not self.bm.predictor_ready):
      print("predictor ready")
      self.bm.predictor_ready = True
      
    if self.bm.predictor_ready and self.bm.enemy_ready:
      return self.attack()
    if self.supply_depot is None:
      return self.build_supply_depot()
    if self.barracks is None:
      return self.build_barracks()
    if len(self.marines) + self.queued_marine_count < self.bm.predictor_marines:
      return self.train_marine()
    return actions.RAW_FUNCTIONS.no_op()


def main(unused_argv):
  bm = BattleManager()
  agent1 = PredictorAgent(bm)
  agent2 = EnemyAgent(bm)
  try:
    with sc2_env.SC2Env(
        map_name="Flat128",
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
        ),
        step_mul=128,
        disable_fog=True,
    ) as env:
      run_loop.run_loop([agent1, agent2], env, max_episodes=20)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
