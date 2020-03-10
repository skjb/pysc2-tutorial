from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random


class TerranAgent(base_agent.BaseAgent):
  def build_supply_depot(self):
    if (self.supply_depot is None and
        self.obs.observation.player.minerals >= 100 and len(self.scvs) > 0):
      base_x, base_y = self.base_xy
      supply_depot_xy = (base_x + 3 if base_x < 32 else base_x - 3, base_y)
      scv = random.choice(self.scvs)
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
          "now", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()
    
  def build_barracks(self):
    if (self.completed_supply_depot and self.barracks is None and 
        self.obs.observation.player.minerals >= 150 and len(self.scvs) > 0):
      base_x, base_y = self.base_xy
      barracks_xy = (base_x, base_y + 3 if base_y < 32 else base_y - 3)
      scv = random.choice(self.scvs)
      return actions.RAW_FUNCTIONS.Build_Barracks_pt(
          "now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()

  def train_marine(self):
    if (self.completed_barracks and
        self.obs.observation.player.minerals >= 100 and self.free_supply > 0):
      if self.barracks.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Marine_quick(
            "now", self.barracks.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def attack(self):
    if len(self.marines) > 0:
      if len(self.enemy_army_units) > 0:
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", [unit.tag for unit in self.marines], (31, 31))
      elif len(self.enemy_buildings) > 0:
        enemy_building = self.enemy_buildings[0]
        return actions.RAW_FUNCTIONS.Attack_unit(
          "now", [unit.tag for unit in self.marines], enemy_building.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def step(self, obs):
    super(TerranAgent, self).step(obs)
    
    self.obs = obs
    self.free_supply = (obs.observation.player.food_cap - 
                        obs.observation.player.food_used)

    self.command_center = None
    self.supply_depot = None
    self.completed_supply_depot = False
    self.barracks = None
    self.completed_barracks = False
    self.marines = []
    self.scvs = []
    self.queued_marine_count = 0
    
    self.enemy_buildings = []
    self.enemy_army_units = []

    for unit in obs.observation.raw_units:
      if unit.alliance == features.PlayerRelative.SELF:
        if unit.unit_type == units.Terran.Barracks:
          self.barracks = unit
          if unit.build_progress == 100:
            self.completed_barracks = True
        elif unit.unit_type == units.Terran.CommandCenter:
          self.command_center = unit
        elif unit.unit_type == units.Terran.Marine:
          self.marines.append(unit)
        elif unit.unit_type == units.Terran.SCV:
          self.scvs.append(unit)
        elif unit.unit_type == units.Terran.SupplyDepot:
          self.supply_depot = unit
          if unit.build_progress == 100:
            self.completed_supply_depot = True

        for i in range(0, 4):
          order_id = unit["order_id_" + str(i)]
          if order_id == 511:
            self.queued_marine_count += 1

      elif unit.alliance == features.PlayerRelative.ENEMY:
        if unit.unit_type in [units.Terran.Barracks,
                              units.Terran.CommandCenter,
                              units.Terran.SupplyDepot]:
          self.enemy_buildings.append(unit)
        elif unit.unit_type == units.Terran.Marine:
          self.enemy_army_units.append(unit)

    if obs.first():
      self.base_xy = (self.command_center.x, self.command_center.y)
