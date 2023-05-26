import numpy as np


class Road:
    def __init__(self, uuid, **kwargs):
        self.uuid = uuid
        self.length = kwargs['length'] # meter
        self.drivers = []
        # u, v is an id of start, end node.
        self.u = kwargs['u'].item()
        self.v = kwargs['v'].item()
        self.speed = kwargs['avg_speed'] # km/h
        self.free_speed = kwargs['free_speed']
        
class Activity:
    def __init__(self, activity_type, activity_start_time, activity_end_time, activity_location):
        self.type: str = activity_type
        self.end_time: int = activity_end_time
        self.start_time: int = activity_start_time
        self.location = activity_location
        self.score: float = 0
        self.selected_leg: int = None
        self.legs : list[Leg] = [] #mode departure_time travel_time

class Leg:
    def __init__(self, leg_mode, dep_time, trav_time):
        self.mode: str = leg_mode
        self.dep_time: int = dep_time
        self.trav_time: int = trav_time
        self.selected: bool = False
        self.route: list = []
        self.end_position: float = end_position
        self.distance: float = distance
        self.score: float = 0
        
class Plan:
    def __init__(self, person: Person):
        self.p = person
        self.activities: list[Activity]=[] #type end_time
        self.create_hwh()
    def create_hwh(self):
        self.activities.append(Activity("home", None, None, self.p.home_location ))
        self.activities.append(Activity("work", 60*8, 60*8+60*9, self.p.work_location ))
        self.activities.append(Activity("work", None, None, self.p.home_location ))
        
class Person:
    def __init__(self, vId, age, gender, income, home_location, work_location):
        self.id = vId
        self.home_location = home_location
        self.age = age
        self.gender = gender
        self.work_location = work_location
        self.income = income
        self.plan: Plan = Plan(self)
        self.current_activity=0
        #road case
        self.movable_time = 0
        self.road_index = None
        self.road_position = None
        self.route: list = None
        self.on_road=False
    def is_online(self, current_time):
        act_end_time = self.plan.activities[self.current_activity].end_time
        if act_end_time is not None:
            if(act_end_time>current_time):
                return False #still doing act
        elif on_road is True:
            return False #still driving
        elif self.current_activity==len(self.plan.activities)-1:
            return False #returned home
        return True #free
        