import dgl
import torch
from miscellaneous import *
import random
import networkx as nx
from math_utils import euclidean_distance, cartesian
from scipy import spatial

class City:
    def __init__(self,
                 G: dgl.DGLGraph,
                 speed_info=None,
                 name='simple_city',
                 consider_speed=False,
                 verbose=False,
                 after_action_random=True,
                 **kwargs
                 ):
        '''
        RL environment for road network.
        :param G: line graph of road graph.
        :param speed_info: SpeedInfo object
        :param name: name for city
        :param verbose: print debugging message
        :param after_action_random: after action, put driver on random position or not.
        '''

        self.name = name
        self.roads: list[Road] = []
        self.drivers: list[Person] = []
        self.city_time = 0
        self.city_time_unit_in_minute = 1
        self.updated_plan_interval = 15

        self.G = G
        
        places = []
        self.idx_ids = []
        for node, data in G.nodes(data=True):
            idx_ids.append(node)
            coordinates = [data["x"], data["y"]]
            cartesian_coord = cartesian(*coordinates)
            places.append(cartesian_coord)
        self.tree = spatial.KDTree(places)

        self.N = G.number_of_nodes()
        self.total_agent = len(self.drivers)
        pops = []
        self.road_key_dict = {}
        for i in range(self.N):
            road = Road(i, **self.G.nodes[i].data)
            self.roads.append(road)
            self.road_key_dict[(road.u, road.v)] = road.uuid
        self.actionable_drivers: list[Person] = []
        self.non_actionable_drivers: list[Person] = []
        self.epsilon = 0
        self.seed = 0
        self.random_seed = True
        self.driver_initializer = driver_initializer
        self.speed_info = speed_info
        self.verbose = verbose
        self.after_action_random = after_action_random

    def get_observation(self):
        #depends on road
        # obs = torch.zeros((self.N, 3),device=torch.device('cuda'))
        # for i in range(self.N):
        #     obs[i][0] = len(self.roads[i].demands)
        #     obs[i][1] = len(self.roads[i].drivers)
        #     obs[i][2] = self.roads[i].speed / 24
        # return obs
        obs = torch.zeros((self.total_agent, 2),device=torch.device('cuda'))
        for driver in self.drivers:
            obs[i][0] = int(driver.is_online())
            obs[i][1] = driver.plan.activities[driver.current_activity].legs[0].score if driver.is_online() else 100
        return obs

    def set_speed(self):
        if self.speed_info is not None:
            self.speed_info.set_speed(self)

    def get_road(self, u, v):
        road_id = self.road_key_dict[(u, v)]
        return road_id

    def update_drivers_status(self):
        '''
        Check whether driver has arrived
        :return:
        '''
        for driver in self.drivers:
            if driver.is_online() and driver.on_road:
                act=driver.plan.activities[driver.current_activity]
                if(driver.road_index==act.legs[act.selected_leg].end_link and driver.road_position>=act.legs[act.selected_leg].end_position):
                    driver.on_road=False #arrived
                    driver.current_activity+=1 #do another activity

    def get_actionable_drivers(self):
        '''
        get actionable / non actionable drivers
        :return: list of actionable / non actionable drivers
        '''
        actionable_drivers = []
        actionable_drivers_count = 0
        non_actionable_drivers = []
        non_actionable_drivers_count = 0
        for driver in self.drivers:
            if driver.on_road:
                if(driver.movable_time > 0):
                    road = self.roads[driver.road_index]
                    left_distance = road.length - driver.road_position
                    road_speed_in_meter_per_min = road.speed * 1000 / 60
                    time_to_finish = left_distance / road_speed_in_meter_per_min
                    
                    if time_to_finish > driver.movable_time:
                        driver.road_position += driver.movable_time * road_speed_in_meter_per_min
                        driver.movable_time = 0
                    else:
                        driver.movable_time -= time_to_finish
                        if driver.road_index+1 < len(driver.route):
                            next_road_index = self.road_key_dict[driver.route[driver.road_index+1]]
                        else:
                            next_road_index = None
                        if(next_road_index!=None):
                            self.roads[self.road_key_dict[driver.route[driver.road_index]]].drivers.remove(driver)
                            self.roads[next_road_index].drivers.append(driver)
                            driver.road_index+=1
                            if self.after_action_random:
                                driver.road_position = np.random.random() * self.roads[next_road_index].length
                            else:
                                road_speed_in_meter_per_min = self.roads[next_road_index].speed * 1000.0 / 60.0
                                x = max(0.0, driver.movable_time - 0.3) * road_speed_in_meter_per_min
                                max_x = 0.9 * self.roads[next_road_index].length
                                min_x = 0.1 * self.roads[next_road_index].length
                                x = max(min(x, max_x), min_x)
                                next_position_ratio += (x / (self.roads[next_road_index].length + 0.01))
                                total_counts += 1
                                driver.road_position = x
                        driver.movable_time = 0
                        
                non_actionable_drivers.append(driver)
                non_actionable_drivers_count += 1
            else:
                actionable_drivers.append(driver)
                actionable_drivers_count += 1

        if self.verbose:
            print("Actionable driver number :", sum(actionable_drivers_count))
            print("Non-Actionable driver number :", sum(non_actionable_drivers_count))

        return actionable_drivers, non_actionable_drivers

    def routing(self,start_point,end_point, k_num = 3):
        cartesian_coord_a = cartesian(start_point[0], start_point[1])
        cartesian_coord_b = cartesian(end_point[0], end_point[1])
        distance = euclidean_distance(cartesian_coord_a, cartesian_coord_b)
        closest_a = self.tree.query([cartesian_coord_b], p=2)
        closest_b = self.tree.query([cartesian_coord_a], p=2)

        def dist(a, b):
            c_a = cartesian(self.G.nodes[a]["x"], self.G.nodes[a]["y"])
            c_b = cartesian(self.G.nodes[b]["x"], self.G.nodes[b]["y"])
            return euclidean_distance(c_a, c_b)

        # sp = nx.astar_path(G, idx_ids[closest_a[1][0]], idx_ids[closest_b[1][0]], heuristic=dist, weight="time")
        temps = []
        found_paths = []
        for i in range(k_num):
            if i == 0:
                sp = nx.astar_path(self.G, self.idx_ids[closest_a[1][0]], self.idx_ids[closest_b[1][0]], heuristic=dist, weight="length")
            else:
                sp = nx.astar_path(self.G, self.idx_ids[closest_a[1][0]], self.idx_ids[closest_b[1][0]], heuristic=dist, weight="time")
            pathGraph = nx.path_graph(sp)
            paths = pathGraph.edges()
            found_paths.append(paths)
            if i > 0:
                temps.append(list(paths))
                # mutate time
                for path in paths:
                    u, v = path
                    self.G[u][v]["time"] = 99
        # restore time
        for p in list(set(sum(temps, []))):
            u, v = p
            self.G[u][v]["time"] = (G[u][v]["length"] / G[u][v]["avg_speed"]) * 60
        return found_paths
    def apply_policy(self, policy):
        '''
        Apply policy to controllable agents
        :param policy: list of policy for all agents.
        :return:
        '''
        #apply action for actionable drivers
        next_position_ratio = 0
        total_counts = 0
        
        action_threshold=0.5

        for index,driver in enumerate(self.city.drivers):
            if driver.is_online():
                # uniformly random (probability of epsilon)
                if policy is None or (self.epsilon > 0 and np.random.binomial(1, self.epsilon) == 1):
                    action = np.random.choice([0,1])
                # random from stochastic policy (probability of 1 - epsilon)
                else:
                    #greedy
                    action = int(np.max(policy[index])>action_threshold)
                if action:
                    if(driver.current_activity<len(driver.plan.activities)):
                        #let's fucking go!
                        driving_legs = driver.plan.activities[driver.current_activity+1].legs
                        # randomly selecting a leg
                        # TODO selecting a leg with no side effect to the network
                        selected_leg=np.random.choice(len(driving_legs))
                        driver.plan.activities[driver.current_activity+1].selected_leg=selected_leg
                        driver.route = driving_legs[selected_leg]
                        driver.on_road=True
                        #teleported to the first road on random location
                        driver.road_index = 0
                        driver.road_position = np.random.random() * self.roads[self.road_key_dict[driver.route[driver.road_index]]].length
                

        if self.verbose and not self.after_action_random:
            print("After movement position ratio average:", next_position_ratio / total_counts)

    def current_total_call_number(self):
        n = 0
        for road in self.roads:
            n += len(road.calls)
        return n

    def current_total_driver_number(self):
        online = 0
        available = 0
        offline = 0
        for driver in self.drivers:
            if driver.on_road:
                online += 1
            elif driver.is_online(self.city_time):
                offline +=1
            else:
                available += 1
        return online + available + offline, online, available, offline


    def reset(self):
        '''
        Clear all drivers, calls
        :return:
        '''
        self.city_time = 0
        for road_index in range(self.N):
            road = self.roads[road_index]
            road.drivers.clear()
            road.demands.clear()
        self.drivers.clear()

    # def get_next_driver_id(self):
    #     self.driver_uuid += 1
    #     return self.driver_uuid - 1

    def initialize(self):
        '''
        Generate agents
        :return: initial state
        '''
        #TODO
        # generate idle drivers
        # for road_index in range(self.N):
        #     number_of_drivers = int(driver_distribution[road_index]) #np.random.choice([0,1,2,3],p=[0.5,0.2,0.2,0.1])
        #     road = self.roads[road_index]
        #     for _ in range(number_of_drivers):
        #         driver = Driver(self.get_next_driver_id(), road_index, np.random.random() * road.length)
        #         road.drivers.append(driver)
        #         self.drivers.append(driver)

        # print("City initialized with total %d drivers" % len(self.drivers))
        self.set_speed()
        return self.get_observation()
    

    def charge_drive_time(self):
        for driver in self.drivers:
            if not driver.is_online() and driver.on_road:
                driver.movable_time = self.city_time_unit_in_minute
    
    def generate_plan(self):
        for driver in self.drivers:
            if driver.is_online():
                current_act=driver.driver.plan.activities[driver.current_activity]
                source=current_act.location
                destination=driver.plan.activities[driver.current_activity+1].location
                paths=self.routing(source, destination)
                current_act.legs=paths
                

    def step(self, policy):
        '''
        Single update cycle.
        :param policy: list of policy for all roads.
        :return: next state, assigned call number, missed call number
        '''
        
        self.charge_drive_time() #riders move
        self.apply_policy(policy)
        self.actionable_drivers, self.non_actionable_drivers = self.get_actionable_drivers()
        self.city_time += 1

        t, a, b, c = self.current_total_driver_number()
        if self.verbose:
            print(self.city_time)
            print("Total driver %d, on_road %d, available %d , do_act %d" % (t, a, b, c))

        # before = self.current_total_call_number()
        # self.update_old_calls()
        # after = self.current_total_call_number()
        # missed_call_number = before - after

        if self.city_time%self.updated_plan_interval==0:
            #update plan for online drivers
            self.generate_plan()
        self.update_drivers_status()
        self.set_speed()
        next_state = self.get_observation()
        # return next_state, assigned_call_number, missed_call_number
        return next_state