from GCN import GCN,GAT
import torch.nn.functional as F
import torch
from city import City

from math_utils import softmax, softmax_pow
from graph_utils import *

POLICY_ARGMAX = 0
POLICY_POW = 1
POLICY_EXP = 2
POLICY_ENTROPY = 3

# class for different agent strategy
class Agent:
    def __init__(self, name):
        self.name = name
        self.do_epsilon_exploration = True
        self.gamma = 0.9

    def train(self, next_observations):
        pass

    def get_policy(self, observations):
        pass

    def set_eval_mode(self):
        pass

    def save_model(self, save_model_path):
        pass

    def load_model(self, load_model_path):
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__('random')

    def get_policy(self, observations):
        return None


class ProportionalAgent(Agent):
    def __init__(self, city: City, proportional='order', policy_pow=1, strategy=1,  **kwargs):
        t_name = 'proportional'
        temperature = kwargs.get("temperature", 1)
        if strategy == POLICY_ARGMAX:
            t_name = t_name + '_max_eps_%s' % (str(kwargs.get("epsilon_min", 0)))
        elif strategy == POLICY_POW:
            t_name = t_name + '_%s' % (str(policy_pow))
        elif strategy == POLICY_EXP:
            t_name = t_name + '_softmax_%s' % (str(temperature))
        super().__init__(t_name)

        self.city = city
        self.order_proportional = (proportional=='order')
        self.policy_pow = policy_pow
        self.strategy = strategy

    def get_policy(self, observations):
        policies = [[] for _ in range(self.city.total_agent)]
        for road in self.city.roads:
            policy = np.zeros((len(road.reachable_roads, )))
            for i, road_index in enumerate(road.reachable_roads):
                v = observations[road_index][1]
                if not self.order_proportional:
                    v = max(v-observations[road_index][0], 0)
                policy[i] = v
            if policy.sum() == 0:
                policy.fill(1)
            if self.strategy == 0:
                policy = np.where(policy == np.amax(policy), 1.0, 0.0)
                policy /= policy.sum()
            elif self.strategy == 1:
                policy /= policy.sum()
                if self.policy_pow != 1:
                    policy = softmax_pow(policy, self.policy_pow)
            else:
                policy = softmax(policy, self.policy_pow)

            policies[road.uuid] = policy
        return policies


class DQNAgent(Agent):
    def __init__(self, city: City, model_type='gcn', policy_pow=1.0, strategy=POLICY_POW, consider_speed=True, **kwargs):
        temperature = kwargs.get("temperature", 1)
        t_name = 'dqn'
        if strategy == POLICY_ARGMAX:
            t_name = t_name + '_%s_max_eps_%s' % (model_type, str(kwargs.get("epsilon_min", 0)))
        elif strategy == POLICY_POW:
            t_name = t_name + '_%s_%s' % (model_type, str(policy_pow))
        elif strategy == POLICY_EXP:
            t_name = t_name + '_%s_softmax_%s' % (model_type, str(temperature))
        elif strategy == POLICY_ENTROPY:
            t_name = t_name + '_%s_entropy_softmax_%s' % (model_type, str(temperature))
        super().__init__(t_name)

        # reverse direction & add self loop
        newG = city.G.reverse()
        for node in newG.nodes():
            newG.add_edges(node, node)
        newG = newG.to('cuda:0')
        self.strategy = strategy

        city.consider_speed = consider_speed

        if model_type == 'gcn':
            self.model = GCN(newG,
                             in_feats=2,
                             n_hidden=8,
                             n_classes=1,
                             n_layers=4,
                             activation=F.relu)
        else:
            self.model = GAT(newG,
                             in_dim=2,
                             num_hidden=8,
                             num_classes=1,
                             num_layers=4,
                             activation=F.relu)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        # define model and target model
        self.model.cuda()
        self.model.train()
        self.target_model = copy.deepcopy(self.model)
        self.target_model.cuda()
        self.target_model_update_period = 10

        self.time_step = 0
        self.city = city

        self.observations = None

        # Q_V(s, t)
        self.q_values = None

        # sigma pi(s,t) Q(s,t)
        self.next_target_expected_return_values = torch.zeros((self.city.total_agent,)).cuda()
        # for memoization
        self.next_target_expected_return_values_valid = np.zeros((self.city.total_agent,), dtype=np.int32)
        self.policy_pow = policy_pow
        self.do_epsilon_exploration = kwargs.get("do_epsilon_exploration", True)
        self.temperature = temperature

        if kwargs.get("q_value_debug", False):
            print("Debug file create")
            self.debug_file = open("%s/q_value_log_%s.txt" % (kwargs.get("log_save_folder"), self.name), 'w')
        else:
            self.debug_file = None
        self.q_values_saved = None

    def save_model(self, save_path):
        print("SAVING")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def set_eval_mode(self):
        self.model.eval()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, next_observations):
        if self.time_step % self.target_model_update_period == 0:
            self.update_target_model()

        with torch.no_grad():
            # update target for Q_V(s, t)
            target_q_values = torch.zeros(self.city.total_agent, 1).cuda()
            target_q_values_counts = torch.zeros(self.city.total_agent, 1).cuda()

            # Q_V(s, t+1) = f(s_{t+1})
            # next_observations = next_observations.cuda()
            next_target_q_values = self.target_model(next_observations)

            # for memoization
            self.next_target_expected_return_values_valid.fill(-1)

            for index,driver in enumerate(self.city.drivers):
                # got reward this turn
                # if not driver.is_online():
                #     target_q_values[index] = 0 #should stay 
                # else:
                if driver.is_online():
                    # pi(s, t+1)
                    next_target_policy = self.get_policy_from_action_values(next_target_q_values[[index]].squeeze())

                    # sigma pi(s,t+1) Q(s,t+1)
                    next_q_values = next_target_q_values[[index]].squeeze()

                    if self.strategy == POLICY_ENTROPY:
                        m = self.temperature * torch.log(torch.sum(torch.exp(next_q_values / self.temperature)))
                    else:
                        m = torch.dot(next_q_values, next_target_policy)
                    
                    target_q_values[index] += self.gamma * m   # gamma = 0.9
                    target_q_values_counts[index] += 1

     

            # For some roads, there are no drivers
            no_info = (target_q_values_counts == 0).int()

            # for road with >= 1 drivers: sum / (N + 0) = avg
            # for road with 0 driver: 0 / (0 + 1) = 0
            target_q_values /= (target_q_values_counts + no_info)

        # for road with 0 driver : don't have to update.
        # but to give a penalty for uncertainty(no experience), multiply by 0.9
        target_q_values += self.q_values * no_info * 0.9

        # should be between (0, 1)
        target_q_values = torch.clamp(target_q_values, min=1e-8, max=1)

        # set loss as weighted MSE
        difference = (self.q_values - target_q_values)
        weighted_mse = (difference ** 2) * (target_q_values_counts + no_info)
        loss = torch.mean(weighted_mse)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for debugging
        debug = False
        if self.city.city_time % 10 == 0 and debug:
            debug_target_q_values = target_q_values.squeeze().cpu().tolist()
            debug_q_values = self.q_values.squeeze().cpu().tolist()
            index = list(range(self.city.total_agent))
            debug_q_values_info = list(zip(debug_target_q_values, debug_q_values, index))
            debug_q_values_info.sort(reverse=True, key=lambda x:x[0])
            print(debug_q_values_info[0:30])
            print(loss)

    def get_policy(self, observations, use_target_model=False, to_numpy=True):
        policy = [None for _ in range(self.city.total_agent)]
        model = self.model if not use_target_model else self.target_model
        model = model.cuda()
        # Q_V(j, t) = f(s_t)
        q_values = model(observations)

        for index, driver in enumerate(self.city.drivers):
            out_nodes = [index]
            possible_action_values = q_values[out_nodes].squeeze()
            policy_v = self.get_policy_from_action_values(possible_action_values)
            if to_numpy:
                policy_v = policy_v.cpu().detach().numpy()
            policy[v] = policy_v
        self.q_values = q_values
        self.observations = observations
        return policy

    def get_policy_from_action_values(self, q_values: torch.Tensor):
        strategy = self.strategy
        if strategy == POLICY_ARGMAX:
            m = torch.max(q_values)
            p = (q_values == m).float()
        # Q^policy_pow
        elif strategy == POLICY_POW:
            if q_values.sum() == 0:
                p = torch.ones_like(q_values)
                p = p / p.sum()
            else:
                p = q_values / q_values.sum()
            p = p / torch.max(p)
            p = p**self.policy_pow
        # exp(Q/temperature)
        elif strategy == POLICY_EXP or strategy == POLICY_ENTROPY:
            q_values_max = torch.max(q_values)
            p = torch.exp((q_values-q_values_max) / self.temperature)

        if torch.isnan(p).any():
            print(q_values.sum())
            print(q_values)
            print("NAN")
        p /= p.sum()
        return p
