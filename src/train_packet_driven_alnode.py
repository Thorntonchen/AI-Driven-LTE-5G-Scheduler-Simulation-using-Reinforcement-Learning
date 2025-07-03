# =============================================================================
# FILE: train_packet_driven_ainode.py
#
# DESCRIPTION:
# This version implements the final, correct solution to the action space
# problem. It uses a custom ActionWrapper to "flatten" the complex Dict
# action space into a single Box space that the standard PPO algorithm
# can handle. This is a robust and standard technique.
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
from collections import deque
from numba import njit
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# --- SECTION 1: UNCHANGED CONSTANTS & HELPERS ---
CQI_MIMO_TABLE = np.array([
    [0.0, 1], [0.23, 1], [0.38, 1], [0.60, 1], [0.88, 1], [1.18, 1],
    [1.48, 2], [1.91, 2], [2.41, 2], [2.73, 2], [3.32, 4], [3.90, 4],
    [4.52, 4], [5.12, 4], [5.55, 4], [6.25, 4]
])
TDD_PATTERNS = {
    0: ['D', 'S', 'U', 'U', 'U', 'D', 'S', 'U', 'U', 'U'],
    1: ['D', 'S', 'U', 'U', 'D', 'D', 'S', 'U', 'U', 'D'],
    2: ['D', 'S', 'U', 'D', 'D', 'D', 'S', 'U', 'D', 'D'],
    5: ['D', 'S', 'U', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    6: ['D', 'S', 'D', 'D', 'D', 'D', 'S', 'D', 'D', 'D']
}
TDD_ACTION_MAP = list(TDD_PATTERNS.keys())
SPECIAL_SUBFRAME_CONFIG = {'dl_symbols': 9, 'ul_symbols': 2}
CAPPED_MAX_PENALTY = 2000.0 # A very large, but fixed, maximum penalty
sinr_mean = 18    # Target mean SINR for good coverage
sinr_std_dev = 4   # Controls spread (smaller = more concentrated around mean)
QUALITY_FOCUSED_CONFIG = {
    'log_throughput_scaler': 60.0,
    'latency_budget_s': {'high_priority': 0.003, 'medium_priority': 0.020, 'low_priority': 0.080},
    'qfi_weights': {'high_priority': 200.0, 'medium_priority': 15.0, 'low_priority': 1.0},
    'final_reward_scaler': 100.0
}
REWARD_CONFIG = QUALITY_FOCUSED_CONFIG

@njit
def schedule_rbs_v5(rbs_to_allocate, scores, ue_buffer_kb, cqi, num_symbols,
                    cqi_mimo_table):
    
    rbs_remaining = rbs_to_allocate
    local_scores = scores.copy()
    local_scores[ue_buffer_kb <= 0] = -1.0

    # --- NEW: Create the array to be returned INSIDE the function ---
    packets_served_kb = np.zeros_like(ue_buffer_kb)
    #cqi_sum=0
    #cqi_count=0
    while rbs_remaining >= 1 and np.sum(local_scores) > 1e-9:
        w_idx = np.argmax(local_scores)

        cqi_val = int(cqi[w_idx])
        se, mimo = cqi_mimo_table[cqi_val]
        cap_per_rb = (12 * num_symbols * se * mimo) / (8 * 1024)
        #cqi_sum +=cqi_val
        #cqi_count +=1
        if cap_per_rb < 1e-6:
            local_scores[w_idx] = -1.0
            continue
        
        rbs_remaining -= 1
        data_served_this_rb = min(ue_buffer_kb[w_idx], cap_per_rb)
        
        packets_served_kb[w_idx] += data_served_this_rb
        ue_buffer_kb[w_idx] -= data_served_this_rb
        
        local_scores[w_idx] *= 0.95
        if ue_buffer_kb[w_idx] < 1e-6:
            local_scores[w_idx] = -1.0
    #if(cqi_count>0):
    #    print(cqi_sum/cqi_count)            
    # --- NEW: Return the array with the results ---
    return packets_served_kb
def sinr_to_cqi(sinr_db):
    if sinr_db < -6: return 1
    if sinr_db < -4: return 2
    if sinr_db < -2: return 3
    if sinr_db < 0: return 4
    if sinr_db < 2: return 5
    if sinr_db < 4: return 6
    if sinr_db < 8: return 7
    if sinr_db < 12: return 9
    if sinr_db < 16: return 11
    if sinr_db < 20: return 13
    return 15


# --- SECTION 2: THE PACKET-DRIVEN ENVIRONMENT (UNCHANGED INTERNALLY) ---

class LtePacketDrivenEnv(gym.Env):

    def __init__(self, packet_array, max_obs_ues=80, reward_config=None, verbose=False):
        super().__init__()
        self.all_packets = packet_array
        self.verbose = verbose
        self.reward_config = reward_config if reward_config is not None else REWARD_CONFIG
        self.max_obs_ues = max_obs_ues
        self.total_rbs = 275
        self.tti_duration_s = 0.001

        self.manager_decision_frequency = 100
        self.max_manager_steps = 50
        self.episode_length_steps = self.manager_decision_frequency * self.max_manager_steps

        self.active_ues = {}
        
        self.obs_per_ue = 5 # ul_buffer, dl_buffer, cqi, qfi, worst_latency
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.max_obs_ues * self.obs_per_ue,), dtype=np.float32)
        self.action_space = spaces.Dict({
            "manager": spaces.Discrete(len(TDD_PATTERNS)),
            "worker": spaces.Box(low=0, high=1, shape=(self.max_obs_ues * 2,), dtype=np.float32)
        })
        self.PACKET_TTL_S = 1.0 #1 SECOND TIMEOUT
        self.DROP_PENALTY_SCALER = 50.0
        self.packet_pointer = 0
        self.current_time_s = 0.0
        self.current_step_in_episode = 0
        self.slot_pattern = TDD_PATTERNS[2]
        self.cqi_mimo_table = CQI_MIMO_TABLE

    def _update_packet_arrivals(self):
        while (self.packet_pointer < len(self.all_packets) and
               self.all_packets[self.packet_pointer]['time'] <= self.current_time_s):
            
            packet = self.all_packets[self.packet_pointer]
            ue_id = int(packet['session_id'])
            
            if ue_id not in self.active_ues:
                base_sinr = np.random.normal(sinr_mean, sinr_std_dev)
                base_sinr = np.clip(base_sinr, -5, 22)
                self.active_ues[ue_id] = {
                    'qfi': packet['qfi'], 'ul_buffer': deque(), 'dl_buffer': deque(),
                    'ul_buffer_kb': 0.0, 'dl_buffer_kb': 0.0,
                    'base_sinr':base_sinr,'cqi': 10,
                    'avg_throughput_kbps': 10.0  # <-- ADD THIS LINE, start at a small non-zero value
                }

            ue_state = self.active_ues[ue_id]
            packet_info = (packet['time'], packet['Size_KB'])
            
            if packet['PDUType'] == 1:
                ue_state['ul_buffer'].append(packet_info)
                ue_state['ul_buffer_kb'] += packet['Size_KB']
            else:
                ue_state['dl_buffer'].append(packet_info)
                ue_state['dl_buffer_kb'] += packet['Size_KB']
            self.packet_pointer += 1

    def _update_radio_conditions(self):
        for ue_state in self.active_ues.values():
            ue_state['base_sinr'] += np.random.normal(0, 0.05)
            ue_state['cqi'] = sinr_to_cqi(ue_state['base_sinr'] + np.random.normal(0, 1.5))

    def _get_sorted_ues_for_obs(self):
        if not self.active_ues:
            return []
        return sorted(self.active_ues.items(),
                      key=lambda item: item[1]['ul_buffer_kb'] + item[1]['dl_buffer_kb'],
                      reverse=True)

    def _get_observation(self, sorted_ues):
        obs_array = np.zeros((self.max_obs_ues, self.obs_per_ue), dtype=np.float32)
        num_ues_to_report = min(len(sorted_ues), self.max_obs_ues)
        for i in range(num_ues_to_report):
            ue_id, ue_state = sorted_ues[i]
            obs_array[i, 0] = ue_state['ul_buffer_kb']
            obs_array[i, 1] = ue_state['dl_buffer_kb']
            obs_array[i, 2] = ue_state['cqi']
            obs_array[i, 3] = ue_state['qfi']
            ul_arrival = ue_state['ul_buffer'][0][0] if ue_state['ul_buffer'] else self.current_time_s
            dl_arrival = ue_state['dl_buffer'][0][0] if ue_state['dl_buffer'] else self.current_time_s
            obs_array[i, 4] = self.current_time_s - min(ul_arrival, dl_arrival)
        return obs_array.flatten()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {} # Ensure options is a dictionary
        
        self.active_ues.clear()
        self.current_step_in_episode = 0
        
        # ---  Controlled Starting Logic ---
        start_time = options.get('start_time', -1)
        
        if start_time >= 0:
            # If a start time is provided, find the first packet at or after that time.
            # np.searchsorted is very fast for this.
            self.packet_pointer = np.searchsorted(self.all_packets['time'], start_time, side='left')
            if self.packet_pointer >= len(self.all_packets): # Handle case where start_time is after all packets
                self.packet_pointer = 0 
        else:
            # Default behavior for training: start at a random point
            self.packet_pointer = self.np_random.integers(len(self.all_packets) // 4)

        # Set the current time to the time of the starting packet
        if self.packet_pointer < len(self.all_packets):
            self.current_time_s = self.all_packets[self.packet_pointer]['time']
        else:
            self.current_time_s = 0.0 # Fallback


        self.slot_pattern = TDD_PATTERNS[self.np_random.choice(list(TDD_PATTERNS.keys()))]
        self._update_packet_arrivals()
        
        return self._get_observation([]), {}

    # =============================================================================
    # === The Final, Correct, and Complete `step` method with Stable Scheduler ===
    # =============================================================================
    def step(self, action):
        # --- Stage 1: Manager's Decision ---
        if self.current_step_in_episode % self.manager_decision_frequency == 0:
            self.slot_pattern = TDD_PATTERNS[TDD_ACTION_MAP[action["manager"]]]

        # --- Stage 2: System State Update & Pre-scheduling Buffer snapshot ---
        self._update_packet_arrivals()
        self._update_radio_conditions()
    
        buffer_before_dl = sum(u['dl_buffer_kb'] for u in self.active_ues.values())
        buffer_before_ul = sum(u['ul_buffer_kb'] for u in self.active_ues.values())

        # --- Stage 3: Packet Dropping (TTL Check) & Pruning ---
        total_drop_penalty = 0.0
        ues_to_prune_from_drops = []
        for ue_id, ue_state in list(self.active_ues.items()):
            # Check Downlink buffer
            while ue_state['dl_buffer'] and (self.current_time_s - ue_state['dl_buffer'][0][0] > self.PACKET_TTL_S):
                _, packet_size = ue_state['dl_buffer'].popleft()
                ue_state['dl_buffer_kb'] -= packet_size
                qfi = ue_state['qfi']
                if qfi <= 3: qfi_weight = self.reward_config['qfi_weights']['high_priority']
                elif qfi <= 8: qfi_weight = self.reward_config['qfi_weights']['medium_priority']
                else: qfi_weight = self.reward_config['qfi_weights']['low_priority']
                total_drop_penalty += self.DROP_PENALTY_SCALER * qfi_weight
            # Check Uplink buffer
            while ue_state['ul_buffer'] and (self.current_time_s - ue_state['ul_buffer'][0][0] > self.PACKET_TTL_S):
                _, packet_size = ue_state['ul_buffer'].popleft()
                ue_state['ul_buffer_kb'] -= packet_size
                qfi = ue_state['qfi']
                if qfi <= 3: qfi_weight = self.reward_config['qfi_weights']['high_priority']
                elif qfi <= 8: qfi_weight = self.reward_config['qfi_weights']['medium_priority']
                else: qfi_weight = self.reward_config['qfi_weights']['low_priority']
                total_drop_penalty += self.DROP_PENALTY_SCALER * qfi_weight
            
            if ue_state['ul_buffer_kb'] < 1e-6 and ue_state['dl_buffer_kb'] < 1e-6:
                ues_to_prune_from_drops.append(ue_id)
                del self.active_ues[ue_id]

        # --- Stage 4: High-Speed Scheduling with CORRECT Resource Splitting ---
        sorted_ues = self._get_sorted_ues_for_obs()
        num_ues_to_schedule = len(sorted_ues)

        dl_served_kb_total = 0.0
        ul_served_kb_total = 0.0

        if num_ues_to_schedule > 0:
            # Create arrays for the scheduler (this part is correct)
            ue_indices = np.zeros(num_ues_to_schedule, dtype=np.int32)
            cqi_values = np.zeros(num_ues_to_schedule, dtype=np.int32)
            ul_buffers = np.zeros(num_ues_to_schedule, dtype=np.float32)
            dl_buffers = np.zeros(num_ues_to_schedule, dtype=np.float32)
            for i, (ue_id, ue_state) in enumerate(sorted_ues):
                ue_indices[i] = ue_id; cqi_values[i] = ue_state['cqi']
                ul_buffers[i] = ue_state['ul_buffer_kb']; dl_buffers[i] = ue_state['dl_buffer_kb']

            # Scoring logic (this part is also correct and stable)
            dl_scores = cqi_values.copy().astype(np.float32)
            ul_scores = cqi_values.copy().astype(np.float32)
            num_ues_in_action = min(num_ues_to_schedule, self.max_obs_ues)
            worker_action = action["worker"]
            dl_scores[:num_ues_in_action] *= (1 + 100 * worker_action[1:num_ues_in_action*2:2])
            ul_scores[:num_ues_in_action] *= (1 + 100 * worker_action[0:num_ues_in_action*2:2])

            # --- Run Numba Scheduler with INDEPENDENT Resource Pools ---
            slot_type = self.slot_pattern[self.current_step_in_episode % len(self.slot_pattern)]
            dl_symbols = 14 if slot_type == 'D' else (SPECIAL_SUBFRAME_CONFIG['dl_symbols'] if slot_type == 'S' else 0)
            ul_symbols = 14 if slot_type == 'U' else (SPECIAL_SUBFRAME_CONFIG['ul_symbols'] if slot_type == 'S' else 0)

            packets_served_dl = np.zeros(num_ues_to_schedule)
            packets_served_ul = np.zeros(num_ues_to_schedule)

            # DL scheduler call
            if dl_symbols > 0:
                packets_served_dl = schedule_rbs_v5(
                    float(self.total_rbs), dl_scores, dl_buffers, cqi_values, 
                    dl_symbols, self.cqi_mimo_table
                )
            
            # UL scheduler call
            if ul_symbols > 0:
                packets_served_ul = schedule_rbs_v5(
                    float(self.total_rbs), ul_scores, ul_buffers, cqi_values, 
                    ul_symbols, self.cqi_mimo_table
                )

            # --- Stage 5: Buffer Updates and Pruning After Service ---
            for i in range(num_ues_to_schedule):
                ue_id = ue_indices[i]
                if ue_id not in self.active_ues: continue # Already pruned by TTL
                ue_state = self.active_ues[ue_id]
            
                served_dl = packets_served_dl[i]
                if served_dl > 0:
                    ue_state['dl_buffer_kb'] -= served_dl
                    dl_served_kb_total += served_dl
                    buf = ue_state['dl_buffer']
                    temp_served = served_dl
                    while temp_served > 0 and buf:
                        if temp_served >= buf[0][1]: temp_served -= buf.popleft()[1]
                        else: buf[0] = (buf[0][0], buf[0][1] - temp_served); break # Partial service
            
                served_ul = packets_served_ul[i]
                if served_ul > 0:
                    ue_state['ul_buffer_kb'] -= served_ul
                    ul_served_kb_total += served_ul
                    buf = ue_state['ul_buffer']
                    temp_served = served_ul
                    while temp_served > 0 and buf:
                        if temp_served >= buf[0][1]: temp_served -= buf.popleft()[1]
                        else: buf[0] = (buf[0][0], buf[0][1] - temp_served); break # Partial service

                if ue_state['ul_buffer_kb'] < 1e-6 and ue_state['dl_buffer_kb'] < 1e-6:
                    del self.active_ues[ue_id]
        
        # --- Stage 6: Reward Calculation ---
        throughput_reward = self.reward_config['log_throughput_scaler'] * np.log1p(dl_served_kb_total + ul_served_kb_total)
        max_weighted_latency = 0.0
        for ue_id, ue_state in self.active_ues.items():
            ul_arrival = ue_state['ul_buffer'][0][0] if ue_state['ul_buffer'] else self.current_time_s
            dl_arrival = ue_state['dl_buffer'][0][0] if ue_state['dl_buffer'] else self.current_time_s
            worst_latency = self.current_time_s - min(ul_arrival, dl_arrival)
        
            qfi = ue_state['qfi']
            if qfi <= 3: qfi_weight, budget = self.reward_config['qfi_weights']['high_priority'], self.reward_config['latency_budget_s']['high_priority']
            elif qfi <= 8: qfi_weight, budget = self.reward_config['qfi_weights']['medium_priority'], self.reward_config['latency_budget_s']['medium_priority']
            else: qfi_weight, budget = self.reward_config['qfi_weights']['low_priority'], self.reward_config['latency_budget_s']['low_priority']
        
            excess_latency = max(0, worst_latency - budget)
            max_weighted_latency = max(max_weighted_latency, excess_latency * qfi_weight)
        
        reward_unscaled = throughput_reward - max_weighted_latency - total_drop_penalty
        reward = reward_unscaled / self.reward_config['final_reward_scaler']

        # --- Stage 7: Debug Logging ---
        if self.verbose and self.current_step_in_episode % 103  == 0:#if you use the 100,it always the last tti of frame it 's D.So use prime 103
            print(f"[T={self.current_time_s:.3f}s|S {self.current_step_in_episode:4d}] "
                  f"Active:{len(self.active_ues):<4} | "
                  f"TDD:{''.join(self.slot_pattern)} | "
                  f"DL Buf/Tx: {buffer_before_dl:8.2f}/{dl_served_kb_total:7.2f} KB | "
                  f"UL Buf/Tx: {buffer_before_ul:8.2f}/{ul_served_kb_total:7.2f} KB")

        # --- Stage 8: Finalize Step ---
        self.current_time_s += self.tti_duration_s
        self.current_step_in_episode += 1
    
        observation = self._get_observation(self._get_sorted_ues_for_obs())
        terminated = self.current_step_in_episode >= self.episode_length_steps
        info = {'ul_served_kb': ul_served_kb_total, 'dl_served_kb': dl_served_kb_total}
    
        return observation, reward, terminated, False, info


# --- SECTION 3: GUARANTEED-TO-WORK ACTION WRAPPER ---
class FlattenActionWrapper(gym.ActionWrapper):
    """
    A wrapper to flatten the Dict action space into a single Box space,
    making it compatible with standard PPO and MlpPolicy.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.original_action_space = env.action_space
        
        # Define the new, flattened action space
        worker_action_size = self.original_action_space['worker'].shape[0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1 + worker_action_size,), # 1 for manager + size of worker action
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> dict:
        """
        Translates the agent's flat action vector back into the
        environment's expected Dict action.
        """
        # Manager action: scale from [-1, 1] to [0, num_patterns-1] and discretize
        num_manager_actions = self.original_action_space['manager'].n
        manager_action_raw = (action[0] + 1) / 2 * num_manager_actions
        manager_action = np.clip(int(np.floor(manager_action_raw)), 0, num_manager_actions - 1)
        
        # Worker action: scale from [-1, 1] to [0, 1]
        worker_action_raw = action[1:]
        worker_action = (worker_action_raw + 1) / 2
        
        return {
            "manager": manager_action,
            "worker": worker_action
        }

# --- SECTION 4: CALLBACK and MAIN SCRIPT ---
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq; self.save_path = save_path; self.name_prefix = name_prefix
    def _init_callback(self) -> None:
        if self.save_path is not None: os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(path)
            if self.verbose > 0: print(f"Saving model checkpoint to {path}")
        return True

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    print("--- Loading Packet Data ---")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'packet_array_train.pkl')
    try:
        with open(data_path, 'rb') as f:
            packet_array = pickle.load(f)
        print(f"Loaded {len(packet_array)} total packets.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Packet data file not found at '{data_path}'."); exit()

    def create_env():
        env = LtePacketDrivenEnv(packet_array=packet_array, max_obs_ues=80, verbose=True)
        env = FlattenActionWrapper(env)
        return env

    num_cpu = 8
    model_path = "lte_packet_driven_wrapped_agent.zip"
    log_dir = "./logs/packet_driven_wrapped_log/"
    checkpoint_dir = "./checkpoints/packet_driven_wrapped/"
    

    env = make_vec_env(create_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # --- THE FINAL, CORRECTED MODEL ---
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir, 
        device='cuda', 
        n_steps=5000, 
        batch_size=512, 
        n_epochs=4
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=checkpoint_dir, name_prefix="packet_driven_wrapped")
    
    try:
        model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
    finally:
        model.save(model_path)
        print(f"Model saved to {model_path}")
        env.close()

    print("\n--- Packet-Driven Training Complete ---")

