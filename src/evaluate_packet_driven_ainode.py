# =============================================================================
# FILE: evaluate_packet_driven.py
#
# DESCRIPTION:
# This script evaluates our FINAL packet-driven unified agent.
# It loads the trained model, runs a single, continuous simulation starting
# from T=0, and visualizes key performance indicators (KPIs) like
# throughput, buffer latency, and TDD pattern selection.
# =============================================================================

import gymnasium as gym
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# --- Import the building blocks from our NEW training script ---
# Make sure this import path is correct for your project structure.
from train_packet_driven_ainode import LtePacketDrivenEnv, FlattenActionWrapper, TDD_PATTERNS, REWARD_CONFIG

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
# --- Path to our trained agent and the evaluation dataset ---
MODEL_PATH = "lte_packet_driven_wrapped_agent.zip"  # <-- Path to your trained model
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'packet_array_eval.pkl') # Adjust path if needed
# --- Simulation Parameters ---
EVALUATION_DURATION_S = 120  # Run the simulation for 120 seconds for a good overview
MANAGER_DECISION_INTERVAL_MS = 100 # This must match the setting in LtePacketDrivenEnv

# =============================================================================
# --- 2. THE MAIN EVALUATION LOGIC ---
# =============================================================================
if __name__ == '__main__':
    print("--- Starting Packet-Driven LTE System Evaluation ---")

    # --- Step A: Load the model and evaluation data ---
    print(f"Loading agent from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model not found at '{MODEL_PATH}'! Please train it first."); exit()
    model = PPO.load(MODEL_PATH, device='cpu')

    print(f"Loading evaluation data from: {DATA_PATH}")
    try:
        with open(DATA_PATH, 'rb') as f:
            packet_array = pickle.load(f)
    except FileNotFoundError:
        print(f"FATAL: Evaluation packet data not found at {DATA_PATH}!"); exit()

    # --- Step B: Set up a SINGLE, controlled environment with the wrapper ---
    print("Initializing simulation environment...")
    env = FlattenActionWrapper(
        LtePacketDrivenEnv(packet_array=packet_array, max_obs_ues=80, reward_config=REWARD_CONFIG)
    )
    
    # --- Step C: Prepare for Data Collection ---
    metrics = {
        'time': [], 'tdd_pattern_id': [], 'tdd_pattern_str': [],
        'ul_throughput_kbps': [], 'dl_throughput_kbps': [],
        'avg_ul_buffer_kb': [], 'avg_dl_buffer_kb': [],
        'active_ues': []
    }
    
    # --- FORCE THE SIMULATION TO START AT T=0 ---
    obs, info = env.reset(options={'start_time': 0.0})
    
    simulation_start_time = env.unwrapped.current_time_s
    print(f"Running simulation from t={simulation_start_time:.3f}s for {EVALUATION_DURATION_S} seconds...")
    start_wall_time = time.time()

    # --- Step D: The Main Simulation Loop ---
    # Accumulators for metrics over each manager interval
    ul_throughput_accumulator = 0.0
    dl_throughput_accumulator = 0.0
    
    # Loop until the simulation duration is reached
    while env.unwrapped.current_time_s < simulation_start_time + EVALUATION_DURATION_S:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Accumulate throughput from the info dictionary
        ul_throughput_accumulator += info.get('ul_served_kb', 0)
        dl_throughput_accumulator += info.get('dl_served_kb', 0)

        # Check if we've reached the end of a manager's decision interval
        if env.unwrapped.current_step_in_episode % MANAGER_DECISION_INTERVAL_MS == 0 and env.unwrapped.current_step_in_episode > 0:
            metrics['time'].append(env.unwrapped.current_time_s)
            
            # --- TDD Pattern Logging ---
            current_pattern_str = "".join(env.unwrapped.slot_pattern)
            try:
                current_pattern_id = [k for k, v in TDD_PATTERNS.items() if "".join(v) == current_pattern_str][0]
            except IndexError:
                current_pattern_id = -1
            metrics['tdd_pattern_id'].append(current_pattern_id)
            metrics['tdd_pattern_str'].append(current_pattern_str)
            
            # --- Throughput Calculation ---
            interval_duration_s = MANAGER_DECISION_INTERVAL_MS * 0.001
            dl_kbps = (dl_throughput_accumulator * 8) / interval_duration_s
            ul_kbps = (ul_throughput_accumulator * 8) / interval_duration_s
            metrics['dl_throughput_kbps'].append(dl_kbps)
            metrics['ul_throughput_kbps'].append(ul_kbps)
            
            # Reset accumulators for the next interval
            dl_throughput_accumulator = 0.0
            ul_throughput_accumulator = 0.0

            # --- Buffer and Active UE Calculation ---
            active_ues_dict = env.unwrapped.active_ues
            metrics['active_ues'].append(len(active_ues_dict))
            if active_ues_dict:
                avg_ul_buffer = sum(u['ul_buffer_kb'] for u in active_ues_dict.values()) / len(active_ues_dict)
                avg_dl_buffer = sum(u['dl_buffer_kb'] for u in active_ues_dict.values()) / len(active_ues_dict)
            else:
                avg_ul_buffer, avg_dl_buffer = 0, 0
            metrics['avg_ul_buffer_kb'].append(avg_ul_buffer)
            metrics['avg_dl_buffer_kb'].append(avg_dl_buffer)
            
        if done:
            print("Episode finished. Ending evaluation as the data trace may have ended.")
            break

    end_wall_time = time.time()
    print(f"Simulation finished. Total wall-clock time: {end_wall_time - start_wall_time:.2f}s")
    print("--- Generating Performance Plots ---")

# =============================================================================
# --- 3. VISUALIZATION ---
# =============================================================================
fig, (ax_tp, ax_buffer, ax_manager) = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                                                  gridspec_kw={'height_ratios': [3, 3, 2]})
fig.suptitle(f'Packet-Driven Agent Performance (Model: {MODEL_PATH})', fontsize=16)

# --- Plot 1: UL/DL Throughput ---
ax_tp.plot(metrics['time'], np.array(metrics['dl_throughput_kbps'])/1000, label='DL Throughput (Mbps)', color='salmon', linewidth=2)
ax_tp.fill_between(metrics['time'], 0, np.array(metrics['dl_throughput_kbps'])/1000, color='salmon', alpha=0.3)
ax_tp.plot(metrics['time'], np.array(metrics['ul_throughput_kbps'])/1000, label='UL Throughput (Mbps)', color='skyblue', linewidth=2)
ax_tp.fill_between(metrics['time'], 0, np.array(metrics['ul_throughput_kbps'])/1000, color='skyblue', alpha=0.3)
ax_tp.set_ylabel('Throughput (Mbps)')
ax_tp.set_title('Achieved Throughput')
ax_tp.grid(True, linestyle='--', alpha=0.6)
ax_tp.legend(loc='upper left')

# --- Plot 2: Latency Proxy (Buffer & Active UEs) ---
ax_buffer_twin = ax_buffer.twinx()
ax_buffer.plot(metrics['time'], metrics['avg_ul_buffer_kb'], label='Avg. UL Buffer/UE (KB)', color='blue', linestyle='--')
ax_buffer.plot(metrics['time'], metrics['avg_dl_buffer_kb'], label='Avg. DL Buffer/UE (KB)', color='red', linestyle='--')
ax_buffer.set_ylabel('Average Buffer Size per UE (KB)')
ax_buffer.legend(loc='upper left')
ax_buffer.set_ylim(bottom=0)

ax_buffer_twin.plot(metrics['time'], metrics['active_ues'], label='Active UEs', color='green', marker='.', markersize=4, linestyle=':')
ax_buffer_twin.set_ylabel('Number of Active UEs', color='green')
ax_buffer_twin.tick_params(axis='y', labelcolor='green')
ax_buffer_twin.legend(loc='upper right')
ax_buffer_twin.set_ylim(bottom=0)
ax_buffer.set_title('System Congestion: Buffers and Active Users')
ax_buffer.grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Agent's TDD Decisions ---
ax_manager.step(metrics['time'], metrics['tdd_pattern_id'], where='post', color='black', label='Chosen TDD Pattern ID', linewidth=2.5)
ax_manager.set_ylabel('TDD Pattern ID')
ax_manager.set_title("Agent's TDD Pattern Selection")
ax_manager.set_xlabel('Time (s)')
ax_manager.grid(True, linestyle='--', alpha=0.6)
ax_manager.set_yticks(list(TDD_PATTERNS.keys()))

for i in range(len(metrics['time'])):
    if i == 0 or metrics['tdd_pattern_id'][i] != metrics['tdd_pattern_id'][i-1]:
        ax_manager.text(metrics['time'][i] + 0.5, metrics['tdd_pattern_id'][i], f" {metrics['tdd_pattern_str'][i]}", 
                      va='center', ha='left', fontsize=9, color='purple', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("evaluate_packet_driven_analysis.png")
plt.show()