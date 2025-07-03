# =============================================================================
# FILE: visualize_packet_driven_ainode.py
#
# DESCRIPTION:
# This is a presentation-quality visualization script for our trained agent.
# It creates enhanced plots to better understand the agent's policy,
# correlating its decisions with system state and fairness metrics.
# =============================================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import os
from stable_baselines3 import PPO

# --- Import from your final training script ---
from train_packet_driven_v1_6_1_wrapper import (
    LtePacketDrivenEnv, FlattenActionWrapper, TDD_PATTERNS, TDD_ACTION_MAP,
    CQI_MIMO_TABLE, REWARD_CONFIG
)

# --- CONFIGURATION ---
MODEL_PATH = "lte_packet_driven_wrapped_agent.zip"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'packet_array_eval.pkl') # Adjust path if needed
SIMULATION_DURATION_S = 10 # A shorter duration is better for detailed policy plots
NUM_UE_PANELS = 2

# =============================================================================
# --- SECTION 2: SIMULATION AND DATA COLLECTION (Enhanced Logging) ---
# =============================================================================
print("--- Visualizing Agent Policy with Enhanced Logging ---")
try:
    model = PPO.load(MODEL_PATH, device='cpu')
    with open(DATA_PATH, 'rb') as f:
        packet_array = pickle.load(f)
    print("Agent and evaluation data loaded.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find required file. Missing: {e.filename}"); exit()

env = FlattenActionWrapper(
    LtePacketDrivenEnv(packet_array=packet_array, max_obs_ues=80, reward_config=REWARD_CONFIG)
)

log_data = []
obs, info = env.reset(options={'start_time': 0.0})
simulation_start_time = env.unwrapped.current_time_s
print(f"Running simulation for {SIMULATION_DURATION_S} seconds...")

while env.unwrapped.current_time_s < simulation_start_time + SIMULATION_DURATION_S:
    action_flat, _ = model.predict(obs, deterministic=True)
    action_dict = env.action(action_flat)
    
    # Get a snapshot of the full system state for logging
    all_active_ues = {ue_id: data.copy() for ue_id, data in env.unwrapped._get_sorted_ues_for_obs()}
    
    log_entry = {
        'time_s': env.unwrapped.current_time_s,
        'tdd_pattern_id': TDD_ACTION_MAP[action_dict['manager']],
        'slot_pattern': env.unwrapped.slot_pattern,
        'action_scores': action_dict['worker'],
        'all_active_ues': all_active_ues,
        'total_dl_buffer': sum(ue['dl_buffer_kb'] for ue in all_active_ues.values()),
        'total_ul_buffer': sum(ue['ul_buffer_kb'] for ue in all_active_ues.values())
    }
    log_data.append(log_entry)
    
    obs, reward, done, truncated, info = env.step(action_flat)
    if done:
        print("Episode finished during visualization run.")
        break
print("Simulation finished. Preparing enhanced plots...")

# =============================================================================
# --- SECTION 3: PLOTTING (with the suggested improvements) ---
# =============================================================================


ue_qfi_map = {}; all_ues_in_episode = set()
for log in log_data:
    for ue_id, data in log['all_active_ues'].items():
        all_ues_in_episode.add(ue_id)
        if ue_id not in ue_qfi_map: ue_qfi_map[ue_id] = data.get('qfi', 9)
ue_pairs = []; high_prio_ues = {ue for ue, qfi in ue_qfi_map.items() if qfi <= 3}; low_prio_ues = {ue for ue, qfi in ue_qfi_map.items() if qfi >= 9}
if high_prio_ues and low_prio_ues:
    high_prio_list, low_prio_list = list(high_prio_ues), list(low_prio_ues)
    for i in range(NUM_UE_PANELS):
        if i < len(high_prio_list) and i < len(low_prio_list): ue_pairs.append((high_prio_list[i], low_prio_list[i]))
        else: break
if not ue_pairs:
    print("ERROR: Not enough UEs to plot."); exit()


# --- PLOTTING ---
fig, axes = plt.subplots(NUM_UE_PANELS + 2, 1, figsize=(20, 7 * (NUM_UE_PANELS + 1)), sharex=True,
                         gridspec_kw={'height_ratios': [2, 3] + [4] * NUM_UE_PANELS})
fig.suptitle(f'Advanced Policy Analysis (Model: {MODEL_PATH})', fontsize=20, y=0.99)
time_axis = [log['time_s'] for log in log_data]

# --- Plot 1: Manager Intelligence  ---
ax_manager = axes[0]; ax_manager_twin = ax_manager.twinx()
total_dl_buffer = [log['total_dl_buffer'] for log in log_data]
total_ul_buffer = [log['total_ul_buffer'] for log in log_data]
tdd_ids = [log['tdd_pattern_id'] for log in log_data]

# Plot the system load as filled areas
ax_manager.fill_between(time_axis, 0, total_dl_buffer, color='darkblue', alpha=0.3, label='Total DL Load (KB)')
ax_manager.fill_between(time_axis, 0, total_ul_buffer, color='darkred', alpha=0.3, label='Total UL Load (KB)')
ax_manager.set_ylabel('Total System Load (KB)'); ax_manager.legend(loc='upper left'); ax_manager.grid(True, linestyle=':')
ax_manager.set_title("Manager Intelligence: TDD Selection vs. System Load")

# Overlay the agent's TDD choice
ax_manager_twin.step(time_axis, tdd_ids, where='post', label='Chosen TDD Pattern ID', color='green', linewidth=2)
ax_manager_twin.set_ylabel('TDD Pattern ID'); ax_manager_twin.set_yticks(list(TDD_PATTERNS.keys())); ax_manager_twin.legend(loc='upper right')

# --- Plot 2: Per-UE Latency (IMPROVEMENT 3) ---
ax_latency = axes[1]
colors = ['firebrick', 'darkblue', 'darkorange', 'darkviolet']
for i, (ue1_id, ue2_id) in enumerate(ue_pairs):
    ue1_lat, ue2_lat = [], []
    ue1_qfi = ue_qfi_map.get(ue1_id, 'N/A')
    ue2_qfi = ue_qfi_map.get(ue2_id, 'N/A')

    # ---  Safer data extraction loop ---
    for log in log_data:
        # For UE 1
        ue1_state = log['all_active_ues'].get(ue1_id)
        if ue1_state and ue1_state['dl_buffer']: # Check if UE exists AND its buffer is not empty
            oldest_packet_time = ue1_state['dl_buffer'][0][0]
            ue1_lat.append(log['time_s'] - oldest_packet_time)
        else:
            ue1_lat.append(0) # If buffer is empty, latency is zero

        # For UE 2
        ue2_state = log['all_active_ues'].get(ue2_id)
        if ue2_state and ue2_state['dl_buffer']: # Check if UE exists AND its buffer is not empty
            oldest_packet_time = ue2_state['dl_buffer'][0][0]
            ue2_lat.append(log['time_s'] - oldest_packet_time)
        else:
            ue2_lat.append(0) # If buffer is empty, latency is zero
    
    ax_latency.plot(time_axis, np.array(ue1_lat) * 1000, label=f'UE-{ue1_id} DL Latency (QFI={ue1_qfi})', color=colors[i*2], linestyle='-')
    ax_latency.plot(time_axis, np.array(ue2_lat) * 1000, label=f'UE-{ue2_id} DL Latency (QFI={ue2_qfi})', color=colors[i*2+1], linestyle='--')

ax_latency.set_title('Quality of Service: Oldest Packet Latency')
ax_latency.set_ylabel('Latency (ms)')
ax_latency.grid(True, linestyle=':')
ax_latency.legend()
ax_latency.set_ylim(bottom=0)

# --- Plots 3+: Worker Fairness Panels ---
for panel_idx, (ue1_id, ue2_id) in enumerate(ue_pairs):
    ax_panel = axes[panel_idx + 2]
    ax_panel_twin = ax_panel.twinx()
    ue1_qfi = ue_qfi_map.get(ue1_id, 'N/A')
    ue2_qfi = ue_qfi_map.get(ue2_id, 'N/A')
    
    # Get scores and buffers
    ue1_scores, ue2_scores = [], []
    ue1_buffers_dl, ue2_buffers_dl = [], []
    ue1_buffers_ul, ue2_buffers_ul = [], []

    # ---  Safer and Correct data extraction loop ---
    for log in log_data:
        # Get the list of UE IDs in the order they were presented to the agent
        # The 'all_active_ues' dictionary in the log is already sorted by buffer size
        sorted_ue_ids = list(log['all_active_ues'].keys())
        
        # Get buffer data for the UEs we are tracking
        ue1_buffers_dl.append(log['all_active_ues'].get(ue1_id, {}).get('dl_buffer_kb', 0))
        ue1_buffers_ul.append(log['all_active_ues'].get(ue1_id, {}).get('ul_buffer_kb', 0))
        ue2_buffers_dl.append(log['all_active_ues'].get(ue2_id, {}).get('dl_buffer_kb', 0))
        ue2_buffers_ul.append(log['all_active_ues'].get(ue2_id, {}).get('ul_buffer_kb', 0))

        # Get action scores for UE 1
        try:
            idx = sorted_ue_ids.index(ue1_id)
            if idx < env.unwrapped.max_obs_ues:
                # Agent's action is for both UL and DL, let's plot the DL score for now
                ue1_scores.append(log['action_scores'][idx*2 + 1]) 
            else:
                ue1_scores.append(0) # UE was active but not in the top group seen by the agent
        except ValueError:
            ue1_scores.append(0) # UE was not active at this TTI

        # Get action scores for UE 2
        try:
            idx = sorted_ue_ids.index(ue2_id)
            if idx < env.unwrapped.max_obs_ues:
                ue2_scores.append(log['action_scores'][idx*2 + 1])
            else:
                ue2_scores.append(0)
        except ValueError:
            ue2_scores.append(0)

    # Plot Scores on left axis (Primary)
    ax_panel.plot(time_axis, ue1_scores, label=f'UE-{ue1_id} DL Score (QFI={ue1_qfi})', color='red', alpha=0.8)
    ax_panel.plot(time_axis, ue2_scores, label=f'UE-{ue2_id} DL Score (QFI={ue2_qfi})', color='blue', alpha=0.8)
    ax_panel.set_ylabel('Agent Priority Score', color='black')
    ax_panel.set_ylim(-0.05, 1.05)
    ax_panel.legend(loc='upper left')
    ax_panel.grid(True, linestyle=':')
    
    # Plot Buffers on right axis (Secondary)
    # Plot both UL and DL buffers for clarity
    ax_panel_twin.fill_between(time_axis, 0, ue1_buffers_dl, color='red', alpha=0.2, label=f'UE-{ue1_id} DL Buffer')
    ax_panel_twin.plot(time_axis, ue1_buffers_ul, color='red', alpha=0.5, linestyle=':', label=f'UE-{ue1_id} UL Buffer')
    ax_panel_twin.fill_between(time_axis, 0, ue2_buffers_dl, color='blue', alpha=0.2, label=f'UE-{ue2_id} DL Buffer')
    ax_panel_twin.plot(time_axis, ue2_buffers_ul, color='blue', alpha=0.5, linestyle=':', label=f'UE-{ue2_id} UL Buffer')
    ax_panel_twin.set_ylabel('Buffer Size (KB)', color='gray')
    ax_panel_twin.set_ylim(bottom=0)
    ax_panel_twin.legend(loc='upper right')
    
    ax_panel.set_title(f"Scheduler Fairness Panel {panel_idx+1}: UE-{ue1_id} vs UE-{ue2_id}")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("advanced_policy_analysis_V6_1.png")
print("Saved enhanced analysis plot to advanced_policy_analysis.png")
plt.show()