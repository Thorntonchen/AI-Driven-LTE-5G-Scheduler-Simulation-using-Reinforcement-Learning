# AI-Driven LTE/5G Scheduler Simulation using Reinforcement Learning

This project presents a high-fidelity, packet-driven discrete-event simulator for an LTE/5G cell. It includes a unified Reinforcement Learning agent, trained using Proximal Policy Optimization (PPO), that learns to perform two critical tasks simultaneously:

1.  **Strategic TDD Pattern Selection:** The agent acts as a "manager," dynamically selecting the optimal TDD (Time-Division Duplex) configuration based on real-time traffic load to balance uplink and downlink capacity.
2.  **Real-Time UE Scheduling:** The agent acts as a "worker," scheduling user equipment (UEs) on a TTI-by-TTI basis to maximize throughput while ensuring Quality of Service (QoS) by minimizing latency.

This work serves as a proof-of-concept for applying modern AI techniques to solve complex resource management problems in 5G and O-RAN RIC (xApp/rApp) environments.

---

## Key Features

- **High-Performance Simulation Core:** Built in Python with NumPy and accelerated with Numba's JIT compiler for high-speed event processing.
- **Realistic Packet-Driven Model:** Simulates traffic based on a time-sorted array of individual packet arrivals, providing a more realistic model than session-based approaches.
- **Sophisticated RL Agent:** A single PPO agent learns a complex, unified policy for both long-term (TDD) and short-term (scheduling) decisions.
- **QoS-Aware:** The environment models latency and QFI (QoS Flow Identifier), and the agent is trained with a reward function that penalizes high latency for priority users.
- **Dynamic TDD Frame Adaptation:** Demonstrates the core functionality required for an intelligent rApp in an O-RAN architecture.

---

## Performance Showcase

The trained agent demonstrates intelligent and adaptive behavior. After training, it successfully stabilizes the network under heavy load, achieving high throughput while maintaining low latency for all users.

#### Agent Intelligence: Adapting to System Load

The agent learns to switch to a Downlink-heavy TDD pattern (ID 5/6) during periods of high DL congestion, and then switches back to a more balanced pattern once the load is cleared.

![Performance Plot](results/advanced_policy_analysis.png)
*(This plot shows the agent's TDD choice (green) adapting to the total system load (blue/red areas).)*

---

## Getting Started

### Prerequisites

- Python 3.12.9+
- The required libraries can be installed via pip:
  ```bash
  pip install -r requirements.txt
