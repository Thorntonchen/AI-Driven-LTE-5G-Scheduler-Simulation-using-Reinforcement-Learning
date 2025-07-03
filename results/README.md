==================================================
PLOT 1: PACKET-DRIVEN AGENT PERFORMANCE
==================================================
Top Panel (Achieved Throughput):
• Fantastic performance: Very high & stable DL Throughput (150-200 Mbps)
• UL Throughput lower but stable/non-zero
• Indicates healthy, working system clearing data

Middle Panel (System Congestion - KEY INSIGHT):
• Active UEs (green dotted line) now COMPLETELY STABLE (10-40 UEs)
• No longer explodes to 900 UEs → Proves final architecture/scheduler work perfectly
• Avg. Buffer Size remains low and controlled

Bottom Panel (Agent's TDD Selection):
• Shows agent intelligence:
  - Starts with balanced Pattern ID 2: DSUDDDSUDD
  - Switches to heavy DL Pattern ID 6: DSDDDDSDDD to maximize throughput
• Makes smart, long-term decisions

==================================================
PLOT 2: PACKET-DRIVEN AGENT POLICY ANALYSIS
==================================================
Top Panel (TDD Slot Pattern):
• Confirms agent's chosen pattern with individual 'D','S','U' slots

Second Panel (System Load):
• Shows total buffer size rise/fall as agent manages traffic

Performance Panels (UE-12291 vs UE-18432):
• Reveals agent's decision-making:
  - Gives scheduling scores (bottom graph spikes)
  - Bases scores on buffer size and QFI
  - Actively makes UE prioritization choices
