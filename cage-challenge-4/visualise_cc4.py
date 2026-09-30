import os
import matplotlib
matplotlib.use("Agg")  # force non-interactive backend before pyplot is touched anywhere

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Agents.Wrappers.VisualiseRedExpansion import VisualiseRedExpansion

steps = 200
sg = EnterpriseScenarioGenerator(
    blue_agent_class=SleepAgent,
    green_agent_class=EnterpriseGreenAgent,
    red_agent_class=DiscoveryFSRed,
    steps=steps,
)
cyborg = CybORG(scenario_generator=sg, seed=7629)

visualise = VisualiseRedExpansion(cyborg, steps)
visualise.run()  # still runs the episode + builds self.fig via show_graph()

# --- save each step as a PNG instead of opening an interactive window ---
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

for i in range(len(visualise.collected_networks)):
    visualise._draw_network(i)
    visualise.fig.savefig(os.path.join(out_dir, f"frame_{i:03d}.png"), dpi=150)

print(f"Saved {len(visualise.collected_networks)} frames to {out_dir}/")
