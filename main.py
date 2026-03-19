from crowd_simulation import CrowdSimulation
import sys


def main():

    print("=" * 60)
    print("REAL-TIME CROWD SIMULATION USING AGENT-BASED MODELS")
    print("=" * 60)

    sim = CrowdSimulation()

    if len(sys.argv) > 1:

        arg = sys.argv[1]

        if arg in ("--demo", "-d"):

            sim.spawn_agents(30)

        else:

            print("Usage: python main.py [--demo]")
            print("--demo  : run simulation with 50 agents")

    sim.run()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.run()