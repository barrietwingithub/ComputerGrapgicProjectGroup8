import time
import json
import numpy as np
from typing import List, Tuple
from agent_manager import AgentManager

class StatisticsCollector:
    """Collects and analyzes simulation statistics"""

    def __init__(self):
        self.density_history: List[np.ndarray] = []
        self.flow_rates: List[float] = []
        self.congestion_points: List[Tuple[float,float]] = []
        self.time_steps: List[float] = []
        self.start_time = time.time()

    def update(self, agent_manager: AgentManager):
        current_time = time.time() - self.start_time
        self.time_steps.append(current_time)

        # Density map
        density = agent_manager.get_density_map()
        self.density_history.append(density)

        # Flow rate
        if agent_manager.agents:
            avg_speed = np.mean([a.velocity.magnitude() for a in agent_manager.agents])
            self.flow_rates.append(avg_speed)

        # Congestion points (high density)
        if len(self.density_history) > 0:
            current_density = self.density_history[-1]
            threshold = np.mean(current_density) + np.std(current_density)
            congestion_indices = np.where(current_density > threshold)
            for i,j in zip(congestion_indices[0], congestion_indices[1]):
                x = i / current_density.shape[0] * 800
                y = j / current_density.shape[1] * 600
                self.congestion_points.append((x,y))

    def get_average_flow_rate(self) -> float:
        return np.mean(self.flow_rates) if self.flow_rates else 0.0

    def export_data(self, filename:str):
        data = {
            'time_steps': self.time_steps,
            'flow_rates': self.flow_rates,
            'congestion_points': self.congestion_points,
            'density_history': [d.tolist() for d in self.density_history]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)