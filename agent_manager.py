import random
from typing import List, Optional
from agent import Agent, Vector2D, AgentState
from environment import Environment
from behavior import BehaviorSystem
import numpy as np

class AgentManager:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.agents: List[Agent] = []
        self.next_agent_id = 0
        self.behavior_system = BehaviorSystem()

        self.behavior_weights = {
            AgentState.NORMAL: {'seek':1.0,'avoid':1.5,'cohesion':0.3,'separation':0.5},
            AgentState.PANIC: {'seek':2.0,'avoid':3.0,'cohesion':0.1,'separation':0.8},
            AgentState.FOLLOWING: {'seek':0.5,'avoid':1.5,'cohesion':1.0,'separation':0.3},
        }

    def spawn_agent(self, position: Optional[Vector2D] = None) -> Agent:
        if position is None:
            position = random.choice(self.environment.spawn_points) if self.environment.spawn_points else Vector2D(
                random.uniform(50,self.environment.width-50),
                random.uniform(50,self.environment.height-50)
            )
        destination = random.choice(self.environment.goals) if self.environment.goals else Vector2D(self.environment.width/2, self.environment.height/2)
        agent = Agent(
            id=self.next_agent_id,
            position=position,
            velocity=Vector2D(0,0),
            destination=destination,
            personal_space_radius=random.uniform(1.0,2.0),
            max_speed=random.uniform(1.5,2.5),
            max_force=0.5,
            color=(random.randint(50,200), random.randint(50,200), random.randint(50,200))
        )
        self.agents.append(agent)
        self.next_agent_id +=1
        return agent

    def spawn_agents(self, count:int):
        for _ in range(count):
            self.spawn_agent()

    def remove_agent(self, agent_id:int):
        self.agents = [a for a in self.agents if a.id != agent_id]

    def _get_neighbors(self, agent:Agent) -> List[Agent]:
        return [other for other in self.agents if other.id != agent.id and (other.position - agent.position).magnitude() < agent.perception_radius]

    def update(self, dt:float):
        for agent in self.agents:
            agent.trajectory.append(agent.position.to_tuple())
            neighbors = self._get_neighbors(agent)
            seek_force = self.behavior_system.seek(agent, agent.destination)
            avoid_force = self.behavior_system.collision_avoidance(agent, neighbors, self.environment.obstacles)
            cohesion_force = self.behavior_system.cohesion(agent, neighbors)
            separation_force = self.behavior_system.separation(agent, neighbors)
            weights = self.behavior_weights.get(agent.state, self.behavior_weights[AgentState.NORMAL])
            total_force = (seek_force*weights['seek'] + avoid_force*weights['avoid'] + cohesion_force*weights['cohesion'] + separation_force*weights['separation'])
            acceleration = total_force / agent.mass
            agent.velocity += acceleration * dt
            if agent.velocity.magnitude() > agent.max_speed:
                agent.velocity = agent.velocity.normalize() * agent.max_speed
            new_pos = agent.position + agent.velocity*dt
            if self.environment.is_valid_position(new_pos, agent.personal_space_radius):
                agent.position = new_pos
            if (agent.destination - agent.position).magnitude() < 5:
                if self.environment.goals:
                    agent.destination = random.choice(self.environment.goals)

    def get_density_map(self, resolution:int=20) -> np.ndarray:
        density = np.zeros((resolution,resolution))
        for agent in self.agents:
            x = int(agent.position.x / self.environment.width * resolution)
            y = int(agent.position.y / self.environment.height * resolution)
            x = min(x, resolution-1)
            y = min(y, resolution-1)
            density[x,y] +=1
        return density