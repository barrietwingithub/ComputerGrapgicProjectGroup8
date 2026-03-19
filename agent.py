from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
from collections import deque
import math

class AgentState(Enum):
    NORMAL = "normal"
    PANIC = "panic"
    FOLLOWING = "following"
    AVOIDING = "avoiding"
    GOAL_SEEKING = "goal_seeking"

@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        mag = self.magnitude()
        return self / mag if mag > 0 else Vector2D(0, 0)

    def to_tuple(self):
        return (self.x, self.y)

@dataclass
class Agent:
    id: int
    position: Vector2D
    velocity: Vector2D
    destination: Vector2D
    personal_space_radius: float = 1.0
    max_speed: float = 2.0
    max_force: float = 0.5
    mass: float = 1.0
    state: AgentState = AgentState.NORMAL
    perception_radius: float = 5.0
    path: List[Vector2D] = field(default_factory=list)
    color: Tuple[int, int, int] = field(default_factory=lambda: (100, 150, 200))

    def __post_init__(self):
        self.trajectory = deque(maxlen=50)
        self.phase_offset = hash(self.id) % 100 / 100
        
    def update(self, neighbors: List['Agent'], obstacles: List, destination: Optional[Vector2D] = None):
        """
        Update agent position using behavior system
        Call this every frame instead of manually updating position
        """
        from behaviour import BehaviorSystem
        
        target = destination if destination else self.destination
        
        seek_force = BehaviorSystem.seek(self, target)
        avoidance_force = BehaviorSystem.collision_avoidance(self, neighbors, obstacles)
        separation_force = BehaviorSystem.separation(self, neighbors)
        cohesion_force = BehaviorSystem.cohesion(self, neighbors)
        
        total_force = BehaviorSystem.combine_behaviors(
            self, seek_force, avoidance_force, separation_force, cohesion_force
        )
        
        self.velocity += total_force
        
        if self.velocity.magnitude() > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed
        
        self.position += self.velocity
        
        self.trajectory.append((self.position.x, self.position.y))
        
        self._update_state(neighbors)
        
        self._check_destination_reached()
        
    def _update_state(self, neighbors: List['Agent']):
        """Update agent state based on nearby agents"""
        close_agents = 0
        very_close_agents = 0
        
        for other in neighbors:
            if other.id != self.id:
                dist = (other.position - self.position).magnitude()
                
                if dist < self.personal_space_radius:
                    very_close_agents += 1
                elif dist < self.personal_space_radius * 2:
                    close_agents += 1
        
        if very_close_agents > 0:
            self.state = AgentState.AVOIDING
            self.color = (200, 50, 50)
        elif close_agents > 2:
            self.state = AgentState.AVOIDING
            self.color = (255, 150, 150)
        else:
            self.state = AgentState.GOAL_SEEKING
            self.color = (100, 150, 200)
    
    def _check_destination_reached(self, threshold: float = 0.5):
        """Check if agent has reached its destination"""
        dist_to_dest = (self.destination - self.position).magnitude()
        
        if dist_to_dest < threshold:
            
            pass
    
    def get_predicted_position(self, time_ahead: float = 1.0) -> Vector2D:
        """
        Predict where this agent will be in 'time_ahead' seconds
        Useful for advanced collision avoidance
        """
        return self.position + self.velocity * time_ahead
    
    def apply_avoidance_force(self, force: Vector2D):
        """Directly apply an avoidance force (useful for external obstacles)"""
        self.velocity += force
        
        if self.velocity.magnitude() > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed
    
    def set_destination(self, new_destination: Vector2D):
        """Set a new destination for the agent"""
        self.destination = new_destination
        
    def stop(self):
        """Bring agent to a stop"""
        self.velocity = Vector2D(0, 0)
        
    def is_moving(self) -> bool:
        """Check if agent is moving"""
        return self.velocity.magnitude() > 0.01
    
    def distance_to(self, other: 'Agent') -> float:
        """Calculate distance to another agent"""
        return (other.position - self.position).magnitude()
    
    def direction_to(self, other: 'Agent') -> Vector2D:
        """Get normalized direction to another agent"""
        diff = other.position - self.position
        return diff.normalize()
    
    def __repr__(self):
        return f"Agent(id={self.id}, pos=({self.position.x:.1f}, {self.position.y:.1f}), state={self.state.value})"