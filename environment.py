import pygame
from typing import List
from agent import Vector2D
from enum import Enum
from dataclasses import dataclass, field

class ObstacleType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    WALL = "wall"
    BARRIER = "barrier"

@dataclass
class Obstacle:
    id: int
    position: Vector2D
    width: float
    height: float
    type: ObstacleType
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0, 0))

    def get_rect(self):
        return pygame.Rect(self.position.x - self.width/2, 
                           self.position.y - self.height/2,
                           self.width, self.height)

class Environment:
    def __init__(self, width: float = 800, height: float = 600):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
        self.goals: List[Vector2D] = []
        self.spawn_points: List[Vector2D] = []

        self._setup_default_environment()

    def _setup_default_environment(self):
        wall_thickness = 20
        self.obstacles.extend([
            Obstacle(len(self.obstacles), Vector2D(self.width/2, wall_thickness/2), self.width, wall_thickness, ObstacleType.WALL),
            Obstacle(len(self.obstacles), Vector2D(self.width/2, self.height - wall_thickness/2), self.width, wall_thickness, ObstacleType.WALL),
            Obstacle(len(self.obstacles), Vector2D(wall_thickness/2, self.height/2), wall_thickness, self.height, ObstacleType.WALL),
            Obstacle(len(self.obstacles), Vector2D(self.width - wall_thickness/2, self.height/2), wall_thickness, self.height, ObstacleType.WALL)
        ])
        self.obstacles.extend([
            Obstacle(len(self.obstacles), Vector2D(300, 300), 50, 50, ObstacleType.STATIC),
            Obstacle(len(self.obstacles), Vector2D(500, 400), 30, 80, ObstacleType.STATIC)
        ])
        self.goals.extend([Vector2D(750, 550), Vector2D(50, 50)])
        self.spawn_points.extend([Vector2D(100, 100), Vector2D(700, 500), Vector2D(400, 300)])

    def add_obstacle(self, obstacle: Obstacle):
        obstacle.id = len(self.obstacles)
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle_id: int):
        self.obstacles = [o for o in self.obstacles if o.id != obstacle_id]

    def is_valid_position(self, position: Vector2D, agent_radius: float = 1.0) -> bool:
        for obs in self.obstacles:
            if obs.get_rect().collidepoint(position.x, position.y):
                return False
        return True