import pygame
from agent import Vector2D
from environment import Obstacle, ObstacleType
from crowd_simulation import CrowdSimulation

class ScenarioEditor:
    """Scenario editor for custom environments"""
    def __init__(self, simulation:CrowdSimulation):
        self.simulation = simulation
        self.editing_mode = "obstacle"
        self.obstacle_size = 50
        self.running = True

    def run(self):
        print("Scenario Editor Started")
        print("O: Obstacle | G: Goal | S: Spawn | +/-: Size | D: Delete | ESC: Exit")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:
                        self.running=False
                    elif event.key==pygame.K_o:
                        self.editing_mode="obstacle"
                    elif event.key==pygame.K_g:
                        self.editing_mode="goal"
                    elif event.key==pygame.K_s:
                        self.editing_mode="spawn"
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.obstacle_size = min(200,self.obstacle_size+10)
                    elif event.key==pygame.K_MINUS:
                        self.obstacle_size=max(20,self.obstacle_size-10)
                    elif event.key==pygame.K_d:
                        if self.editing_mode=="obstacle" and self.simulation.environment.obstacles:
                            self.simulation.environment.obstacles.pop()
                        elif self.editing_mode=="goal" and self.simulation.environment.goals:
                            self.simulation.environment.goals.pop()
                        elif self.editing_mode=="spawn" and self.simulation.environment.spawn_points:
                            self.simulation.environment.spawn_points.pop()
                elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    mouse_pos = pygame.mouse.get_pos()
                    pos = Vector2D(mouse_pos[0],mouse_pos[1])
                    if self.editing_mode=="obstacle":
                        obstacle = Obstacle(len(self.simulation.environment.obstacles), pos, self.obstacle_size, self.obstacle_size, ObstacleType.STATIC)
                        self.simulation.environment.add_obstacle(obstacle)
                    elif self.editing_mode=="goal":
                        self.simulation.environment.goals.append(pos)
                    elif self.editing_mode=="spawn":
                        self.simulation.environment.spawn_points.append(pos)
            self.simulation.visualization.render(self.simulation.environment, self.simulation.agent_manager, self.simulation.stats)
            if self.editing_mode=="obstacle":
                mouse_pos = pygame.mouse.get_pos()
                preview_rect = pygame.Rect(mouse_pos[0]-self.obstacle_size/2, mouse_pos[1]-self.obstacle_size/2, self.obstacle_size, self.obstacle_size)
                pygame.draw.rect(self.simulation.visualization.screen,(100,255,100),preview_rect,2)
            pygame.display.flip()
        print("Editor Closed")