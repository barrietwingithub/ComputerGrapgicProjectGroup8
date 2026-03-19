import pygame
import numpy as np
from environment import Environment, ObstacleType
from agent_manager import AgentManager
from stats import StatisticsCollector

class Visualization:
    """Handles real-time rendering using Pygame"""

    def __init__(self, environment: Environment, width:int=800, height:int=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width,height))
        pygame.display.set_caption("Real-Time Crowd Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None,24)
        self.show_heatmap = False
        self.show_paths = True
        self.show_stats = True

    def draw_environment(self, env:Environment):
        self.screen.fill((255,255,255))
        for obs in env.obstacles:
            color = (100,100,100)
            if obs.type == ObstacleType.WALL:
                color = (150,150,150)
            elif obs.type == ObstacleType.DYNAMIC:
                color = (200,100,100)
            rect = obs.get_rect()
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen,(50,50,50),rect,2)

        for i, goal in enumerate(env.goals):
            pygame.draw.circle(self.screen,(0,255,0),(int(goal.x),int(goal.y)),10)
            pygame.draw.circle(self.screen,(0,200,0),(int(goal.x),int(goal.y)),10,2)
            label = self.font.render(f"G{i+1}",True,(0,100,0))
            self.screen.blit(label,(int(goal.x)-10,int(goal.y)-25))

    def draw_agents(self, agent_manager:AgentManager):
        for agent in agent_manager.agents:
            if self.show_paths and len(agent.trajectory)>1:
                points = [(int(x),int(y)) for x,y in agent.trajectory]
                pygame.draw.lines(self.screen,(200,200,200),False,points,1)
            color = agent.color
            if agent.state.name == "PANIC":
                color = (255,100,100)
            pos = (int(agent.position.x), int(agent.position.y))
            pygame.draw.circle(self.screen,color,pos,int(agent.personal_space_radius*3))
            pygame.draw.circle(self.screen,(0,0,0),pos,int(agent.personal_space_radius*3),1)
            if agent.velocity.magnitude()>0.1:
                end_pos = (int(agent.position.x+agent.velocity.x*5), int(agent.position.y+agent.velocity.y*5))
                pygame.draw.line(self.screen,(0,0,0),pos,end_pos,2)

    def draw_heatmap(self, agent_manager:AgentManager):
        density = agent_manager.get_density_map(resolution=40)
        if np.max(density)>0:
            density = density / np.max(density)
            cell_w = self.width / density.shape[0]
            cell_h = self.height / density.shape[1]
            for i in range(density.shape[0]):
                for j in range(density.shape[1]):
                    if density[i,j]>0.1:
                        alpha = min(255,int(density[i,j]*200))
                        surf = pygame.Surface((int(cell_w),int(cell_h)))
                        surf.set_alpha(alpha)
                        surf.fill((255,100,100))
                        self.screen.blit(surf,(i*cell_w,j*cell_h))

    def draw_statistics(self, stats:StatisticsCollector, agent_manager:AgentManager):
        if not self.show_stats: return
        y_offset = 10
        texts = [
            f"Agents: {len(agent_manager.agents)}",
            f"Flow Rate: {stats.get_average_flow_rate():.2f} units/s",
            f"FPS: {int(self.clock.get_fps())}",
            f"Sim Time: {stats.time_steps[-1] if stats.time_steps else 0:.1f}s"
        ]
        for text in texts:
            label = self.font.render(text, True, (0,0,0))
            self.screen.blit(label,(10,y_offset))
            y_offset += 25

    def render(self, env:Environment, agent_manager:AgentManager, stats:StatisticsCollector):
        self.draw_environment(env)
        if self.show_heatmap:
            self.draw_heatmap(agent_manager)
        self.draw_agents(agent_manager)
        self.draw_statistics(stats, agent_manager)
        instructions = self.font.render("Space: Pause | H: Heatmap | P: Paths | +/-: Speed | C: Clear | S: Spawn", True, (50,50,50))
        self.screen.blit(instructions,(10,self.height-30))
        pygame.display.flip()
        self.clock.tick(60)