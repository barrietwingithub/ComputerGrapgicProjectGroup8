import pygame
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from collections import deque
from enum import Enum



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
        if scalar != 0:
            return Vector2D(self.x / scalar, self.y / scalar)
        return Vector2D(0, 0)

    def magnitude(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector2D(0, 0)

    def to_tuple(self):
        return (self.x, self.y)


class AgentState(Enum):
    NORMAL = "normal"
    PANIC = "panic"
    FOLLOWING = "following"
    AVOIDING = "avoiding"
    GOAL_SEEKING = "goal_seeking"
    WANDERING = "wandering"


@dataclass
class Obstacle:
    position: Vector2D
    width: float
    height: float
    color: Tuple[int, int, int] = (100, 100, 100)


# ==========================================================
# Agent Class
# ==========================================================
@dataclass
class Agent:
    id: int
    position: Vector2D
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    destination: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    personal_space_radius: float = 15.0
    max_speed: float = 2.0
    max_force: float = 0.5
    mass: float = 1.0
    state: AgentState = AgentState.GOAL_SEEKING
    perception_radius: float = 60.0
    path: List[Vector2D] = field(default_factory=list)
    color: Tuple[int, int, int] = field(
        default_factory=lambda: (100, 150, 200)
    )
    trajectory: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )
    
    # Personality traits
    personality: dict = field(default_factory=dict)
    
    # Class variables for screen dimensions
    width: int = 1200
    height: int = 800

    def __post_init__(self):
        # Add a small random offset
        self.phase_offset = hash(self.id) % 100 / 100
        
        # Generate unique color
        hue = (self.id * 37) % 360
        self.original_color = self.hsv_to_rgb(hue, 0.9, 0.9)
        self.color = self.original_color
        
        # Random personality traits
        self.personality = {
            'speed_factor': random.uniform(0.8, 1.2),
            'caution': random.uniform(0.8, 1.8),
            'social': random.uniform(0.0, 0.5),
            'curiosity': random.uniform(0.0, 0.3),
            'patience': random.uniform(0.5, 1.5)
        }
        
        # Apply speed factor
        self.max_speed *= self.personality['speed_factor']
        
        self.time_since_dest_change = 0

    def hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV color values to RGB."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )

    def update(
        self,
        neighbors: List['Agent'],
        obstacles: List[Obstacle],
        destination: Optional[Vector2D] = None
    ):
        """Update agent position and state."""
        target = destination if destination else self.destination
        
        # Calculate all behavior forces
        seek_force = self.seek(target)
        avoidance_force = self.collision_avoidance(neighbors, obstacles)
        separation_force = self.separation(neighbors)
        
        # Combine forces with MUCH STRONGER avoidance
        total_force = self.combine_behaviors(
            seek_force,
            avoidance_force,
            separation_force
        )
        
        # Apply force to velocity
        self.velocity += total_force
        
        # Limit speed
        if self.velocity.magnitude() > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed
        
        # Update position
        self.position += self.velocity
        
        # Store trajectory
        self.trajectory.append((self.position.x, self.position.y))
        
        # Update state based on proximity
        self._update_state(neighbors)
        
        # Check if destination reached
        self._check_destination_reached()
        
        self.time_since_dest_change += 1

    def seek(self, target: Vector2D) -> Vector2D:
        """Seek behavior - move towards target."""
        desired = (target - self.position).normalize() * self.max_speed
        steer = (desired - self.velocity)
        if steer.magnitude() > self.max_force:
            steer = steer.normalize() * self.max_force
        return steer

    def collision_avoidance(
        self,
        neighbors: List['Agent'],
        obstacles: List[Obstacle]
    ) -> Vector2D:
        """SUPER STRONG collision avoidance."""
        avoidance = Vector2D(0, 0)
        
        for other in neighbors:
            if other.id == self.id:
                continue
            
            diff = self.position - other.position
            dist = diff.magnitude()
            
            # Safe distance is personal space plus a buffer
            safe_distance = (
                self.personal_space_radius
                + other.personal_space_radius
                + 5
            )
            
            if dist < safe_distance and dist > 0.01:
                # Direction away from other agent
                direction = diff.normalize()
                
                # Calculate how close they are to safe distance
                closeness = (safe_distance - dist) / safe_distance
                
                # EXPONENTIAL force - VERY strong when close
                if dist < self.personal_space_radius:
                    # Inside personal space - MAXIMUM force
                    strength = (
                        10.0
                        * self.max_force
                        * self.personality['caution']
                    )
                    # Flash red when colliding
                    self.color = (255, 100, 100)
                else:
                    # Normal avoidance
                    strength = (
                        closeness
                        * self.max_force
                        * 8.0
                        * self.personality['caution']
                    )
                
                avoidance += direction * strength
        
        return avoidance

    def separation(self, neighbors: List['Agent']) -> Vector2D:
        """Strong separation force."""
        separation_force = Vector2D(0, 0)
        
        for other in neighbors:
            if other.id != self.id:
                diff = self.position - other.position
                dist = diff.magnitude()
                
                # Larger separation distance
                separation_distance = self.personal_space_radius * 3.5
                
                if dist < separation_distance and dist > 0.01:
                    direction = diff.normalize()
                    # Force increases exponentially as distance decreases
                    force_magnitude = (
                        ((separation_distance - dist) / separation_distance)
                        ** 2
                        * self.max_force
                        * 10.0
                    )
                    separation_force += direction * force_magnitude
        
        # Allow very strong separation force
        if separation_force.magnitude() > self.max_force * 3:
            separation_force = (
                separation_force.normalize()
                * self.max_force
                * 3
            )
            
        return separation_force

    def combine_behaviors(
        self,
        seek_force: Vector2D,
        avoidance_force: Vector2D,
        separation_force: Vector2D
    ) -> Vector2D:
        """
        Combine forces with AVOIDANCE as TOP PRIORITY.
        """
        # Weights - AVOIDANCE is king!
        SEEK_WEIGHT = 1.0
        AVOIDANCE_WEIGHT = 8.0
        SEPARATION_WEIGHT = 5.0
        
        # Start with seek force
        total_force = seek_force * SEEK_WEIGHT
        
        # Add avoidance forces with much higher weights
        total_force += avoidance_force * AVOIDANCE_WEIGHT
        total_force += separation_force * SEPARATION_WEIGHT
        
        # Allow higher total force for emergency avoidance
        max_allowed_force = self.max_force * 5
        if total_force.magnitude() > max_allowed_force:
            total_force = total_force.normalize() * max_allowed_force
            
        return total_force

    def _update_state(self, neighbors: List['Agent']):
        """Update agent state with visual feedback."""
        closest_dist = float('inf')
        
        for other in neighbors:
            if other.id != self.id:
                dist = (other.position - self.position).magnitude()
                closest_dist = min(closest_dist, dist)
                
                # If VERY close, force them apart
                if dist < self.personal_space_radius:
                    # Emergency push
                    push_dir = (self.position - other.position).normalize()
                    self.position += push_dir * 2.0
        
        # Update state and color based on proximity
        if closest_dist < self.personal_space_radius:
            self.state = AgentState.PANIC
            self.color = (255, 0, 0)  # Bright red
        elif closest_dist < self.personal_space_radius * 1.5:
            self.state = AgentState.AVOIDING
            self.color = (255, 150, 150)  # Light red
        else:
            self.state = AgentState.GOAL_SEEKING
            self.color = self.original_color

    def _check_destination_reached(self, threshold: float = 30.0):
        """Check if destination reached and set new one."""
        dist_to_dest = (self.destination - self.position).magnitude()
        
        if dist_to_dest < threshold:
            # Pick new destination in a different quadrant
            quadrant = random.choice(['NW', 'NE', 'SW', 'SE'])
            
            if quadrant == 'NW':
                x = random.uniform(50, self.width // 2 - 50)
                y = random.uniform(50, self.height // 2 - 50)
            elif quadrant == 'NE':
                x = random.uniform(self.width // 2 + 50, self.width - 50)
                y = random.uniform(50, self.height // 2 - 50)
            elif quadrant == 'SW':
                x = random.uniform(50, self.width // 2 - 50)
                y = random.uniform(self.height // 2 + 50, self.height - 50)
            else:
                x = random.uniform(self.width // 2 + 50, self.width - 50)
                y = random.uniform(self.height // 2 + 50, self.height - 50)
            
            self.destination = Vector2D(x, y)
            self.time_since_dest_change = 0


# ==========================================================
# Crowd Simulation Class
# ==========================================================
class CrowdSimulation:
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        
        self.width = width
        self.height = height
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []
        self.running = True
        self.paused = False
        self.show_destinations = True
        self.show_trajectories = True
        
        # Set up display with RESIZABLE window
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.RESIZABLE  # type: ignore
        )
        pygame.display.set_caption(
            "Real-Time Crowd Simulation Using Agent-Based Models"
        )
        
        # Font for instructions
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.clock = pygame.time.Clock()
        
        # Store screen dimensions for agents
        Agent.width = width
        Agent.height = height
        
        # Spawn agents
        self.spawn_agents(50)
        
        print("=" * 60)
        print("REAL-TIME CROWD SIMULATION")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Add 5 agents")
        print("  C - Clear all agents")
        print("  P - Pause/Resume")
        print("  D - Toggle destinations")
        print("  T - Toggle trajectories")
        print("  + / - - Increase/Decrease window size")
        print("  ESC - Exit")
        print("=" * 60)

    def spawn_agents(self, num: int):
        """Spawn new agents at valid positions."""
        for i in range(num):
            # Try to find valid spawn position
            max_attempts = 100
            for attempt in range(max_attempts):
                pos = Vector2D(
                    random.uniform(50, self.width - 50),
                    random.uniform(50, self.height - 50)
                )
                
                valid = True
                for agent in self.agents:
                    dist = math.hypot(
                        pos.x - agent.position.x,
                        pos.y - agent.position.y
                    )
                    if dist < 40:
                        valid = False
                        break
                
                if valid or attempt == max_attempts - 1:
                    break
            
            # Create agent with destination in random quadrant
            quadrant = random.choice(['NW', 'NE', 'SW', 'SE'])
            
            if quadrant == 'NW':
                dest_x = random.uniform(50, self.width // 2 - 50)
                dest_y = random.uniform(50, self.height // 2 - 50)
            elif quadrant == 'NE':
                dest_x = random.uniform(
                    self.width // 2 + 50,
                    self.width - 50
                )
                dest_y = random.uniform(50, self.height // 2 - 50)
            elif quadrant == 'SW':
                dest_x = random.uniform(50, self.width // 2 - 50)
                dest_y = random.uniform(
                    self.height // 2 + 50,
                    self.height - 50
                )
            else:
                dest_x = random.uniform(
                    self.width // 2 + 50,
                    self.width - 50
                )
                dest_y = random.uniform(
                    self.height // 2 + 50,
                    self.height - 50
                )
            
            agent = Agent(
                id=len(self.agents) + i,
                position=pos,
                destination=Vector2D(dest_x, dest_y)
            )
            self.agents.append(agent)

    def update_agents(self):
        """Update all agents."""
        if self.paused:
            return
        
        for agent in self.agents:
            agent.update(self.agents, self.obstacles)
            
            # Keep within bounds
            agent.position.x = max(5, min(self.width - 5, agent.position.x))
            agent.position.y = max(5, min(self.height - 5, agent.position.y))

    def draw_human(self, agent: Agent):
        """Draw a simple human figure."""
        x = int(agent.position.x)
        y = int(agent.position.y)

        head = 6
        body = 18
        arm = 10
        leg = 14

        # Draw glow effect when colliding
        if agent.state == AgentState.PANIC:
            # Draw a bright red glow
            glow_surface = pygame.Surface(
                (self.width, self.height),
                pygame.SRCALPHA
            )
            for radius in [20, 15, 10]:
                alpha = 100 - radius * 5
                pygame.draw.circle(
                    glow_surface,
                    (255, 0, 0, alpha),
                    (x, y),
                    radius
                )
            self.screen.blit(glow_surface, (0, 0))
        
        # Draw agent
        pygame.draw.circle(self.screen, agent.color, (x, y), head)
        pygame.draw.circle(self.screen, (0, 0, 0), (x, y), head, 1)
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (x, y + head), (x, y + body), 2
        )
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (x - arm, y + 10), (x + arm, y + 10), 2
        )
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (x, y + body), (x - leg, y + body + leg), 2
        )
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (x, y + body), (x + leg, y + body + leg), 2
        )

    def draw(self):
        """Draw the entire simulation."""
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        for i in range(0, self.width, 50):
            pygame.draw.line(
                self.screen, (240, 240, 240),
                (i, 0), (i, self.height), 1
            )
        for i in range(0, self.height, 50):
            pygame.draw.line(
                self.screen, (240, 240, 240),
                (0, i), (self.width, i), 1
            )
        
        # Draw quadrant lines
        pygame.draw.line(
            self.screen, (200, 200, 200),
            (self.width // 2, 0), (self.width // 2, self.height), 2
        )
        pygame.draw.line(
            self.screen, (200, 200, 200),
            (0, self.height // 2), (self.width, self.height // 2), 2
        )
        
        # Draw agents
        for agent in self.agents:
            if self.show_trajectories and len(agent.trajectory) > 1:
                points = [(int(x), int(y)) for x, y in agent.trajectory]
                if len(points) > 1:
                    pygame.draw.lines(
                        self.screen, agent.color,
                        False, points, 1
                    )
            
            self.draw_human(agent)
            
            if self.show_destinations:
                dest_x, dest_y = (
                    int(agent.destination.x),
                    int(agent.destination.y)
                )
                pygame.draw.circle(
                    self.screen, agent.color,
                    (dest_x, dest_y), 4
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0),
                    (dest_x, dest_y), 4, 1
                )
        
        # Draw instructions
        y_offset = 10
        texts = [
            f"Agents: {len(self.agents)}",
            "SPACE: Add 5 | C: Clear | P: Pause | D: Toggle Dest | "
            "T: Toggle Traj | +/-: Resize | ESC: Exit"
        ]
        for text in texts:
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            self.screen.blit(
                pause_text,
                (self.width // 2 - 40, 10)
            )
        
        pygame.display.flip()

    def run(self):
        """Main simulation loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.spawn_agents(5)
                    elif event.key == pygame.K_c:
                        self.agents.clear()
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_d:
                        self.show_destinations = not self.show_destinations
                    elif event.key == pygame.K_t:
                        self.show_trajectories = not self.show_trajectories
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        # Increase window size
                        self.width += 100
                        self.height += 100
                        self.screen = pygame.display.set_mode(
                            (self.width, self.height),
                            pygame.RESIZABLE  # type: ignore
                        )
                        Agent.width = self.width
                        Agent.height = self.height
                    elif event.key == pygame.K_MINUS:
                        # Decrease window size
                        self.width = max(800, self.width - 100)
                        self.height = max(600, self.height - 100)
                        self.screen = pygame.display.set_mode(
                            (self.width, self.height),
                            pygame.RESIZABLE  # type: ignore
                        )
                        Agent.width = self.width
                        Agent.height = self.height
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.width, self.height = event.size
                    self.screen = pygame.display.set_mode(
                        (self.width, self.height),
                        pygame.RESIZABLE  # type: ignore
                    )
                    Agent.width = self.width
                    Agent.height = self.height

            self.update_agents()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


# # ==========================================================
# # Main Entry Point
# # ==========================================================
# if __name__ == "__main__":
#     sim = CrowdSimulation(1200, 800)
#     sim.run()