from typing import List
from agent import Agent, Vector2D
from environment import Obstacle


class BehaviorSystem:
    """Handles agent behaviors: seek, flee, collision avoidance,
      cohesion, separation"""

    @staticmethod
    def limit_force(force: Vector2D, max_force: float) -> Vector2D:
        """Utility to cap force magnitude"""
        if force.magnitude() > max_force:
            return force.normalize() * max_force
        return force

    @staticmethod
    def seek(agent: Agent, target: Vector2D) -> Vector2D:
        desired = (target - agent.position).normalize() * agent.max_speed
        steer = desired - agent.velocity
        return BehaviorSystem.limit_force(steer, agent.max_force)

    @staticmethod
    def flee(agent: Agent, threat: Vector2D) -> Vector2D:
        desired = (agent.position - threat).normalize() * agent.max_speed
        steer = desired - agent.velocity
        return BehaviorSystem.limit_force(steer, agent.max_force)

    @staticmethod
    def collision_avoidance(agent: Agent, neighbors: List[Agent], 
                            obstacles: List[Obstacle]) -> Vector2D:
        avoidance = Vector2D(0, 0)

        for other in neighbors:
            if other.id == agent.id:
                continue

            diff = agent.position - other.position
            dist = diff.magnitude()

            safe_distance = agent.personal_space_radius  + other.personal_space_radius

            if 0.01 < dist < safe_distance:
                direction = diff.normalize()

                if dist < safe_distance * 0.5:
                    strength = 2.0 * agent.max_force
                else:
                    strength = ((safe_distance - dist) / safe_distance) * agent.max_force * 1.5

                avoidance += direction * strength

        for obs in obstacles:
            obs_center = obs.position
            diff = agent.position - obs_center
            dist = diff.magnitude()

            min_dist = agent.personal_space_radius + max(obs.width, obs.height) / 2

            if 0.01 < dist < min_dist:
                direction = diff.normalize()
                strength = ((min_dist - dist) / min_dist) * agent.max_force * 3
                avoidance += direction * strength

        return BehaviorSystem.limit_force(avoidance, agent.max_force * 2)

    @staticmethod
    def cohesion(agent: Agent, neighbors: List[Agent]) -> Vector2D:
        center = Vector2D(0, 0)
        count = 0

        for other in neighbors:
            if other.id != agent.id:
                dist = (other.position - agent.position).magnitude()
                if dist < agent.perception_radius:
                    center += other.position
                    count += 1

        if count > 0:
            center = center / count
            return BehaviorSystem.seek(agent, center) * 0.2

        return Vector2D(0, 0)

    @staticmethod
    def separation(agent: Agent, neighbors: List[Agent]) -> Vector2D:
        force = Vector2D(0, 0)

        for other in neighbors:
            if other.id != agent.id:
                diff = agent.position - other.position
                dist = diff.magnitude()

                separation_distance = agent.personal_space_radius * 2.5

                if 0.01 < dist < separation_distance:
                    direction = diff.normalize()
                    strength = (separation_distance - dist) / (dist + 0.1) * agent.max_force
                    force += direction * strength

        return BehaviorSystem.limit_force(force, agent.max_force)

    @staticmethod
    def combine_behaviors(agent: Agent,
                          seek_force: Vector2D,
                          avoidance_force: Vector2D,
                          separation_force: Vector2D,
                          cohesion_force: Vector2D) -> Vector2D:

        SEEK_WEIGHT = 1.0
        AVOIDANCE_WEIGHT = 2.0
        SEPARATION_WEIGHT = 1.5
        COHESION_WEIGHT = 0.3

        total = Vector2D(0, 0)
        total += seek_force * SEEK_WEIGHT
        total += avoidance_force * AVOIDANCE_WEIGHT
        total += separation_force * SEPARATION_WEIGHT
        total += cohesion_force * COHESION_WEIGHT

        return BehaviorSystem.limit_force(total, agent.max_force * 2)
