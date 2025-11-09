import math
import random
import pygame
import numpy as np

class DodgerEnv:
    """A tiny arcade-like environment:
    - Player moves horizontally at the bottom, dodging falling obstacles.
    - Vector observation, discrete actions.
    """

    metadata = {"render_fps": 60}

    def __init__(self, width=400, height=600, seed=0, max_steps=2000, render_mode=None):
        self.width = width
        self.height = height
        self.seed(seed)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Player
        self.player_w = 30
        self.player_h = 12
        self.player_speed = 6.0

        # Obstacles
        self.spawn_interval = 40  # steps between spawns
        self.obs_w = 28
        self.obs_h = 28
        self.obs_min_v = 3.0
        self.obs_max_v = 7.0

        self.reset()

    def seed(self, seed=0):
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)

    def _spawn_obstacle(self):
        x = self.np_random.randint(self.obs_w, self.width - self.obs_w)
        vy = self.np_random.uniform(self.obs_min_v, self.obs_max_v) * self.speed_scale
        rect = pygame.Rect(int(x), -self.obs_h, self.obs_w, self.obs_h)
        return {"rect": rect, "vy": vy, "passed": False}

    def _nearest_obstacle_features(self):
        # pick the lowest (closest to player along y)
        if not self.obstacles:
            return np.zeros(5, dtype=np.float32)
        obs = min(self.obstacles, key=lambda o: abs((self.height - self.player_rect.y) - o["rect"].y))
        o = obs["rect"]
        # normalize features
        nx = o.centerx / self.width
        ny = o.centery / self.height
        nvy = obs["vy"] / (self.obs_max_v * 2.0)  # scale
        gap_left = self.player_rect.left / self.width
        gap_right = (self.width - self.player_rect.right) / self.width
        return np.array([nx, ny, nvy, gap_left, gap_right], dtype=np.float32)

    def _get_obs(self):
        px = self.player_rect.centerx / self.width
        pvx = (self.player_vx + self.player_speed) / (2 * self.player_speed)  # map [-speed, speed] -> [0,1]
        nearest = self._nearest_obstacle_features()
        t = (self.steps_since_spawn % self.spawn_interval) / max(1, self.spawn_interval)
        speed_scale = (self.speed_scale - 0.5) / 1.5  # if scale in [0.5,2.0] -> [-0.33, 1.0] then normalize to [0,1]
        speed_scale = (speed_scale + 0.33) / 1.33
        return np.concatenate([np.array([px, pvx], dtype=np.float32), nearest, np.array([t, speed_scale], dtype=np.float32)])

    def reset(self):
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("DodgerEnv")
            self.clock = pygame.time.Clock()

        self.player_rect = pygame.Rect(self.width // 2 - self.player_w // 2, self.height - 40, self.player_w, self.player_h)
        self.player_vx = 0.0
        self.obstacles = []
        self.steps = 0
        self.episode_reward = 0.0
        self.steps_since_spawn = 0
        self.speed_scale = 1.0
        return self._get_obs()

    def step(self, action):
        reward = 0.0
        done = False

        # --- Handle action: 0 stay, 1 left, 2 right
        if action == 1:
            self.player_vx = -self.player_speed
            reward -= 0.01
        elif action == 2:
            self.player_vx = self.player_speed
            reward -= 0.01
        else:
            self.player_vx = 0.0

        # Update player position
        self.player_rect.x += int(self.player_vx)
        self.player_rect.x = max(0, min(self.player_rect.x, self.width - self.player_w))

        # Spawn logic (slightly faster over time)
        if self.steps % 600 == 0 and self.steps > 0:
            self.speed_scale = min(2.0, self.speed_scale + 0.1)

        self.steps_since_spawn += 1
        if self.steps_since_spawn >= self.spawn_interval:
            self.obstacles.append(self._spawn_obstacle())
            self.steps_since_spawn = 0

        # Update obstacles
        for obs in self.obstacles:
            obs["rect"].y += int(obs["vy"])
            # Score if passed player line
            if not obs["passed"] and obs["rect"].bottom > self.player_rect.top:
                obs["passed"] = True

        # Remove off-screen & bonus for clean pass
        kept = []
        for obs in self.obstacles:
            if obs["rect"].top > self.height:
                reward += 2.0
            else:
                kept.append(obs)
        self.obstacles = kept

        # Collision check
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs["rect"]):
                reward -= 1.0
                done = True
                break

        # Living reward
        reward += 1.0
        self.episode_reward += reward
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {"episode_reward": self.episode_reward, "steps": self.steps}
        if self.render_mode == "human":
            self.render()
        return obs, reward, done, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((15, 18, 25))
        # Draw player
        pygame.draw.rect(self.screen, (240, 240, 240), self.player_rect)
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100, 200, 255), obs["rect"])
        # HUD
        font = pygame.font.SysFont(None, 22)
        txt = font.render(f"R:{self.episode_reward:.1f}  t:{self.steps}", True, (200, 200, 200))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
