import time
import sys
import pygame

GRAVITY = 9.8
DT = 0.01
WIDTH, HEIGHT = 600, 400
PIXELS_PER_METER = 100
radius = 10

class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(velocity)
        self.force = pygame.Vector2(0, 0)

    def apply_force(self, force):
        self.force += force

    def update(self):
        acceleration = self.force / self.mass
        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.force = pygame.Vector2(0, 0)

class Slider:
    def __init__(self, width, acceleration, position):
        self.width = width
        self.acceleration = acceleration
        self.velocity = 0
        self.max_speed = 10
        self.position = position

    def update(self, direction):
        if direction == "right":
            self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        elif direction == "left":
            self.velocity = max(self.velocity - self.acceleration, -self.max_speed)
        else:
            self.velocity *= 0.9  # friction

        self.position += self.velocity
        # Clamp within screen
        self.position = max(0, min(WIDTH - self.width, self.position))


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A simple ricochet game")
clock = pygame.time.Clock()

ball = Particle(mass=1.0, position=(WIDTH / 2, 0), velocity=(300, 0))
slide = Slider(width=100, acceleration=0.6, position=WIDTH // 2 - 50)

running = True

while running:
    screen.fill((255, 255, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Key hold movement (smoother)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        direction = "left"
    elif keys[pygame.K_RIGHT]:
        direction = "right"
    else:
        direction = None

    slide.update(direction)

    # Apply gravity
    ball.apply_force(pygame.Vector2(0, ball.mass * GRAVITY))
    ball.update()

    # Collision with screen boundaries (left, right, top)
    if ball.position.x - radius < 0 or ball.position.x + radius > WIDTH:
        ball.velocity.x *= -1
        ball.position.x = max(radius, min(WIDTH - radius, ball.position.x))

    if ball.position.y - radius < 0:
        ball.velocity.y *= -1
        ball.position.y = radius

    # Collision with slider/platform
    platform_rect = pygame.Rect(int(slide.position), HEIGHT - 10, slide.width, 10)
    ball_rect = pygame.Rect(int(ball.position.x - radius), int(ball.position.y - radius), radius * 2, radius * 2)

    if ball_rect.colliderect(platform_rect) and ball.velocity.y > 0:
        ball.velocity.y *= -1
        ball.position.y = HEIGHT - 10 - radius

    # Collision with bottom (fallback if it misses the slider)
    if ball.position.y + radius > HEIGHT:
        ball.velocity.y *= -1
        ball.position.y = HEIGHT - radius

    # Draw ball
    pygame.draw.circle(
        screen,
        (0, 200, 255),
        (int(ball.position.x), int(ball.position.y)),
        radius
    )

    # Draw slider
    pygame.draw.rect(screen, (0, 0, 0), platform_rect)

    pygame.display.flip()
    clock.tick(1 / DT)

pygame.quit()
sys.exit()
