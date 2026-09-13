"""
main.py

Main entry point for Pygame Ganapati Vector Drawing Application.
Renders 3x supersampled vector line art (2700x3300) and downsamples to logical resolution (900x1100).
Clean continuous line art rendering without any debug points, trails, or extra circles.
Handles keyboard controls:
- SPACE : Restart animation
- R     : Reset drawing
- + / = : Increase speed
- -     : Decrease speed
- ESC   : Exit
"""

import sys
import pygame
from config import (
    LOGICAL_WIDTH, LOGICAL_HEIGHT, SUPERSAMPLE_WIDTH, SUPERSAMPLE_HEIGHT,
    BACKGROUND_COLOR, HUD_TEXT_COLOR, HUD_BG_COLOR, FPS
)
from paths import get_ganapati_paths
from drawing import draw_ganapati, draw_pointer
from animation import DrawingAnimator


def render_hud(surface, animator, font_large, font_small):
    """
    Renders top & bottom subtle overlay HUD info (Step name, speed multiplier, hints).
    """
    step_name, total_pct = animator.get_active_step_info()

    # Top HUD Bar
    hud_bar = pygame.Surface((LOGICAL_WIDTH - 40, 45), pygame.SRCALPHA)
    pygame.draw.rect(hud_bar, (245, 243, 235, 220), (0, 0, LOGICAL_WIDTH - 40, 45), border_radius=10)
    pygame.draw.rect(hud_bar, (200, 195, 180), (0, 0, LOGICAL_WIDTH - 40, 45), width=1, border_radius=10)
    
    # Step Text
    txt_step = font_large.render(f"Drawing: {step_name} ({total_pct}%)", True, (30, 30, 30))
    hud_bar.blit(txt_step, (20, 10))

    # Speed Text
    speed_str = f"Speed: {animator.draw_speed:.2f}x"
    txt_speed = font_large.render(speed_str, True, (150, 40, 40))
    hud_bar.blit(txt_speed, (LOGICAL_WIDTH - 200, 10))

    surface.blit(hud_bar, (20, 20))

    # Bottom Control Hints Panel
    hints_bar = pygame.Surface((LOGICAL_WIDTH - 40, 35), pygame.SRCALPHA)
    pygame.draw.rect(hints_bar, (245, 243, 235, 200), (0, 0, LOGICAL_WIDTH - 40, 35), border_radius=8)
    
    hints_txt = font_small.render("[SPACE] Restart  |  [R] Reset  |  [+] Speed Up  |  [-] Speed Down  |  [ESC] Exit", True, (80, 80, 80))
    hints_rect = hints_txt.get_rect(center=((LOGICAL_WIDTH - 40) // 2, 17))
    hints_bar.blit(hints_txt, hints_rect)

    surface.blit(hints_bar, (20, LOGICAL_HEIGHT - 55))


def main():
    pygame.init()
    pygame.font.init()

    # Set up Pygame window
    screen = pygame.display.set_mode((LOGICAL_WIDTH, LOGICAL_HEIGHT))
    pygame.display.set_caption("Ganapati Animated Vector Sketch - Python Pygame (3x Supersampled)")
    clock = pygame.time.Clock()

    # Create 3x Supersample Surface (2700x3300) for silky anti-aliased curves
    supersample_surface = pygame.Surface((SUPERSAMPLE_WIDTH, SUPERSAMPLE_HEIGHT))

    # Load paths and initialize animator
    try:
        paths_tuple = get_ganapati_paths()
        paths = paths_tuple[0] if isinstance(paths_tuple, tuple) else paths_tuple
    except Exception as e:
        print(f"Error loading path data: {e}")
        pygame.quit()
        sys.exit(1)

    animator = DrawingAnimator(paths)

    # Load fonts
    font_large = pygame.font.SysFont("Segoe UI", 18, bold=True)
    font_small = pygame.font.SysFont("Segoe UI", 13, bold=False)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    animator.restart()
                elif event.key == pygame.K_r:
                    animator.reset()
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    animator.increase_speed()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    animator.decrease_speed()

        # Update animation
        animator.update(dt)

        # Render Ganapati vector line art on 3x Supersample Surface (2700x3300)
        draw_ganapati(
            supersample_surface,
            paths=paths,
            active_path_idx=animator.current_path_idx,
            active_path_progress=animator.current_path_progress
        )

        # Draw ONE drawing pointer at current drawing tip if not completed
        if not animator.is_completed:
            draw_pointer(supersample_surface, animator.pointer_pos)

        # Downsample 3x supersample surface to logical display surface (900x1100)
        downsampled_surface = pygame.transform.smoothscale(
            supersample_surface, (LOGICAL_WIDTH, LOGICAL_HEIGHT)
        )

        # Blit downsampled image onto screen
        screen.blit(downsampled_surface, (0, 0))

        # Render HUD overlay on display
        render_hud(screen, animator, font_large, font_small)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
