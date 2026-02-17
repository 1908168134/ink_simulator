# src/interactive_player.py
import cv2, pygame, sys

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    pygame.init()
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    paused = False
    speed = 1.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    speed = min(3.0, speed + 0.5)
                if event.key == pygame.K_DOWN:
                    speed = max(0.5, speed - 0.5)

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

        pygame.display.flip()
        clock.tick(fps * speed)

    cap.release()
    pygame.quit()