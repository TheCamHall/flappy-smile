import pygame
import sys
import random
import cv2
import numpy as np

# Init Pygame
pygame.init()

# Set up window
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird - Smile to Flap")

# Load face and smile cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Check if the cascades are loaded properly
if face_cascade.empty():
    print("Error loading face cascade.")
    sys.exit()
if smile_cascade.empty():
    print("Error loading smile cascade.")
    sys.exit()

# Game state
game_active = True
font = pygame.font.SysFont("Arial", 40)
score = 0
scored_pipe = False

# Game variables
clock = pygame.time.Clock()
gravity = 0.4
bird_movement = 0
bird_y = HEIGHT // 2
flap_strength = -8

# Pipe settings
pipe_width = 70
pipe_gap = 175
pipe_x = WIDTH
pipe_height = random.randint(100, 400)
pipe_speed = 3.5

# Camera setup (use the first camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# Set smaller camera resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Smile detection variables
smile_detected = False
last_smile = False
smile_cooldown = 0
SMILE_COOLDOWN_MAX = 15  # Frames to wait before detecting another smile

# Debug mode to show face detection rectangles
DEBUG = True

def detect_smile(frame):
    """Detect smile in the frame with improved parameters."""
    global smile_cooldown
    
    # If on cooldown, decrease counter and return previous smile state
    if smile_cooldown > 0:
        smile_cooldown -= 1
        return last_smile
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram to improve detection in various lighting
    gray = cv2.equalizeHist(gray)
    
    # Detect faces with adjusted parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return False  # No faces detected
    
    # Sort faces by size (largest first) and only use the largest
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    (x, y, w, h) = faces[0]
    
    if DEBUG:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # ROI for smile detection (focus on lower part of the face)
    roi_y_start = y + int(h * 0.5)  # Lower half of face
    roi_height = h - int(h * 0.5)
    face_roi = gray[roi_y_start:y + h, x:x + w]
    
    # Apply additional preprocessing to the ROI
    face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
    
    # Detect smiles with carefully tuned parameters
    smiles = smile_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.7,
        minNeighbors=22,        # Higher value for more strictness
        minSize=(25, 15),      # Minimum smile size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    smile_found = len(smiles) > 0
    
    if smile_found and DEBUG:
        for (sx, sy, sw, sh) in smiles:
            # Draw smile rectangle (adjust coordinates to match the ROI position)
            cv2.rectangle(
                frame,
                (x + sx, roi_y_start + sy),
                (x + sx + sw, roi_y_start + sy + sh),
                (0, 255, 0),
                2
            )
        
        # Add "SMILE DETECTED" text
        cv2.putText(
            frame,
            "SMILE DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    # If smile detected, start cooldown
    if smile_found:
        smile_cooldown = SMILE_COOLDOWN_MAX
    
    return smile_found

# Game loop
running = True

# Previous video frame for smoothing
prev_frame = None

while running:
    screen.fill((135, 206, 250))  # Sky blue background
    
    # Read frame from webcam
    ret, frame = cap.read()
    
    if ret:
        # Flip horizontally for a more natural feeling
        frame = cv2.flip(frame, 1)
        
        # Create a copy of the original frame for display
        display_frame = frame.copy()
        
        # Resize frame to speed up smile detection
        frame_resized = cv2.resize(frame, (320, 240))
        
        # Temporal smoothing to reduce noise
        if prev_frame is not None:
            frame_resized = cv2.addWeighted(frame_resized, 0.7, prev_frame, 0.3, 0)
        prev_frame = frame_resized.copy()
        
        # Detect smile and update bird movement
        current_smile = detect_smile(display_frame)  # Use the display frame for drawing debug info
        
        # Only flap on new smiles (transition from not smiling to smiling)
        if current_smile and not last_smile and game_active:
            bird_movement = flap_strength
        
        # Update smile state
        last_smile = current_smile
        
        # Show webcam feed in separate window with game state indicator
        if not game_active:
            cv2.putText(
                display_frame,
                "Game Over - Smile to Restart",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Add debug mode status indicator
        debug_status = "DEBUG ON" if DEBUG else "DEBUG OFF"
        cv2.putText(
            display_frame,
            debug_status, 
            (display_frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if DEBUG else (128, 128, 128),
            2 if DEBUG else 1
        )
        
        cv2.imshow("Face Cam - Smile to Flap", display_frame)
    
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Manual flap with spacebar (backup control)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird_movement = flap_strength
            # Restart game with R key
            elif event.key == pygame.K_r and not game_active:
                game_active = True
                bird_y = HEIGHT // 2
                bird_movement = 0
                pipe_x = WIDTH
                pipe_height = random.randint(100, 400)
                score = 0
                scored_pipe = False
            # Toggle debug mode with D key
            elif event.key == pygame.K_d:
                DEBUG = not DEBUG
                print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    # Restart game if smiling while game is over
    if not game_active and current_smile:
        game_active = True
        bird_y = HEIGHT // 2
        bird_movement = 0
        pipe_x = WIDTH
        pipe_height = random.randint(100, 400)
        score = 0
        scored_pipe = False
    
    if game_active:
        # Bird physics
        bird_movement += gravity
        bird_y += bird_movement
        
        # Ground/ceiling limits
        if bird_y > HEIGHT - 20:
            bird_y = HEIGHT - 20
            bird_movement = 0
        if bird_y < 20:
            bird_y = 20
            bird_movement = 0
        
        # Bird
        bird_rect = pygame.Rect(100 - 20, bird_y - 20, 40, 40)
        pygame.draw.ellipse(screen, (255, 255, 0), bird_rect)  # Yellow bird
        
        # Pipes
        pipe_x -= pipe_speed
        if pipe_x < -pipe_width:
            pipe_x = WIDTH
            pipe_height = random.randint(100, 400)
            scored_pipe = False  # reset scoring for this pipe
        
        # Pipe rects
        top_pipe = pygame.Rect(pipe_x, 0, pipe_width, pipe_height)
        bottom_pipe = pygame.Rect(pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT)
        
        pygame.draw.rect(screen, (0, 200, 0), top_pipe)
        pygame.draw.rect(screen, (0, 200, 0), bottom_pipe)
        
        # Score when passing pipe
        if pipe_x + pipe_width < 100 and not scored_pipe:
            score += 1
            scored_pipe = True
        
        # Collision
        if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
            game_active = False
        
        # Draw score
        score_surface = font.render(str(score), True, (0, 0, 0))
        screen.blit(score_surface, (WIDTH // 2 - 10, 10))
        
    else:
        # Game over message + restart
        msg = font.render("Game Over!", True, (255, 0, 0))
        restart = font.render("Smile to Restart", True, (0, 0, 0))
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 80))
        screen.blit(restart, (WIDTH // 2 - restart.get_width() // 2, HEIGHT // 2))
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 + 60))
    
    # Display smile status indicator
    if current_smile:
        indicator = font.render("😊", True, (255, 255, 0))
        screen.blit(indicator, (10, 10))
    
    # Add instructions
    instructions = pygame.font.SysFont("Arial", 20).render("Smile to flap!", True, (0, 0, 0))
    # debug_instruction = pygame.font.SysFont("Arial", 20).render("Press D to toggle debug view", True, (0, 0, 0))
    screen.blit(instructions, (10, HEIGHT - 25))
    # screen.blit(debug_instruction, (10, HEIGHT - 25))
    
    pygame.display.update()
    clock.tick(60)  # 60 FPS for smoother gameplay
    
    # Process OpenCV window keys
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        running = False
    elif key == ord('d') or key == ord('D'):  # D key in OpenCV window
        DEBUG = not DEBUG
        print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()