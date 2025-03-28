from deepface import DeepFace
import cv2
import pygame
import random
import time
import numpy as np

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Juego de Expresiones Multijugador")
font = pygame.font.Font(None, 50)

# Emociones posibles
emociones = ['happy', 'sad', 'angry', 'surprise', 'neutral']

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Variables del juego
puntos = [0, 0]  # Puntos para cada jugador
tiempo_limite = 5  # Segundos para cada emoción
ultimo_cambio = time.time()
emocion_objetivo = random.choice(emociones)
jugador_actual = 0  # 0 para jugador 1, 1 para jugador 2

running = True
while running:
    # Capturar imagen de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir tamaño de la imagen para mejorar rendimiento
    small_frame = cv2.resize(frame, (300, 300))

    # Intentar detectar emoción
    try:
        result = DeepFace.analyze(small_frame, actions=[
                                  'emotion'], enforce_detection=False)
        emocion_detectada = result[0]['dominant_emotion']
    except:
        emocion_detectada = "No face detected"

    # Comprobar si la emoción detectada coincide con la objetivo
    if emocion_detectada == emocion_objetivo:
        puntos[jugador_actual] += 1
        emocion_objetivo = random.choice(emociones)  # Nueva emoción a imitar
        ultimo_cambio = time.time()  # Reiniciar tiempo
        # Cambiar de turno
        jugador_actual = (jugador_actual + 1) % 2

    # Reiniciar emoción objetivo después de tiempo límite
    if time.time() - ultimo_cambio > tiempo_limite:
        emocion_objetivo = random.choice(emociones)
        ultimo_cambio = time.time()
        # Cambiar de turno
        jugador_actual = (jugador_actual + 1) % 2

    # Convertir el frame de OpenCV (BGR) a RGB para Pygame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)

    # Dibujar en Pygame
    screen.fill((255, 255, 255))
    screen.blit(frame_surface, (0, 0))
    text = font.render(f"Imita: {emocion_objetivo}", True, (0, 0, 0))
    score_text = font.render(
        f"P1: {puntos[0]}  P2: {puntos[1]}", True, (0, 0, 255))
    turno_text = font.render(
        f"Turno: Jugador {jugador_actual + 1}", True, (0, 128, 0))
    screen.blit(text, (500, 100))
    screen.blit(score_text, (500, 200))
    screen.blit(turno_text, (500, 300))

    # Eventos de Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar Pygame y OpenCV
cap.release()
cv2.destroyAllWindows()
pygame.quit()
