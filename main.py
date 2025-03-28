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
font = pygame.font.Font(None, 30)

# Emociones posibles
emociones = ['happy', 'sad', 'angry', 'surprise', 'neutral']

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Registrar las caras de los jugadores


def registrar_cara(jugador):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mostrar instrucciones
        screen.fill((255, 255, 255))
        text = font.render(
            f"Jugador {jugador}: Coloca tu cara frente a la cámara y presiona la tecla C", True, (0, 0, 0))
        screen.blit(text, (10, 0))

        # Mostrar feed de la cámara
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        # Ajustar posición según sea necesario
        frame_surface = pygame.transform.scale(frame_surface, (400, 300))
        screen.blit(frame_surface, (200, 80))

        pygame.display.flip()

        # Capturar cara al presionar 'c'
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                face_representation = DeepFace.represent(
                    frame, model_name='Facenet', enforce_detection=False)
                return face_representation


# Registrar caras de los jugadores
jugador_1_face = registrar_cara(1)
jugador_2_face = registrar_cara(2)

# Variables del juego
puntos = [0, 0]  # Puntos para cada jugador
tiempo_limite = 5  # Segundos para cada emoción
ultimo_cambio = time.time()
emocion_objetivo = random.choice(emociones)

running = True
while running:
    # Capturar imagen de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir tamaño de la imagen para mejorar rendimiento
    small_frame = cv2.resize(frame, (300, 300))

    # Intentar detectar emoción y jugador
    try:
        result = DeepFace.analyze(small_frame, actions=[
                                  'emotion'], enforce_detection=False)
        emocion_detectada = result[0]['dominant_emotion']

        # Identificar jugador
        face_representation = DeepFace.represent(
            small_frame, model_name='Facenet', enforce_detection=False)
        dist_1 = DeepFace.verify(face_representation, jugador_1_face,
                                 model_name='Facenet', enforce_detection=False)['distance']
        dist_2 = DeepFace.verify(face_representation, jugador_2_face,
                                 model_name='Facenet', enforce_detection=False)['distance']
        jugador_actual = 0 if dist_1 < dist_2 else 1
    except:
        emocion_detectada = "No face detected"
        jugador_actual = None

    # Comprobar si la emoción detectada coincide con la objetivo
    if jugador_actual is not None and emocion_detectada == emocion_objetivo:
        puntos[jugador_actual] += 1
        emocion_objetivo = random.choice(emociones)  # Nueva emoción a imitar
        ultimo_cambio = time.time()  # Reiniciar tiempo

    # Reiniciar emoción objetivo después de tiempo límite
    if time.time() - ultimo_cambio > tiempo_limite:
        emocion_objetivo = random.choice(emociones)
        ultimo_cambio = time.time()

    # Convertir el frame de OpenCV (BGR) a RGB para Pygame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    # Ajustar posición según sea necesario
    frame_surface = pygame.transform.scale(frame_surface, (400, 300))

    # Dibujar en Pygame
    screen.fill((255, 255, 255))
    screen.blit(frame_surface, (200, 80))
    text = font.render(f"Imita: {emocion_objetivo}", True, (0, 0, 0))
    score_text = font.render(
        f"P1: {puntos[0]}  P2: {puntos[1]}", True, (0, 0, 255))
    turno_text = font.render(
        f"Turno: Jugador {jugador_actual + 1}" if jugador_actual is not None else "Detectando...", True, (0, 128, 0))
    screen.blit(text, (10, 100))
    screen.blit(score_text, (10, 200))
    screen.blit(turno_text, (10, 300))

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
