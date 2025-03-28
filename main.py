from deepface import DeepFace
import cv2
import pygame
import random
import time

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Juego de Expresiones")
font = pygame.font.Font(None, 50)

# Emociones posibles
emociones = ['happy', 'sad', 'angry', 'surprise', 'neutral']

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Variables del juego
puntos = 0
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

    # Intentar detectar emoción
    try:
        result = DeepFace.analyze(small_frame, actions=[
                                  'emotion'], enforce_detection=False)
        emocion_detectada = result[0]['dominant_emotion']
    except:
        emocion_detectada = "No face detected"

    # Comprobar si la emoción detectada coincide con la objetivo
    if emocion_detectada == emocion_objetivo:
        puntos += 1
        emocion_objetivo = random.choice(emociones)  # Nueva emoción a imitar
        ultimo_cambio = time.time()  # Reiniciar tiempo

    # Reiniciar emoción objetivo después de tiempo límite
    if time.time() - ultimo_cambio > tiempo_limite:
        emocion_objetivo = random.choice(emociones)
        ultimo_cambio = time.time()

    # Mostrar la emoción detectada en OpenCV
    cv2.putText(frame, f"Emocion: {emocion_detectada}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    # Dibujar en Pygame
    screen.fill((255, 255, 255))
    text = font.render(f"Imita: {emocion_objetivo}", True, (0, 0, 0))
    score_text = font.render(f"Puntos: {puntos}", True, (0, 0, 255))
    screen.blit(text, (150, 150))
    screen.blit(score_text, (220, 250))

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
