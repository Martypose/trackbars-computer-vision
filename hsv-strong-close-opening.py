import cv2 as cv
import numpy as np

def nothing(x):
    pass

# ----------------------------------------------------------------
# 1) Cargar la imagen
# ----------------------------------------------------------------
ruta_imagen = "lateral-natural-iphone-regular.png"  # Ajusta la ruta a tu archivo
img = cv.imread(ruta_imagen)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen en la ruta especificada.")

# (Opcional) Redimensionar
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

# Convertir a HSV
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# ----------------------------------------------------------------
# 2) Crear ventanas y trackbars
# ----------------------------------------------------------------
cv.namedWindow("Trackbars", cv.WINDOW_NORMAL)
cv.namedWindow("Vista", cv.WINDOW_NORMAL)

# Inicializa trackbars (ejemplo: [H=10..25, S=80..255, V=80..255])
cv.createTrackbar("H Min", "Trackbars",  10, 179, nothing)
cv.createTrackbar("H Max", "Trackbars",  25, 179, nothing)
cv.createTrackbar("S Min", "Trackbars",  80, 255, nothing)
cv.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv.createTrackbar("V Min", "Trackbars",  80, 255, nothing)
cv.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# Trackbars para morfología
# (aplicaremos closing + opening con un único kernel e iteraciones)
cv.createTrackbar("Kernel",     "Trackbars", 12, 21, nothing)
cv.createTrackbar("Iterations", "Trackbars",  2, 10, nothing)

while True:
    # Leer posiciones de trackbars
    h_min = cv.getTrackbarPos("H Min", "Trackbars")
    h_max = cv.getTrackbarPos("H Max", "Trackbars")
    s_min = cv.getTrackbarPos("S Min", "Trackbars")
    s_max = cv.getTrackbarPos("S Max", "Trackbars")
    v_min = cv.getTrackbarPos("V Min", "Trackbars")
    v_max = cv.getTrackbarPos("V Max", "Trackbars")

    ksize = cv.getTrackbarPos("Kernel", "Trackbars")
    iterations = cv.getTrackbarPos("Iterations", "Trackbars")
    if ksize < 1:
        ksize = 1

    # Generar máscara por color
    lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask_color = cv.inRange(hsv_img, lower_hsv, upper_hsv)

    # Crear kernel para la operación morfológica
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))

    # 1) Cierre (closing): rellena agujeros
    mask_closed = cv.morphologyEx(mask_color, cv.MORPH_CLOSE, kernel, iterations=iterations)
    # 2) Apertura (opening): elimina ruidos pequeños
    mask_refined = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel, iterations=iterations)

    # Aplicar la máscara refinada a la imagen original
    result = cv.bitwise_and(img, img, mask=mask_refined)

    # Convertir las máscaras a BGR para visualización
    mask_color_bgr = cv.cvtColor(mask_color, cv.COLOR_GRAY2BGR)
    mask_refined_bgr = cv.cvtColor(mask_refined, cv.COLOR_GRAY2BGR)

    # Mostrar en una cuadrícula:
    # 1) Original
    # 2) Mask Color
    # 3) Mask Refinada
    # 4) Resultado final
    top_row = np.hstack((img, mask_color_bgr))
    bottom_row = np.hstack((mask_refined_bgr, result))
    combined = np.vstack((top_row, bottom_row))

    cv.imshow("Vista", combined)

    # Salir con 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
