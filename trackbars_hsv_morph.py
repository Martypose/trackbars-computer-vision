import cv2 as cv
import numpy as np

def nothing(x):
    pass

# ----------------------------------------------------------------
# 1) Cargar la imagen
# ----------------------------------------------------------------
ruta_imagen = "lateral-natural-iphone-regular.png"  # Ajusta si es necesario
img = cv.imread(ruta_imagen)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen en la ruta especificada.")

# Redimensionar (opcional)
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

# Convertir a HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# ----------------------------------------------------------------
# 2) Crear ventanas y trackbars
# ----------------------------------------------------------------
cv.namedWindow("Trackbars", cv.WINDOW_NORMAL)
cv.namedWindow("Vista", cv.WINDOW_NORMAL)

# Asignar los valores iniciales tal como en lower_hsv y upper_hsv:
# [10, 80, 80] - [25, 255, 255]
cv.createTrackbar("H Min", "Trackbars", 10,   179, nothing)
cv.createTrackbar("H Max", "Trackbars", 25,   179, nothing)
cv.createTrackbar("S Min", "Trackbars", 80,   255, nothing)
cv.createTrackbar("S Max", "Trackbars", 255,  255, nothing)
cv.createTrackbar("V Min", "Trackbars", 80,   255, nothing)
cv.createTrackbar("V Max", "Trackbars", 255,  255, nothing)

# Trackbar para tipo de morfología: 0=Sin, 1=Opening, 2=Closing, 3=Erode, 4=Dilate
cv.createTrackbar("Morph", "Trackbars", 0, 4, nothing)

# Kernel e iteraciones
cv.createTrackbar("Kernel",     "Trackbars", 3,  21, nothing)
cv.createTrackbar("Iterations", "Trackbars", 1,  10, nothing)

while True:
    # Leer trackbars HSV
    h_min = cv.getTrackbarPos("H Min", "Trackbars")
    h_max = cv.getTrackbarPos("H Max", "Trackbars")
    s_min = cv.getTrackbarPos("S Min", "Trackbars")
    s_max = cv.getTrackbarPos("S Max", "Trackbars")
    v_min = cv.getTrackbarPos("V Min", "Trackbars")
    v_max = cv.getTrackbarPos("V Max", "Trackbars")

    # Leer trackbars morfología
    morph_type = cv.getTrackbarPos("Morph", "Trackbars")
    ksize      = cv.getTrackbarPos("Kernel", "Trackbars")
    iterations = cv.getTrackbarPos("Iterations", "Trackbars")

    # Asegurar kernel >=1
    if ksize < 1:
        ksize = 1

    # Rango para la máscara
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Generar máscara
    mask = cv.inRange(img_hsv, lower_hsv, upper_hsv)

    # Crear kernel de morfología
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))

    # Aplicar la operación morfológica según 'Morph'
    if morph_type == 1:  # Apertura
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=iterations)
    elif morph_type == 2:  # Cierre
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=iterations)
    elif morph_type == 3:  # Erosión
        mask = cv.erode(mask, kernel, iterations=iterations)
    elif morph_type == 4:  # Dilatación
        mask = cv.dilate(mask, kernel, iterations=iterations)
    # Si morph_type==0, no se hace morfología

    # Aplicar máscara a la imagen original
    result = cv.bitwise_and(img, img, mask=mask)

    # Mostrar: Original + Máscara + Resultado
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    combined = np.hstack((img, mask_bgr, result))
    cv.imshow("Vista", combined)

    # Salir con 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
