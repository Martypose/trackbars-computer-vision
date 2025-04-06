import cv2 as cv
import numpy as np

# 1) Cargar imagen y convertir a HSV
ruta_imagen = "lateral-natural-iphone.png"
img = cv.imread(ruta_imagen)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 2) Crear máscara con tu rango
lower_hsv = np.array([10, 84, 128])
upper_hsv = np.array([25, 255, 255])
mask = cv.inRange(hsv, lower_hsv, upper_hsv)

# 3) Apertura -> Cierre con distintos kernels
kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))    # Más pequeño
kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (21, 21)) # Más grande

# Apertura
mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open, iterations=2)
# Cierre
mask_refined = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close, iterations=1)

# 4) Contornos en la máscara refinada
contours, _ = cv.findContours(mask_refined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

output_image = img.copy()
image_height, image_width = output_image.shape[:2]

found_contour = None
max_area = 0

for cnt in contours:
    # boundingRect para filtrar contornos >= 90% ancho y centrados
    x, y, w, h = cv.boundingRect(cnt)
    if w >= 0.9 * image_width:
        center_x = x + w // 2
        center_img_x = image_width // 2
        if abs(center_x - center_img_x) <= 0.1 * image_width:
            area = cv.contourArea(cnt)
            if area > max_area:
                max_area = area
                found_contour = cnt

if found_contour is not None:
    # Dibuja contorno y minAreaRect
    cv.drawContours(output_image, [found_contour], 0, (255, 0, 0), 2)
    min_rect = cv.minAreaRect(found_contour)
    box_points = cv.boxPoints(min_rect)
    box_points = np.intp(box_points)
    cv.drawContours(output_image, [box_points], 0, (0, 255, 0), 3)

    # Calcular área y mostrar
    area = cv.contourArea(found_contour)
    x_text, y_text = box_points[0]
    cv.putText(output_image, f"Area={int(area)}", (x_text, y_text - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
else:
    print("No se halló contorno grande y centrado.")

# 5) Mostrar resultado
cv.namedWindow("Mask Refinada", cv.WINDOW_NORMAL)
cv.namedWindow("Resultado", cv.WINDOW_NORMAL)
cv.imshow("Mask Refinada", mask_refined)
cv.imshow("Resultado", output_image)
cv.waitKey(0)
cv.destroyAllWindows()
