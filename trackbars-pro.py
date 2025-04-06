import cv2 as cv
import numpy as np

def nothing(x):
    pass

# --------------------------------------------------
# 1) CONFIGURACIÓN INICIAL
# --------------------------------------------------
ruta_imagen = "lateral-natural-iphone.png"  # Ajusta a tu archivo
img_original = cv.imread(ruta_imagen)
if img_original is None:
    raise FileNotFoundError("No se pudo cargar la imagen en la ruta especificada.")

# Redimensionar (opcional)
scale_percent = 50
width = int(img_original.shape[1] * scale_percent / 100)
height = int(img_original.shape[0] * scale_percent / 100)
img_original = cv.resize(img_original, (width, height), interpolation=cv.INTER_AREA)

# Convertir a escala de grises para ciertos métodos
gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

# --------------------------------------------------
# 2) CREAR VENTANAS
# --------------------------------------------------
cv.namedWindow("Resultado", cv.WINDOW_NORMAL)
cv.namedWindow("Trackbars", cv.WINDOW_NORMAL)

# --------------------------------------------------
# 3) CREAR TRACKBARS (agrupados por nombres/prefijos)
# --------------------------------------------------

# -- Selección de método principal (índice)
# 0: Original, 1:Canny, 2:SobelX, 3:SobelY, 4:Laplacian, 5:Prewitt
# 6:Threshold binario, 7:Adaptative Mean, 8:Adaptative Gaussian, 9:Otsu
# 10:Erode, 11:Dilate, 12:Open, 13:Close
cv.createTrackbar("Metodo", "Trackbars", 0, 13, nothing)

# -- CANNY
cv.createTrackbar("[CANNY] MinT", "Trackbars", 50, 255, nothing)
cv.createTrackbar("[CANNY] MaxT", "Trackbars", 150, 255, nothing)

# -- FILTROS DE BORDES (Sobel / Laplacian / Prewitt) => KernelSize
cv.createTrackbar("[EDGE] Kernel", "Trackbars", 3, 21, nothing)

# -- BINARIZACIÓN GLOBAL
cv.createTrackbar("[BIN] Value", "Trackbars", 127, 255, nothing)

# -- BINARIZACIÓN ADAPTATIVA (Mean/Gaussian)
cv.createTrackbar("[ADAPT] BlockSize", "Trackbars", 11, 51, nothing)
cv.createTrackbar("[ADAPT] C", "Trackbars", 2, 20, nothing)

# -- MORFOLOGÍA
cv.createTrackbar("[MORPH] Kernel", "Trackbars", 3, 21, nothing)
cv.createTrackbar("[MORPH] Iter", "Trackbars", 1, 10, nothing)

# --------------------------------------------------
# 4) BUCLE PRINCIPAL
# --------------------------------------------------
while True:
    # Leer trackbars:
    metodo       = cv.getTrackbarPos("Metodo", "Trackbars")
    canny_min    = cv.getTrackbarPos("[CANNY] MinT", "Trackbars")
    canny_max    = cv.getTrackbarPos("[CANNY] MaxT", "Trackbars")
    edge_kernel  = cv.getTrackbarPos("[EDGE] Kernel", "Trackbars")
    bin_value    = cv.getTrackbarPos("[BIN] Value", "Trackbars")
    adapt_bsize  = cv.getTrackbarPos("[ADAPT] BlockSize", "Trackbars")
    adapt_c      = cv.getTrackbarPos("[ADAPT] C", "Trackbars")
    morph_kernel = cv.getTrackbarPos("[MORPH] Kernel", "Trackbars")
    morph_iter   = cv.getTrackbarPos("[MORPH] Iter", "Trackbars")

    # Ajustar kernel_size (EDGE) para que sea impar y >= 1
    if edge_kernel < 1:
        edge_kernel = 1
    if edge_kernel % 2 == 0:
        edge_kernel += 1

    # Ajustar blockSize para umbral adaptativo (impar y >= 3)
    if adapt_bsize < 3:
        adapt_bsize = 3
    if adapt_bsize % 2 == 0:
        adapt_bsize += 1

    # Ajustar kernel para morfología (>=1 y si conviene, impar)
    if morph_kernel < 1:
        morph_kernel = 1
    # (En morfología no es obligatorio que sea impar, se puede dejar tal cual)
    
    # Por defecto, el resultado es la original
    result = img_original.copy()

    # --------------------------------------------------
    # 5) SELECCIÓN DE MÉTODO
    # --------------------------------------------------
    if metodo == 0:
        # 0: Original (sin cambios)
        pass

    elif metodo == 1:
        # 1: CANNY (usa canny_min, canny_max)
        edges = cv.Canny(gray, canny_min, canny_max)
        result = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    elif metodo == 2:
        # 2: SOBEL X (usa edge_kernel)
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=edge_kernel)
        sobelx = cv.convertScaleAbs(sobelx)
        result = cv.cvtColor(sobelx, cv.COLOR_GRAY2BGR)

    elif metodo == 3:
        # 3: SOBEL Y (usa edge_kernel)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=edge_kernel)
        sobely = cv.convertScaleAbs(sobely)
        result = cv.cvtColor(sobely, cv.COLOR_GRAY2BGR)

    elif metodo == 4:
        # 4: LAPLACIAN (usa edge_kernel)
        lap = cv.Laplacian(gray, cv.CV_64F, ksize=edge_kernel)
        lap = cv.convertScaleAbs(lap)
        result = cv.cvtColor(lap, cv.COLOR_GRAY2BGR)

    elif metodo == 5:
        # 5: PREWITT (usa edge_kernel, pero se hace manualmente)
        #    => Realmente Prewitt no tiene un "kernel_size" variable en la implementación base
        #    => Podríamos ignorar edge_kernel o hacer algo más elaborado.
        #    => De momento, aplicamos Prewitt fijo 3x3
        kernelx = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[ 1,  1,  1],
                            [ 0,  0,  0],
                            [-1, -1, -1]], dtype=np.float32)
        px = cv.filter2D(gray, -1, kernelx)
        py = cv.filter2D(gray, -1, kernely)
        prewitt = cv.addWeighted(px, 0.5, py, 0.5, 0)
        result = cv.cvtColor(prewitt, cv.COLOR_GRAY2BGR)

    elif metodo == 6:
        # 6: UMBRAL BINARIO (usa bin_value)
        _, thresh_bin = cv.threshold(gray, bin_value, 255, cv.THRESH_BINARY)
        result = cv.cvtColor(thresh_bin, cv.COLOR_GRAY2BGR)

    elif metodo == 7:
        # 7: UMBRAL ADAPTATIVO (Mean) (usa adapt_bsize y adapt_c)
        adapt_mean = cv.adaptiveThreshold(
            gray, 255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            adapt_bsize,
            adapt_c
        )
        result = cv.cvtColor(adapt_mean, cv.COLOR_GRAY2BGR)

    elif metodo == 8:
        # 8: UMBRAL ADAPTATIVO (Gaussian)
        adapt_gauss = cv.adaptiveThreshold(
            gray, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            adapt_bsize,
            adapt_c
        )
        result = cv.cvtColor(adapt_gauss, cv.COLOR_GRAY2BGR)

    elif metodo == 9:
        # 9: OTSU
        _, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        result = cv.cvtColor(otsu, cv.COLOR_GRAY2BGR)

    elif metodo in [10, 11, 12, 13]:
        # MORFOLOGÍA: 10:Erode, 11:Dilate, 12:Open, 13:Close
        # Creamos el kernel (elemento estructurante) con morph_kernel x morph_kernel
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_kernel, morph_kernel))
        # Trabajamos sobre gray (o binarizado si quisiéramos)
        morph_input = gray.copy()

        if metodo == 10:
            morph_result = cv.erode(morph_input, kernel, iterations=morph_iter)
        elif metodo == 11:
            morph_result = cv.dilate(morph_input, kernel, iterations=morph_iter)
        elif metodo == 12:
            morph_result = cv.morphologyEx(morph_input, cv.MORPH_OPEN, kernel, iterations=morph_iter)
        else:  # 13
            morph_result = cv.morphologyEx(morph_input, cv.MORPH_CLOSE, kernel, iterations=morph_iter)

        result = cv.cvtColor(morph_result, cv.COLOR_GRAY2BGR)

    # --------------------------------------------------
    # 6) MOSTRAR RESULTADO
    # --------------------------------------------------
    combined = np.hstack((img_original, result))
    cv.imshow("Resultado", combined)

    # Salir con 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
