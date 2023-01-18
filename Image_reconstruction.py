"""
IMAGE RECONSTRUCTION

@authors: 
    - Juan Jose Potes
    - Julie Ibarra
    - Cristian Jimenez
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import copy

# Se leen todas las imagenes y se guardan en una lista
imagenes = [cv2.imread(file) for file in glob.glob("pedazos_aleatorio/*.png")]

# Vector de booleanos para identificar las imagenes que ya fueron ubicadas y asi no volver a revisarlas
unidas = np.full(len(imagenes), False)

# Se guarda la imagen inicial de referencia
imagen_ini = cv2.imread("pedazos_aleatorio/img1_1.png")

# Matriz que guardara las y ordenara posiciones de las imagenes, el -2 indicara que no se ha encontrado la imagen
matriz = np.full([31,31], -2)

# En la primera posicion se pone un -1 ya que es la imagen de referencia y se toma como una imagen aparte de la lista
matriz[0][0] = -1

# Matriz que guardara los datos de la imagen reconstruida, se inicializa con ceros
img_final = np.zeros([1185,1920,3], dtype=np.uint8)

# Variables que indicaran la posicion en la que se intersectan las imagenes horizontal y verticalmente
inter_i = 0
inter_j = 0

# Funcion para pasar de bgr a rgb
def bgr_rgb(img):
    img2 = copy.deepcopy(img)
    # Se recorre la imagen y se intercambian el canal rojo y el azul
    for i in range (0, len(img)):
        for j in range (0, len(img[0])):
            img2[i][j][0] = img[i][j][2]
            img2[i][j][1] = img[i][j][1]
            img2[i][j][2] = img[i][j][0]
    return img2

# Funcion para mostrar imagen mediante un plot
def plot(img):
    img_rgb = bgr_rgb(img)
    plt.imshow(img_rgb)
    plt.show()

# Funcion que realiza la correlacion horizontalmente (Compara los colores de los pixeles)
def correlacion_comp_horiz(img_base, img):
    global inter_j
    match = False
    # Si no se ha encontrado la interseccion, va recorriendo la imagen de base con la otra imagen
    if(inter_j == 0):
        # Se recorre la imagen de base horizontalmente
        for j in range (50, len(img_base[0])):
            contfallas = 0
            # Se recorre cada pixel de las 2 primeras columnas de la otra imagen
            for k in range (0, len(img)):
                for n in range (0, 2):
                    if(j+n < len(img_base[0])):
                        # Si el color no coincide en algun pixel, se aumenta en 1 el contador de fallas
                        if(img_base[k][j+n][1] != img[k][n][1] or img_base[k][j+n][2] != img[k][n][2] or img_base[k][j+n][0] != img[k][n][0]):
                            contfallas += 1
                            break
            # Si el contador de fallas es 0 significa que todos los pixeles coinciden por lo que match se hace verdadero
            if(contfallas == 0):
                match = True
                inter_j = j
                break
    # Si ya se tiene la posicion de interseccion, se empieza a comparar desde ahi
    else:
        j = inter_j
        contfallas = 0
        # Se recorre cada pixel de las 2 primeras columnas de la otra imagen
        for k in range (0, len(img)):
            for n in range (0, 2):
                if(j+n < len(img_base[0])):
                    # Si el color no coincide en algun pixel, se aumenta en 1 el contador de fallas
                    if(img_base[k][j+n][1] != img[k][n][1] or img_base[k][j+n][2] != img[k][n][2] or img_base[k][j+n][0] != img[k][n][0]):
                        contfallas += 1
                        break
        # Si el contador de fallas es 0 significa que todos los pixeles coinciden por lo que match se hace verdadero
        if(contfallas == 0):
            match = True
    return match

# Funcion que realiza la correlacion verticalmente (Compara los colores de los pixeles)
def correlacion_comp_vert(img_base, img):
    global inter_i
    match = False
    # Si no se ha encontrado la interseccion, va recorriendo la imagen de base con la otra imagen
    if(inter_i == 0):
        # Se recorre la imagen de base verticalmente
        for i in range (30, len(img_base)):
            contfallas = 0
            # Se recorre cada pixel de las 2 primeras filas de la otra imagen
            for k in range (0, 2):
                for n in range (0, len(img[0])):
                    if(i+k < len(img_base)):
                        # Si el color no coincide en algun pixel, se aumenta en 1 el contador de fallas
                        if(img_base[i+k][n][1] != img[k][n][1] or img_base[i+k][n][2] != img[k][n][2] or img_base[i+k][n][0] != img[k][n][0]):
                            contfallas += 1
                            break
            # Si el contador de fallas es 0 significa que todos los pixeles coinciden por lo que match se hace verdadero
            if(contfallas == 0):
                match = True
                inter_i = i
                break
    # Si ya se tiene la posicion de interseccion, se empieza a comparar desde ahi
    else:
        i = inter_i
        contfallas = 0
        for k in range (0, 2):
            for n in range (0, len(img[0])):
                if(i+k < len(img_base)):
                    # Si el color no coincide en algun pixel, se aumenta en 1 el contador de fallas
                    if(img_base[i+k][n][1] != img[k][n][1] or img_base[i+k][n][2] != img[k][n][2] or img_base[i+k][n][0] != img[k][n][0]):
                        contfallas += 1
                        break
        # Si el contador de fallas es 0 significa que todos los pixeles coinciden por lo que match se hace verdadero
        if(contfallas == 0):
            match = True
    return match

# Funcion que reconstruye la imagen basandose en las posiciones guardadas en la matriz
def reconstruir():
    # Primero se pone la imagen incial en la esquina superior izquierda
    for k in range (0, len(imagen_ini)):
        for n in range (0, len(imagen_ini[0])):
            img_final[k][n][:] = imagen_ini[k][n][:]
    
    # Se recorre la matriz, dependiendo de las posiciones de esta y de las intersecciones, se sabe en que pixel de la imagen final empieza cada pedazo
    for i in range(0,len(matriz)):
        for j in range(0,len(matriz[0])):
            if(matriz[i][j]>=0):
                img = imagenes[matriz[i][j]]
                # Se recorre cada pedazo
                for k in range(0,len(img)):
                    for n in range(0,len(img[0])):
                        # Se pasan los datos de cada pedazo a su posicion respectiva en la imagen final
                        if(img_final[(i*inter_i)+k][(j*inter_j)+n][0] == 0 and img_final[(i*inter_i)+k][(j*inter_j)+n][1] == 0 and img_final[(i*inter_i)+k][(j*inter_j)+n][2] == 0):
                            img_final[(i*inter_i)+k][(j*inter_j)+n][0] = img[k][n][0]
                            img_final[(i*inter_i)+k][(j*inter_j)+n][1] = img[k][n][1]
                            img_final[(i*inter_i)+k][(j*inter_j)+n][2] = img[k][n][2]


# plot(imagen_ini)

# Ciclos que halla las imagenes de la derecha y de abajo de la primera imagen
# Se halla la imagen de la derecha y la coordenada de interseccion de las columnas "inter_j"
for i in range(0,len(imagenes)):
    if(unidas[i] == False):
        if(correlacion_comp_horiz(imagen_ini, imagenes[i]) == True and inter_j >= int(len(imagen_ini[0])/2)):
            print("[ 0 ][ 1 ] Encontrada derecha = img[",i,"]")
            unidas[i]=True
            matriz[0][1] = i
            break

# Se halla la imagen de abajo y la coordenada de interseccion de las filas "inter_i"
for i in range(0,len(imagenes)):
    if(unidas[i] == False):
        if(correlacion_comp_vert(imagen_ini, imagenes[i]) == True and inter_i > 0 and inter_i < 74):
            print("[ 1 ][ 0 ] Encontrada abajo = img[",i,"]")
            unidas[i]=True
            matriz[1][0] = i
            break


# Ciclo que halla las demas imagenes y guarda sus las posiciones respectivas en una matriz
for i in range(0, len(matriz)):
    for j in range(0, len(matriz[0])):
        # Se va recorriendo las posiciones de la matriz en los que ya se tiene imagen
        # Se busca la imagen de la derecha solo si no esta posicionado en el borde derecho
        if(j+1 < len(matriz[0])):
            # Si en esa posicion ya se encontro una imagen y en la de la derecha no se ha encontrado
            if(matriz[i][j] != -2 and matriz[i][j+1] == -2):
                # Se recorre la lista de las imagenes
                for z in range(0,len(imagenes)):
                    # Se revisan solo las imagenes que no han sido ubicadas
                    if(unidas[z] == False):
                        if(correlacion_comp_horiz(imagenes[matriz[i][j]], imagenes[z]) == True):
                            print("[",i,"][",j+1,"] Encontrada = img[",z,"]")
                            unidas[z]=True
                            matriz[i][j+1] = z
                            break
        # Se busca la imagen de abajo solo si no esta posicionado en el borde inferior
        if(i+1 < len(matriz)):
            # Si en esa posicion ya se encontro una imagen y en la de abajo no se ha encontrado
            if(matriz[i][j] != -2 and matriz[i+1][j] == -2):
                # Se recorre la lista de las imagenes
                for z in range(0,len(imagenes)):
                    # Se revisan solo las imagenes que no han sido ubicadas
                    if(unidas[z] == False):
                        if(correlacion_comp_vert(imagenes[matriz[i][j]], imagenes[z]) == True):
                            print("[",i+1,"][",j,"] Encontrada = img[",z,"]")
                            unidas[z]=True
                            matriz[i+1][j] = z
                            break

# Se muestran las posiciones de interseccion horizontal y vertical
print("\ninter_j = ",inter_j)
print("inter_i = ",inter_i)

# Se revisa si falto alguna posicion por imagen
enc = True
for i in range(0, len(matriz)):
    for j in range(0, len(matriz[0])):
        if(matriz[i][j] == -2):
            enc = False
            print("\nImagen de posicion[",i,"][",j,"] NO ENCONTRADA")
            
if(enc == True):
    print("No hay imagenes faltantes")

print("\nReconstruyendo...\n")
# Se recostruye y se muestra la imagen final
reconstruir()
plot(img_final)

cv2.imwrite("Imagen reconstruida.png", img_final)
# cv2.imshow("Imagen reconstruida",img_final)
# cv2.waitKey()
# cv2.destroyAllWindows()