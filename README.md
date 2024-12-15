# deteccion-unet

Repositorio correspondiente al proyecto de *Detección de cabezas de mascotas con U-Net y contornos*

# Requisitos

Antes de ejecutar el código, es necesario instalar los requisitos:

```bash
pip install -r requirements.txt
```

Asegúrese de que el archivo *unet.weights.h5* esté presente pues contiene los pesos del modelo
entrenado.

# Ejecución de prueba

Para probar con una imagen de prueba abrir el archivo *head_detector.py* y modificar la ruta de la imagen
en la línea:

```python
detect_from_image("test.jpg", use_contours=True)
```

En caso de querer probar la detección en tiempo real con la webcam, se debe comentar la línea *detect_from_image*
y descomentar *detect_from_camera*:

```python
#detect_from_image("test.jpg", use_contours=True)
detect_from_camera(use_contours=True)
```

# Dataset

Este proyecto descarga automáticamente el dataset *The Oxford-IIIT Pet Dataset* en *images.tar.gz* y
*annotations.tar.gz*.

# Entrenamiento con K-Fold

Para hacer el entrenamiento con K-Fold se requiere ejecutar el archivo *train_unet.py*. Este descarga
el dataset y entrena el modelo.

```bash
python train_unet.py
```

# Entrenamiento con todos los datos

Para entrenar con todos los datos, sin validación, se debe ejecutar el archivo *train_full_unet.py*. Este
descarga el dataset y entrena empleando todos los datos disponibles.

```bash
python train_full_unet.py
```
