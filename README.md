# YOLO Splitted Image Analyzer

Aplicación para Raspberry Pi 4 que captura imágenes desde una cámara CSI mediante **Picamera2** y ejecuta inferencia con un modelo **YOLO exportado a NCNN**.

## Requisitos

- Raspberry Pi 4 con Raspberry Pi OS basado en Debian Bookworm.
- Python 3.11.
- Cámara CSI compatible con `libcamera` (probado con sensor OV5647).
- Modelo NCNN exportado y ubicado en `models/yolo26n_320_ncnn_model/`.

> **Importante:** Picamera2 se instala desde los paquetes del sistema (APT), no desde `pip`. El entorno virtual debe crearse con `--system-site-packages` para que Python pueda importar Picamera2 y sus dependencias del sistema.

## Instalación

### 1. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2
```

### 2. Crear y activar el entorno virtual

Desde la raíz del repositorio:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Instalar dependencias Python

```bash
python -m pip install -r requirements.txt
```

El `requirements.txt` debe mantener versiones compatibles con el entorno probado en la Raspberry Pi. En particular, NumPy debe permanecer en una versión compatible con los módulos del sistema usados por Picamera2; no actualices NumPy a ciegas a la versión más reciente.

### 4. Comprobar las dependencias

```bash
python - <<'PY'
import numpy
import cv2
import picamera2
import ultralytics
import ncnn

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Picamera2: OK")
print("Ultralytics:", ultralytics.__version__)
print("NCNN: OK")
PY
```

Si algún módulo no se encuentra, comprueba que el entorno virtual está activo y que Picamera2 se instaló mediante APT.

## Ejecución

Activa el entorno virtual y ejecuta el programa principal:

```bash
source .venv/bin/activate
python main.py
```

La aplicación utiliza Picamera2 para capturar imágenes desde la cámara CSI y carga el modelo NCNN desde:

```text
models/yolo26n_320_ncnn_model
```

Asegúrate de que esa carpeta del modelo existe antes de iniciar el programa.

## Estructura esperada

```text
yolo-splitted-image-analyzer/
├── .venv/                         # Entorno virtual (no versionar)
├── main.py
├── requirements.txt
└── models/
    └── yolo26n_320_ncnn_model/     # Modelo exportado a NCNN
```

La estructura puede variar según los archivos que tenga el repositorio.

## Solución de problemas

### `ModuleNotFoundError: No module named 'picamera2'`

Crea el entorno con acceso a los paquetes del sistema y comprueba que Picamera2 está instalado:

```bash
sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Si ya existía un entorno virtual creado sin `--system-site-packages`, recréalo con el comando anterior.

### Error de incompatibilidad de NumPy (`numpy.dtype size changed`)

Este error suele indicar que una extensión compilada —por ejemplo, una biblioteca de cámara instalada desde APT— se está usando con una versión de NumPy incompatible. Instala las versiones compatibles fijadas en `requirements.txt` y evita actualizar NumPy sin verificar la compatibilidad.

### Ultralytics indica que falta `ncnn`

Instálalo dentro del entorno virtual:

```bash
python -m pip install ncnn
```

Después, vuelve a ejecutar `python main.py` en un proceso nuevo.

### Error `Illegal instruction` al ejecutar PyTorch

Este proyecto utiliza el modelo exportado a NCNN para la inferencia en la Raspberry Pi. Evita ejecutar el modelo mediante el backend PyTorch si la instalación de PyTorch disponible no es compatible con la CPU de la Raspberry Pi.

## Recomendaciones

- No subas `.venv/` al repositorio.
- Conserva el modelo NCNN en la ruta que espera `main.py`, o actualiza la ruta en el código.
- Cuando el entorno funcione correctamente, registra las versiones exactas de las dependencias (`python -m pip freeze`) para facilitar la reproducción.
- Picamera2 y las bibliotecas vinculadas a `libcamera` dependen de la distribución y de los paquetes del sistema; por eso no basta con instalar únicamente `requirements.txt`.

## Licencia

