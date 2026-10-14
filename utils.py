import cv2
import numpy as np
import time
def split_image(image, parts=2):
    """
    Divide una imagen en 2, 3 o 4 partes.

    El orden de las imágenes devueltas es siempre de izquierda a derecha
    y, cuando hay varias filas, de arriba a abajo.

    Particiones:

    parts=2:
        +---------+---------+
        |    0    |    1    |
        +---------+---------+

        0 -> mitad izquierda
        1 -> mitad derecha


    parts=3:
        +---------+---------+
        |         |    1    |
        |         |         |
        +    0    +---------+
        |         |    2    |
        |         |         |
        +---------+---------+

        0 -> mitad izquierda completa
        1 -> mitad superior derecha
        2 -> mitad inferior derecha


    parts=4:
        +---------+---------+
        |    0    |    1    |
        |         |         |
        +---------+---------+
        |    2    |    3    |
        |         |         |
        +---------+---------+

        0 -> superior izquierda
        1 -> superior derecha
        2 -> inferior izquierda
        3 -> inferior derecha

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada en formato NumPy. Se espera que tenga al menos
        dos dimensiones (alto y ancho).

    parts : int, default=2
        Número de particiones en las que dividir la imagen.
        Valores permitidos: 2, 3 o 4.

    Returns
    -------
    list[np.ndarray]
        Lista de imágenes resultantes, en el orden indicado anteriormente.

    Raises
    ------
    ValueError
        Si `parts` no es 2, 3 o 4.
    """
    
    height, width = image.shape[:2]

    if parts == 2:
        mid_x = width // 2

        return [
            image[:, :mid_x],
            image[:, mid_x:]
        ]

    elif parts == 3:
        mid_x = width // 2
        mid_y = height // 2

        return [
            image[:, :mid_x],          # izquierda completa
            image[:mid_y, mid_x:],     # derecha arriba
            image[mid_y:, mid_x:]      # derecha abajo
        ]

    elif parts == 4:
        mid_x = width // 2
        mid_y = height // 2

        return [
            image[:mid_y, :mid_x],
            image[:mid_y, mid_x:],
            image[mid_y:, :mid_x],
            image[mid_y:, mid_x:]
        ]

    else:
        raise ValueError("parts debe ser 2, 3 o 4")
    

def join_images(images, parts=2):
    if parts == 2:
        if len(images) != 2:
            raise ValueError("Para parts=2 se necesitan 2 imágenes")

        return np.concatenate(images, axis=1)

    elif parts == 3:
        if len(images) != 3:
            raise ValueError("Para parts=3 se necesitan 3 imágenes")

        left = images[0]
        right_top = images[1]
        right_bottom = images[2]

        right = np.concatenate(
            [right_top, right_bottom],
            axis=0
        )

        return np.concatenate(
            [left, right],
            axis=1
        )

    elif parts == 4:
        if len(images) != 4:
            raise ValueError("Para parts=4 se necesitan 4 imágenes")

        top = np.concatenate(
            [images[0], images[1]],
            axis=1
        )

        bottom = np.concatenate(
            [images[2], images[3]],
            axis=1
        )

        return np.concatenate(
            [top, bottom],
            axis=0
        )

    else:
        raise ValueError("parts debe ser 2, 3 o 4")
    
    
    
def draw_fps(image, position=(10, 30)):
    """
    Dibuja el FPS actual sobre la imagen.

    Calcula los FPS acumulando frames y actualizando el valor cada segundo.

    Parameters
    ----------
    image : np.ndarray
        Imagen sobre la que dibujar el texto.
    position : tuple[int, int], default=(10, 30)
        Esquina inferior izquierda del texto.

    Returns
    -------
    np.ndarray
        La misma imagen con el FPS dibujado.
    """
    if not hasattr(draw_fps, "start_time"):
        draw_fps.start_time = time.perf_counter()
        draw_fps.frames = 0
        draw_fps.fps = 0.0

    draw_fps.frames += 1

    elapsed = time.perf_counter() - draw_fps.start_time

    if elapsed >= 1.0:
        draw_fps.fps = draw_fps.frames / elapsed
        draw_fps.start_time = time.perf_counter()
        draw_fps.frames = 0

    cv2.putText(
        image,
        f"FPS: {draw_fps.fps:.2f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return image

def get_valid_objects_in_zone(parts: int) -> dict[int, list[int]]:
    configurations = {
        2: {
            0: [],
            1: [0, 1],  # botella y lata
        },
        3: {
            0: [],
            1: [0],     # botella
            2: [1],     # lata
        },
        4: {
            0: [],
            1: [],
            2: [],
            3: [],
        }
    }

    if parts not in configurations:
        raise ValueError(f"PARTS debe ser 2, 3 o 4, recibido: {parts}")

    return configurations[parts]