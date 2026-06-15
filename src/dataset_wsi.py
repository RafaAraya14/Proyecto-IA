from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



"""
se trata de importar, las libreracias pues son oppcionales, porque pueden o no estar instaladas en el entorno, 
en caso de que se usen y esten en None solo tira un error. 
"""
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

try:
    import openslide
except ImportError:
    openslide = None

try:
    import tifffile
except ImportError:  
    tifffile = None


# ----------Funciones para los data augmentations----------

def construir_transformaciones_entrenamiento(tamano_tile: int = 256):
    """ esta funcion crea las transformaciones de augmentation para mejroar el entrenamiento del modelo 
    lo hace mediante un pipeline"""

    if A is None:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=90, border_mode=0, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def construir_transformaciones_evaluacion(tamano_tile: int = 256):
    """
    funcion para crear el tnensor de evaluacion que tiene solo normalizacion
    """
    if A is None:
        return None
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

#--------Funciones para leer los tiles de las imagenes----------

def leer_region_del_slide(ruta_slide: str | Path, ubicacion: tuple[int, int], nivel: int, tamano_tile: int) -> np.ndarray:
    ruta_slide = Path(ruta_slide) # tiene la ruta de del slide

    #ve si openslide esta disponible es mas eficiente para los archivos grande
    if openslide is not None:
        slide = openslide.OpenSlide(str(ruta_slide))
        try:
            #leet la region del slide
            imagen = slide.read_region(ubicacion, nivel, (tamano_tile, tamano_tile)).convert("RGB")
            return np.asarray(imagen)
        finally:
            slide.close()

    # ve si tiffle esta disponible 
    if tifffile is not None:
        with tifffile.TiffFile(str(ruta_slide)) as tif:
            serie = tif.series[0]
            #asegura que solo se puedan mandar niveles que existan 
            indice_nivel = min(nivel, len(serie.levels) - 1)
            arreglo_nivel = serie.levels[indice_nivel].asarray()
            x, y = ubicacion
            #recortar las iamgenes 
            recorte = arreglo_nivel[y : y + tamano_tile, x : x + tamano_tile]
            if recorte.ndim == 2:#si tiene un solo canal lo convierte en rgb
                return np.asarray(recorte)
            return np.asarray(recorte[..., :3])

    #en caso de que ninguna de las librerias este disponible
    raise ImportError("Es necesario tener openslide o tifffile para leer las regiones del tile")


def obtener_dimensiones_del_slide(ruta_slide: str | Path, nivel: int) -> tuple[int, int]:

    """
    fucion para obtener las dimensiones del slide 
    """
    ruta_slide = Path(ruta_slide)

    if openslide is not None: #usa openslide si esta disponible
        slide = openslide.OpenSlide(str(ruta_slide))
        try:
            ancho, alto = slide.level_dimensions[nivel]
            return int(ancho), int(alto)
        finally:
            slide.close()

    if tifffile is not None: #usa tiffile si esta diponible 
        with tifffile.TiffFile(str(ruta_slide)) as tif:
            serie = tif.series[0]
            indice_nivel = min(nivel, len(serie.levels) - 1)
            forma = serie.levels[indice_nivel].shape
            return int(forma[1]), int(forma[0])

    raise ImportError("Se necesita openslide o tifffile para inspeccionar la resolución del WSI.")

#----------funciones para filtrar tiles ----------
def calcular_proporcion_tejido(tile: np.ndarray, umbral_blanco: int = 245) -> float:
  
    """
    funcion que calcula cuanto tejido hay en un slide, filtra los tiles que no tuienen suficiente tejido mediante el brillo de la imagen. 
    """
    if tile.ndim == 2: #escala de grises
        brillo = tile
    else:
        brillo = tile.mean(axis=2) #calcula el brillo general
    proporcion_blanco = (brillo >= umbral_blanco).mean()
    return float(1.0 - proporcion_blanco)


def resolver_ruta_mascara(
    identificador_imagen: str,
    directorio_mascaras: str | Path,
    sufijo_mascara: str = "_mask.tiff",
) -> Path:
    
    """"
    funcion para obtener la ruta de la mascara con el id de la imagen
    """

    directorio_mascaras = Path(directorio_mascaras)
    return directorio_mascaras / f"{identificador_imagen}{sufijo_mascara}"


# ---------------------------------------------------------------------------
# Construcción de un manifiesto de tiles a partir de un DataFrame de slides
# ---------------------------------------------------------------------------

#----------Construccion del manifiesto----------
def construir_manifiesto_tiles(
    df_slides: pd.DataFrame,
    columna_slide: str = "ruta_slide",
    columna_etiqueta: str = "isup_grade",
    columna_paciente: str = "patient_id",
    columna_mascara: str | None = None,
    directorio_mascaras: str | Path | None = None,
    sufijo_mascara: str = "_mask.tiff",
    tamano_tile: int = 256,
    salto: int | None = None,
    nivel: int = 0,
    proporcion_minima_tejido: float = 0.3,
    umbral_blanco: int = 245,
    maximo_tiles_por_slide: int | None = None,
) -> pd.DataFrame:
    
    """
    funcion para el dataframe manifiesto con las coords de los tiles que estan validos
    """
    salto = salto or tamano_tile 
    columnas_requeridas = {columna_slide, columna_etiqueta} 
    faltantes = columnas_requeridas.difference(df_slides.columns)
    if faltantes: #si faltam columnas
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")

    registros: list[dict[str, object]] = [] #guardar los tiles validos

    for _, fila in df_slides.iterrows(): #recorrer cada tile del dataframe de slides
        ruta_slide = fila[columna_slide]
        etiqueta = fila[columna_etiqueta]
        paciente = fila[columna_paciente] if columna_paciente in df_slides.columns else None
        mascara = None

        if columna_mascara is not None and columna_mascara in df_slides.columns:#si la mascara esta en el dataframe se usa esa ruta
            mascara = fila[columna_mascara]
        elif directorio_mascaras is not None and "image_id" in df_slides.columns:#si no se encuentra la mascara 
            mascara = resolver_ruta_mascara(str(fila["image_id"]), directorio_mascaras, sufijo_mascara=sufijo_mascara)

        ancho, alto = obtener_dimensiones_del_slide(ruta_slide, nivel=nivel) #obtener las dimensiones del slide 
        aceptados = 0

        for y in range(0, max(alto - tamano_tile + 1, 1), salto):
            #ciclo for para recorrer el slide y tener los tiles calidos
            for x in range(0, max(ancho - tamano_tile + 1, 1), salto):
                #lee el tile
                tile = leer_region_del_slide(ruta_slide, (x, y), nivel, tamano_tile)
                #calcula la porporcion de tejido
                proporcion_tejido = calcular_proporcion_tejido(tile, umbral_blanco=umbral_blanco)
                #si es menor a la mitad del tile, la elimina
                if proporcion_tejido < proporcion_minima_tejido:
                    continue

                registros.append( #annade el registro del tine valido al manifiesto
                    {
                        columna_slide: ruta_slide,
                        "x": int(x),
                        "y": int(y),
                        "nivel": int(nivel),
                        "tamano_tile": int(tamano_tile),
                        columna_etiqueta: etiqueta,
                        columna_paciente: paciente,
                        "ruta_mascara": str(mascara) if mascara is not None else None,
                        "proporcion_tejido": proporcion_tejido,
                    }
                )
                aceptados += 1

                if maximo_tiles_por_slide is not None and aceptados >= maximo_tiles_por_slide:
                    break
            if maximo_tiles_por_slide is not None and aceptados >= maximo_tiles_por_slide:
                break

    return pd.DataFrame.from_records(registros)

#---------Definición del Dataset---------
class DatasetTilesWSI(Dataset):
    """Dataset que carga tiles (imágenes y máscaras opcionales) para PyTorch.

    Cada fila del `manifiesto` (DataFrame o CSV) debe contener al menos la ruta
    al slide y la etiqueta correspondiente. Opcionalmente puede incluir columnas
    `x`, `y`, `nivel`, `tamano_tile` o una ruta a la máscara.
    """

    """
    clase dataset que carga las imagenes y mascaras para entrenar el modelo
    """
    def __init__( # inicia el dataset con el manifiesto y aplicando las transformaciones 
        self,
        manifiesto: pd.DataFrame | str | Path,
        columna_slide: str = "ruta_slide",
        columna_etiqueta: str = "isup_grade",
        columna_mascara: str | None = None,
        tamano_tile: int = 256,
        nivel: int = 0,
        transformacion: Callable | None = None,
        transformacion_mascara: Callable | None = None,
        devolver_metadatos: bool = False,
    ) -> None:
        
        if isinstance(manifiesto, (str, Path)):
            manifiesto = pd.read_csv(manifiesto)

        if columna_slide not in manifiesto.columns or columna_etiqueta not in manifiesto.columns:
            raise ValueError(f"El manifiesto debe contener '{columna_slide}' y '{columna_etiqueta}'.")

        #guardar los estados
        self.manifiesto = manifiesto.reset_index(drop=True).copy()
        self.columna_slide = columna_slide
        self.columna_etiqueta = columna_etiqueta
        self.columna_mascara = columna_mascara
        self.tamano_tile = tamano_tile
        self.nivel = nivel
        self.transformacion = transformacion
        self.transformacion_mascara = transformacion_mascara
        self.devolver_metadatos = devolver_metadatos

    def __len__(self) -> int:
        return len(self.manifiesto)

    def __getitem__(self, indice: int):
        """
        metodo para obtener un item del dataset 

        """
        fila = self.manifiesto.iloc[indice]
        ruta_slide = fila[self.columna_slide]
        x = int(fila.get("x", 0))
        y = int(fila.get("y", 0))
        tamano_tile = int(fila.get("tamano_tile", self.tamano_tile))
        nivel = int(fila.get("nivel", self.nivel))
        # obinene la ruta de la mascara 
        ruta_mascara = fila.get(self.columna_mascara) if self.columna_mascara is not None else fila.get("ruta_mascara")

        # lee la imagen del tile y la mascara
        imagen = leer_region_del_slide(ruta_slide, (x, y), nivel, tamano_tile)
        mascara = None
        if ruta_mascara is not None and not pd.isna(ruta_mascara):
            mascara = leer_region_del_slide(ruta_mascara, (x, y), nivel, tamano_tile)
        etiqueta = fila[self.columna_etiqueta]

        #aplicar transforamaciones a la imagen y mascara si se definieron, sino convertir a tensor manualmente
        if self.transformacion is not None:
            transformado = self.transformacion(image=imagen)
            imagen_tensor = transformado["image"]
        else:
            #convierte a tensor automaticamente
            imagen_tensor = torch.from_numpy(imagen.copy().transpose(2, 0, 1)).float() / 255.0

        # tensor para la mascara
        if mascara is not None:
            if self.transformacion_mascara is not None:
                mascara_transformada = self.transformacion_mascara(image=mascara)
                mascara_tensor = mascara_transformada["image"]
            else:
                if mascara.ndim == 2:
                    mascara_tensor = torch.from_numpy(mascara).float() / 255.0
                else:
                    mascara_tensor = torch.from_numpy(mascara.transpose(2, 0, 1)).float() / 255.0
        else:
            mascara_tensor = None

        #normalizar las etiquetas
        if isinstance(etiqueta, (np.integer, int)):
            etiqueta_tensor = torch.tensor(int(etiqueta), dtype=torch.long)
        else:
            etiqueta_tensor = etiqueta

        #devolver los metadatos
        if self.devolver_metadatos:
            metadatos = {
                "ruta_slide": ruta_slide,
                "ruta_mascara": ruta_mascara,
                "x": x,
                "y": y,
                "nivel": nivel,
                "tamano_tile": tamano_tile,
            }
            if mascara_tensor is not None:
                return imagen_tensor, etiqueta_tensor, mascara_tensor, metadatos
            return imagen_tensor, etiqueta_tensor, metadatos

        if mascara_tensor is not None:
            return imagen_tensor, etiqueta_tensor, mascara_tensor
        return imagen_tensor, etiqueta_tensor


# ---------------------------------------------------------------------------
# Helper para construir un DataLoader a partir del manifiesto
# ---------------------------------------------------------------------------

#----------funciones para el dataloader----------
def construir_dataloader_wsi(
    manifiesto: pd.DataFrame | str | Path,
    tamano_lote: int = 16,
    mezclar: bool = True,
    num_trabajadores: int = 0,
    pin_memory: bool = True,
    columna_slide: str = "ruta_slide",
    columna_etiqueta: str = "isup_grade",
    columna_mascara: str | None = None,
    tamano_tile: int = 256,
    nivel: int = 0,
    transformacion: Callable | None = None,
    transformacion_mascara: Callable | None = None,
    devolver_metadatos: bool = False,
) -> DataLoader:
    
    """
    funcion para construir el dataloader usando el manifiesto 
    """
    dataset = DatasetTilesWSI( 
        manifiesto=manifiesto,
        columna_slide=columna_slide,
        columna_etiqueta=columna_etiqueta,
        columna_mascara=columna_mascara,
        tamano_tile=tamano_tile,
        nivel=nivel,
        transformacion=transformacion,
        transformacion_mascara=transformacion_mascara,
        devolver_metadatos=devolver_metadatos,
    )
    return DataLoader(
        dataset,
        batch_size=tamano_lote,
        shuffle=mezclar,
        num_workers=num_trabajadores,
        pin_memory=pin_memory,
    )