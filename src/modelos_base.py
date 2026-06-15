from __future__ import annotations
from typing import Iterable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class BaselineClaseMayoritaria:
    """
    clase para baseline de clasificacion 
    """

    def __init__(self) -> None:
        
        self.clase_mayoritaria = None

    def entrenar(self, etiquetas: Iterable) -> "BaselineClaseMayoritaria":
       
        #aprende cual es la clase mas frecuente en las etiquetas.
       
        valores, conteos = np.unique(list(etiquetas), return_counts=True)
        self.clase_mayoritaria = valores[np.argmax(conteos)]
        return self

    def predecir(self, entradas: Iterable) -> np.ndarray:
        
        #devuelve la clase mayoritaria para todas las entradas
        
        if self.clase_mayoritaria is None:
            raise ValueError("El baseline debe entrenarse antes de predecir.")
        entradas = list(entradas)
        return np.full(shape=len(entradas), fill_value=self.clase_mayoritaria)

    def evaluar(self, entradas: Iterable, etiquetas_reales: Iterable) -> float:
        #calcula la accuracy del baseline
        predicciones = self.predecir(entradas)
        return accuracy_score(list(etiquetas_reales), predicciones)


def _convertir_a_numpy_imagen(imagen) -> np.ndarray:
    """
    funcion para convertir una imagen a un arreglo de numpy
    """
    if isinstance(imagen, torch.Tensor):
        arreglo = imagen.detach().cpu().numpy()
        if arreglo.ndim == 3 and arreglo.shape[0] in (1, 3):
            arreglo = np.transpose(arreglo, (1, 2, 0))
        return arreglo
    return np.asarray(imagen)


def extraer_caracteristicas_tile(imagen) -> np.ndarray:
    """
    funcion para extrar las caracteristicas del tile 
    """
    arreglo = _convertir_a_numpy_imagen(imagen).astype(np.float32)

    if arreglo.max() > 1.5:
        arreglo = arreglo / 255.0

    if arreglo.ndim == 2: #si la imagen esta en grises
        arreglo = np.repeat(arreglo[..., None], 3, axis=2)

    #estadisticas rgb
    media_rgb = arreglo.mean(axis=(0, 1))
    desviacion_rgb = arreglo.std(axis=(0, 1))

    #estadisticas de escala de grises
    escala_gris = arreglo.mean(axis=2)
    media_gris = np.array([escala_gris.mean()], dtype=np.float32)
    desviacion_gris = np.array([escala_gris.std()], dtype=np.float32)

    #saturacion 
    saturacion = arreglo.max(axis=2) - arreglo.min(axis=2)
    media_saturacion = np.array([saturacion.mean()], dtype=np.float32)

    gradiente_y = np.abs(np.diff(escala_gris, axis=0)).mean() if escala_gris.shape[0] > 1 else 0.0
    gradiente_x = np.abs(np.diff(escala_gris, axis=1)).mean() if escala_gris.shape[1] > 1 else 0.0
    energia_bordes = np.array([gradiente_x + gradiente_y], dtype=np.float32)

    return np.concatenate([
        media_rgb,
        desviacion_rgb,
        media_gris,
        desviacion_gris,
        media_saturacion,
        energia_bordes,
    ])


def construir_matriz_caracteristicas(dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    funcion que recorre el dataset y construye X features y y etiquetas
    """
    caracteristicas = []
    etiquetas = []

    
    for indice in range(len(dataset)):
        #ciclo for para recorrer el dataset y extraer las caracteristicas de cada tile
        muestra = dataset[indice]
        if len(muestra) == 3:
            imagen, etiqueta, _ = muestra
        else:
            imagen, etiqueta = muestra
        caracteristicas.append(extraer_caracteristicas_tile(imagen)) #extraer las caracteristicas del tile 
        if isinstance(etiqueta, torch.Tensor):
            etiquetas.append(etiqueta.item())
        else:
            etiquetas.append(etiqueta)

    return np.vstack(caracteristicas), np.asarray(etiquetas)


def entrenar_random_forest_baseline(
    dataset_entrenamiento,
    dataset_validacion=None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """
    funcion para entrenar un random forest y ver su desempeno
    """
    x_entrenamiento, y_entrenamiento = construir_matriz_caracteristicas(dataset_entrenamiento)
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    modelo.fit(x_entrenamiento, y_entrenamiento)

    metricas: dict[str, float] = {}
    if dataset_validacion is not None:
        x_validacion, y_validacion = construir_matriz_caracteristicas(dataset_validacion)
        predicciones = modelo.predict(x_validacion)
        metricas = {
            "accuracy": accuracy_score(y_validacion, predicciones),
            "balanced_accuracy": balanced_accuracy_score(y_validacion, predicciones),
            "f1_macro": f1_score(y_validacion, predicciones, average="macro"),
        }

    return modelo, metricas


def evaluar_clasificador(modelo, dataset) -> dict[str, float]:
    """
    funcion para evaluar un clasificador para ver el desempenno de los modelos
    """
    x, y = construir_matriz_caracteristicas(dataset)
    predicciones = modelo.predict(x)
    return {
        "accuracy": accuracy_score(y, predicciones),
        "balanced_accuracy": balanced_accuracy_score(y, predicciones),
        "f1_macro": f1_score(y, predicciones, average="macro"),
    }