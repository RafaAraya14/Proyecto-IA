"""explicabilidad.py

archivo para probar la parte de explicabilidad con grad-cam sobre el modelo.
sirve para ver que zonas de la imagen estan influyendo mas en la prediccion.

nota:
- usa pytorch-grad-cam y matplotlib para crear el mapa de calor.
- la prueba se hace con ruido, entonces es solo para verificar que corra.
"""

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .modelo_principal import AgenteCancerProstata


# ruta raiz del proyecto guarda la salida en una ruta fija
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


def probar_grad_cam():
    """
    funcion de prueba para grad-cam.
    crea un modelo, pasa una imagen random y guarda un mapa de calor.
    """
    print("Iniciando módulo de Explicabilidad (XAI) con Grad-CAM...")

    # detecta si hay gpu
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # crea el modelo con 6 clases
    modelo = AgenteCancerProstata(num_clases=6).to(dispositivo)
    modelo.eval()  
    
    capas_objetivo = [modelo.backbone.conv_head]

    cam = GradCAM(model=modelo, target_layers=capas_objetivo)

    # crea un tensor de entrada
    tensor_entrada = torch.randn(1, 3, 256, 256).to(dispositivo)

    targets = [ClassifierOutputTarget(5)]

    # ejecuta grad-cam y toma el mapa de la primera imagen del batch
    mapa_escala_grises = cam(input_tensor=tensor_entrada, targets=targets)[0, :]

    # prepara la imagen para superponer el mapa de calor 
    imagen_rgb = tensor_entrada[0].cpu().numpy().transpose(1, 2, 0)
    imagen_rgb = (imagen_rgb - np.min(imagen_rgb)) / (np.max(imagen_rgb) - np.min(imagen_rgb))

    #superpone el mapa de calor
    visualizacion = show_cam_on_image(imagen_rgb, mapa_escala_grises, use_rgb=True)

    # guarda la visualizacion 
    plt.imshow(visualizacion)
    plt.title("Grad-CAM: Explicabilidad del Agente (ISUP 5)")
    plt.axis("off")
    ruta_salida = RAIZ_PROYECTO / "mapa_calor_prueba.png"
    plt.savefig(str(ruta_salida), bbox_inches="tight")

    print(f"----------Mapa de calor generado: {ruta_salida}----------")


if __name__ == "__main__":
    probar_grad_cam()