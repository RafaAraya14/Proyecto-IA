from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import timm
import numpy as np
from sklearn.metrics import cohen_kappa_score
import optuna
import json

from .dataset_wsi import construir_dataloader_wsi


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MANIFIESTO = Path("manifests") / "manifiesto_tiles.csv"
RUTA_MANIFIESTO_VAL = Path("manifests") / "manifiesto_tiles_validacion.csv"


def crear_dummy_dataloader(batch_size=8, img_size=256, num_muestras=32):
    """
    funcion para crear un dataloader falso para pruebas
    """
    imagenes_falsas = torch.randn(num_muestras, 3, img_size, img_size)
    etiquetas_falsas = torch.randint(0, 6, (num_muestras,))
    dataset = TensorDataset(imagenes_falsas, etiquetas_falsas)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class AgenteCancerProstata(nn.Module):
    """
    clase modelo de clasificacion multiclase basado en efficientnet de timm
    """

    def __init__(self, num_clases=6, nombre_modelo="efficientnet_b0", preentrenado=True):
        super().__init__()
        self.backbone = timm.create_model(nombre_modelo, pretrained=preentrenado) #crea el backbone de timm
        num_features = self.backbone.classifier.in_features 
        self.backbone.classifier = nn.Linear(num_features, num_clases)

    def forward(self, x):
        
        return self.backbone(x)


def entrenar_una_epoca(modelo, dataloader, criterio, optimizador, dispositivo):
    """
    entrena el modelo durante una epoca y devuelve loss promedio y qwk
    """
    modelo.train()
    perdida_total = 0.0
    todas_predicciones = []
    todas_etiquetas = []

    for imagenes, etiquetas in dataloader:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        # paso de entrenamiento normal: forward, loss, backward, step
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        # guarda info para metricas
        perdida_total += perdida.item()
        _, predicciones = torch.max(salidas, 1)
        todas_predicciones.extend(predicciones.cpu().numpy())
        todas_etiquetas.extend(etiquetas.cpu().numpy())

    perdida_promedio = perdida_total / len(dataloader)
    # calcula qwk, util cuando la clase es ordinal como isup
    qwk = cohen_kappa_score(todas_etiquetas, todas_predicciones, weights="quadratic")
    return perdida_promedio, qwk


def funcion_objetivo(trial):
    """
    funcion que usa optuna para probar un lr, entrenar y devolver una metrica
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optuna propone el learning rate 
    lr_sugerido = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    modelo = AgenteCancerProstata(num_clases=6).to(dispositivo)

    # valida que exista el manifiesto antes de crear el dataloader
    if not RUTA_MANIFIESTO.exists():
        raise FileNotFoundError(
            f"No se encontro el manifiesto esperado en {RUTA_MANIFIESTO}. Primero genera los CSV dentro de data/."
        )

    dataloader = construir_dataloader_wsi(
        manifiesto=RUTA_MANIFIESTO,
        tamano_lote=8,
        mezclar=True,
        num_trabajadores=2,
        columna_slide="ruta_slide",
        columna_etiqueta="isup_grade",
        tamano_tile=256,
    )

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr_sugerido)

    historial_loss = []
    ultima_perdida = 0.0
    # en cada trial entrena 20 epocas
    for epoca in range(20):
        ultima_perdida, qwk = entrenar_una_epoca(modelo, dataloader, criterio, optimizador, dispositivo)
        historial_loss.append(ultima_perdida)

    def validar_modelo(modelo, dataloader_val, dispositivo):
        """
        funcion que evalua el modelo en validacion y devuelve loss u qwk.
        """
        modelo.eval()
        perdida_total = 0.0
        todas_predicciones = []
        todas_etiquetas = []
        criterio_local = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in dataloader_val:
                # toma imagenes y etiquetas del batch
                imagenes, etiquetas = batch[0], batch[1]
                imagenes = imagenes.to(dispositivo)
                etiquetas = etiquetas.to(dispositivo)

                # forward de validacion
                salidas = modelo(imagenes)
                perdida = criterio_local(salidas, etiquetas)
                perdida_total += perdida.item()
                _, predicciones = torch.max(salidas, 1)
                todas_predicciones.extend(predicciones.cpu().numpy())
                todas_etiquetas.extend(etiquetas.cpu().numpy())

        perdida_promedio = perdida_total / len(dataloader_val)
        qwk_val = cohen_kappa_score(todas_etiquetas, todas_predicciones, weights="quadratic")
        return perdida_promedio, qwk_val

    # guarda pesos y curva de loss del trial
    ruta_guardado = RAIZ_PROYECTO / f"models/modelo_efficientnet_lr_{lr_sugerido:.6f}.pth"
    torch.save(modelo.state_dict(), str(ruta_guardado))
    print(f"Modelo guardado en: {ruta_guardado}")

    np.save(RAIZ_PROYECTO / f"models/historial_loss_{lr_sugerido:.6f}.npy", np.array(historial_loss))

    resultado_validacion = None
    ruta_metrics = RAIZ_PROYECTO / "models" / f"metrics_lr_{lr_sugerido:.6f}.json"
    try:
        if RUTA_MANIFIESTO_VAL.exists():
            dataloader_val = construir_dataloader_wsi(
                manifiesto=RUTA_MANIFIESTO_VAL,
                tamano_lote=8,
                mezclar=False,
                num_trabajadores=2,
                columna_slide="ruta_slide",
                columna_etiqueta="isup_grade",
                tamano_tile=256,
            )
            perdida_val, qwk_val = validar_modelo(modelo, dataloader_val, dispositivo)
            resultado_validacion = {"val_loss": float(perdida_val), "val_qwk": float(qwk_val)}
            print(f"Validación - pérdida: {perdida_val:.4f}, QWK: {qwk_val:.4f}")

            # guarda metricas en json
            ruta_metrics.parent.mkdir(parents=True, exist_ok=True)
            with open(ruta_metrics, "w", encoding="utf-8") as fh:
                json.dump({
                    "lr": float(lr_sugerido),
                    "train_last_loss": float(ultima_perdida),
                    **resultado_validacion,
                }, fh, indent=2)
        else:
            print(f"No se encontró manifiesto de validación en {RUTA_MANIFIESTO_VAL}; se omite validación.")
    except Exception as exc:
        print(f"Error durante validación: {exc}")

    # devuelve a optuna la loss de validacion si existe, si no la ultima de train
    if resultado_validacion is not None:
        return float(resultado_validacion["val_loss"])
    return ultima_perdida


if __name__ == "__main__":
    print("Iniciando búsqueda de Hiperparámetros con Optuna...")

    # crea estudio para minimizar la loss objetivo
    estudio = optuna.create_study(direction="minimize")

    # fija semillas para tener algo de reproducibilidad
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # corre 3 trials por defecto
    estudio.optimize(funcion_objetivo, n_trials=3)

    print("Busqueda completada.")
    print(f"Mejor Learning Rate encontrado: {estudio.best_params['lr']:.6f}")
    print(f"Pérdida más baja lograda: {estudio.best_value:.4f}")

