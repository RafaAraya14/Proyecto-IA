#importaciones

import sys
from pathlib import Path
from networkx import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openslide
import subprocess

from src.modelo_principal import funcion_objetivo
from src.explicabilidad import probar_grad_cam

# Procesamiento de imágenes médicas y archivos
import tifffile
import openslide
import imagecodecs

# Deep Learning y PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import timm

# Machine Learning y Métricas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from src.puente_integracion import main as generar_artefactos_datos

# Optimizacion de datos
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from src.DataseTiles import DatasetTiles
from src.DatasetTilesConMascara import DatasetTilesConMascara

# Funciones 



def leer_tiff_como_arreglo(ruta):
    with tifffile.TiffFile(ruta) as tif:
        serie = tif.series[0]
        return serie.asarray() 
    
def calcular_proporcion_tejido(tile, umbral_blanco=245):
        proporcion_fondo = (tile >= umbral_blanco).mean()
        return 1.0 - proporcion_fondo

def cargar_tile(ruta_slide, x, y, tamano_tile):
#extrae el tile de la imagen
    slide = openslide.OpenSlide(ruta_slide)
    imagen = slide.read_region((int(x), int(y)), 0, (int(tamano_tile), int(tamano_tile))).convert("RGB")
    slide.close()
    return np.array(imagen)

def extraer_features(imagen):
    imagen = imagen.astype(np.float32) / 255.0

    #stats de color
    media_rgb, std_rgb = imagen.mean(axis=(0, 1)), imagen.std(axis=(0, 1))

    #stats de intensidad luminosa
    gris = imagen.mean(axis=2)
    media_gris, std_gris = np.array([gris.mean()], dtype=np.float32), np.array([gris.std()], dtype=np.float32)

    #variabilidad y gradiente
    saturacion = imagen.max(axis=2) - imagen.min(axis=2)
    grad_y, grad_x = np.abs(np.diff(gris, axis=0)).mean(), np.abs(np.diff(gris, axis=1)).mean()

    return np.concatenate([media_rgb, std_rgb, media_gris, std_gris, np.array([saturacion.mean()]), np.array([grad_x + grad_y])])

def construir_matriz_features(df_entrada, max_ejemplos=None):
    #crea la matriz de caracteristicas y el vector de etiquetas
    filas = df_entrada.head(max_ejemplos) if max_ejemplos else df_entrada
    X = [extraer_features(cargar_tile(f["ruta_slide"], f["x"], f["y"], f["tamano_tile"])) for _, f in filas.iterrows()]
    return np.vstack(X), filas["isup_grade"].values

def visualizar_muestra(df):
    inicio = time.time()

    # seleccionar el primer elemento que tenga máscara
    fila = df.loc[df["tiene_mascara"]].iloc[0]
    ruta_slide = fila["ruta_slide"]
    ruta_mascara = fila["ruta_mascara"]
    image_id = fila["image_id"]

    # imprimir info
    print("image_id:", image_id)
    print("slide:", ruta_slide)
    print("mask:", ruta_mascara)

    # intentar cargar y visualizar las imágenes con manejo de errores
    try:
        slide = leer_tiff_como_arreglo(ruta_slide)
        mask = leer_tiff_como_arreglo(ruta_mascara)

        # normalizar las dimensiones
        if slide.ndim == 2:
            slide = np.stack([slide] * 3, axis=-1)
        if mask.ndim == 3 and mask.shape[-1] > 1:
            mask = mask[..., 0]

        # definir y recortar las imagenes de un minimo de 512 por 512 pixeles
        alto = min(512, slide.shape[0], mask.shape[0])
        ancho = min(512, slide.shape[1], mask.shape[1])
        slide_crop = slide[:alto, :ancho]
        mask_crop = mask[:alto, :ancho]

        # mostrar una camparativa entre el slide y la mascara usando matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(slide_crop)
        axes[0].set_title(f"Slide: {image_id}")
        axes[0].axis("off")

        axes[1].imshow(mask_crop, cmap="gray")
        axes[1].set_title("Máscara")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        # capturar error si no se encuentran las imagenes o hay fallo en la lectura
        print(f"Error al cargar o procesar las imágenes: {e}")

    fin = time.time()
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")

def extraer_tiles(df):
    inicio = time.time()
    #seleccionar un caso y se carga el slide y la mascara
    fila = df.loc[df["tiene_mascara"]].iloc[0]
    ruta_slide = fila["ruta_slide"]
    ruta_mascara = fila["ruta_mascara"]

    print(ruta_slide, ruta_mascara)


    slide = openslide.OpenSlide(ruta_slide)
    mask = tifffile.imread(ruta_mascara)

    #se normaliza la tgrafica por el formato que sea
    if mask.ndim == 3:
        mask = mask[..., 0]

    #define coordenadas aleatorias, del slide
    tile_size = 512
    ancho, alto = slide.level_dimensions[0]
    x = np.random.randint(0, ancho - tile_size)
    y = np.random.randint(0, alto - tile_size)

    #se lee el tile del  slide y se extrar de la mascara el mismos segmento
    tile_slide = np.array(slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB"))
    tile_mask = mask[y:y+tile_size, x:x+tile_size]


    print("Coordenadas:", x, y)
    print("Slide tile:", tile_slide.shape)
    print("Mask tile:", tile_mask.shape)

    #se muestran los resultados obtenidos del slide y su marca cirrespondiente
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(tile_slide)
    axes[0].set_title("Tile del slide")
    axes[0].axis("off")

    axes[1].imshow(tile_mask, cmap="gray")
    axes[1].set_title("Tile de la máscara")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    fin = time.time()
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")

def generar_manifiesto(df):
    # hiperparametros
    tamano_tile = 512
    salto = 512
    umbral_blanco = 245
    proporcion_minima_tejido = 0.50
    max_tiles_por_imagen = 20
    max_imagenes = 98

    inicio = time.time()

    registros = []
    df_con_mascara = df.loc[df["tiene_mascara"]].head(max_imagenes).copy()

    # Ciclo principal
    for i, (_, fila) in enumerate(df_con_mascara.iterrows(), 1):
        ruta_slide = fila["ruta_slide"]
        ruta_mascara = fila["ruta_mascara"]
        image_id = fila["image_id"]

        print(f"[{i}/{max_imagenes}] Procesando imagen: {image_id}...")

        slide = openslide.OpenSlide(ruta_slide)
        mask_file = tifffile.imread(ruta_mascara)
        if mask_file.ndim == 3:
            mask_file = mask_file[..., 0]

        ancho, alto = slide.level_dimensions[0]
        tiles_guardados = 0

        for y in range(0, alto - tamano_tile + 1, salto):
            for x in range(0, ancho - tamano_tile + 1, salto):
                tile_mask = mask_file[y:y+tamano_tile, x:x+tamano_tile]
                proporcion_tejido = calcular_proporcion_tejido(tile_mask, umbral_blanco=umbral_blanco)

                if proporcion_tejido < proporcion_minima_tejido:
                    continue

                registros.append({
                    "image_id": image_id,
                    "ruta_slide": ruta_slide,
                    "ruta_mascara": ruta_mascara,
                    "x": x,
                    "y": y,
                    "tamano_tile": tamano_tile,
                    "proporcion_tejido": proporcion_tejido,
                })

                tiles_guardados += 1
                # Imprime progreso de tiles por imagen
                print(f"  -> Tile {tiles_guardados} guardado en (x={x}, y={y})")

                if tiles_guardados >= max_tiles_por_imagen: break
            if tiles_guardados >= max_tiles_por_imagen: break

        slide.close()
        print(f"Imagen {image_id} terminada. Tiles totales: {tiles_guardados}")

    manifiesto = pd.DataFrame(registros)
    print("-" * 30)
    print("Proceso finalizado.")
    print("Tiles totales encontrados:", len(manifiesto))
    manifiesto.to_csv("manifests/manifiesto_tiles.csv", index=False)

    fin = time.time()
    print(f"Tiempo total de ejecución: {fin - inicio:.4f} segundos")

def entrenar_baseline(manifiesto, df):
    inicio = time.time()
    #Validar el manifiesto de los tiles con etiquetas de ISUP
    datos = manifiesto.merge(df[["image_id", "isup_grade"]], on="image_id", how="left")

    #balancear la particion en trial y vali para que se mantenga constante en cada dataset
    train_df, val_df = train_test_split(
        datos, test_size=0.2, random_state=42, shuffle=True, stratify=datos["isup_grade"]
    )

    # Guardar el manifiesto de validación para Optuna
    ruta_manifiesto_val_csv = Path('manifests') / 'manifiesto_tiles_validacion.csv'
    ruta_manifiesto_val_csv.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(ruta_manifiesto_val_csv, index=False)
    print(f"Manifiesto de validación guardado en: {ruta_manifiesto_val_csv}")

    #predecir la clase mnas frecuente
    clase_mayoritaria = train_df["isup_grade"].mode()[0]
    pred_val_mayoritaria = np.full(len(val_df), clase_mayoritaria)

    print("=== Baseline Ingenuo (Clase Mayoritaria) ===")
    print(f"F1 Macro: {f1_score(val_df['isup_grade'], pred_val_mayoritaria, average='macro'):.4f}")

    max_ejemplos = 500

    #realizar el entrenamiento
    X_train, y_train = construir_matriz_features(train_df, max_ejemplos=max_ejemplos)
    X_val, y_val = construir_matriz_features(val_df, max_ejemplos=max_ejemplos)

    #modelo no lineal para ver el rendimiento con features manuales
    modelo_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train, y_train)

    # evaluar los resultados
    pred_val_rf = modelo_rf.predict(X_val)
    print("\n=== Random Forest Baseline (Features Estadísticos) ===")
    print(f"Accuracy: {accuracy_score(y_val, pred_val_rf):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, pred_val_rf):.4f}")
    print(f"F1 Macro: {f1_score(y_val, pred_val_rf, average='macro'):.4f}")

    fin = time.time()

    
    print(f"Tiempo total de ejecución: {fin - inicio:.4f} segundos")
    
    return modelo_rf, X_val, y_val, pred_val_rf

def ejecutar_tiling(ruta_csv, dir_imagenes, dir_salida, tamano_tile=256, max_tiles=10):

    """
    Ejecuta el script de integración de tiles como un subproceso.
    """
    inicio = time.time()
    
    # Construimos la lista de argumentos. 
    # Usar una lista es más seguro que pasar un string largo (evita errores de shell).
    comando = [
        sys.executable, "-m", "src.puente_integracion",
        "--ruta-csv", str(ruta_csv),
        "--directorio-imagenes", str(dir_imagenes),
        "--directorio-salida", str(dir_salida),
        "--tamano-tile", str(tamano_tile),
        "--maximo-tiles-por-slide", str(max_tiles)
    ]
    
    print(f"--- Ejecutando comando: {' '.join(comando)} ---")
    
    try:
        # check=True lanzará una excepción si el script falla (retorno != 0)
        # capture_output=True nos permite ver lo que el script imprima
        resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
        
        # Opcional: imprimir la salida del script
        print("Salida del script:")
        print(resultado.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el tiling: {e}")
        print("Error stderr:", e.stderr)
    
    fin = time.time()
    print(f"Tiempo total de ejecución: {fin - inicio:.4f} segundos")

def main():
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    print("-----2. Definiendo rutas del dataset y conteo de archivos-----")
    #definir las imagenes

    ruta_base = Path("data")
    ruta_csv = ruta_base / "train.csv"
    ruta_imagenes = ruta_base / "train_images"
    ruta_mascaras = ruta_base / "train_label_masks"

    imagenes = list(ruta_imagenes.glob("*.tiff"))   
    mascaras = list(ruta_mascaras.glob("*.tiff"))

    #imprimir el conteo de imagenes y mascaras
    print("Cantidad iamgenes:", len(imagenes))
    print("Cantidad mascaras:", len(mascaras))
    print("Primeras imagenes:", [p.name for p in imagenes[:5]])
    print("Primeras mascaras:", [p.name for p in mascaras[:5]])

    print("-----2. Rutas del dataset y conteo de archivos definidas-----")

    print("-----3. Contando imagenes y mascaras-----")

    inicio = time.time()

    #crear el dataframe cargando el csv
    df = pd.read_csv(ruta_csv)

    #crear el conjunto de archivos extrayuendo los IDs de las mascaras
    ids_mascaras = {
        p.name.replace("_mask.tiff", "")
        for p in ruta_mascaras.glob("*.tiff")
    }

    #ve si si imagen tiene una mascara asociada
    df["tiene_mascara"] = df["image_id"].isin(ids_mascaras)

    #imprimir cuantas imagenes tienen mascaras
    print(df["tiene_mascara"].value_counts())

    # Esto es muy útil para saber qué parte de tu dataset NO podrás usar para segmentación.
    # muestra los id que no tiene mascara,
    print("Ejemplos sin máscara:", df.loc[~df["tiene_mascara"], "image_id"].head(20).tolist())

    fin = time.time()
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")

    print("-----3. Imagenes y mascaras contadas-----")

    print("-----4. Configurando rutas de cargas dinamicas-----")
    inicio = time.time()
    #ruta para cada iamgen
    df["ruta_slide"] = df["image_id"].apply(lambda x: str(ruta_imagenes / f"{x}.tiff"))

    #ruta para las mascaras
    df["ruta_mascara"] = df["image_id"].apply(
        lambda x: str(ruta_mascaras / f"{x}_mask.tiff")
        if (ruta_mascaras / f"{x}_mask.tiff").exists() else None
    )

    fin = time.time()
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")
    print("-----4. Rutas de cargas dinamicas configuradas-----")

    print("----------Modelo de prediccion de cancer de prostata basado en imagenes de biopsias----------")

    while True:

        
        print("\nSeleccione una opción:")
        print("A. Visualizar una muestra de slide y su máscara")
        print("B. Extraer y visualizar tiles de una imagen")
        print("C. Generar el manifiesto de tiles")
        print("D. Entrenar el baseline con Random Forest")
        print("E. Ejecutar tiling con el script de integración")
        print("F. Optimización de hiperparámetros con Optuna y prueba de Grad-CAM")
        print("G. Ejecutar matriz de confusión para el Random Forest")
        print("H. Visualizar curva de aprendizaje del mejor modelo")
        print("I. Comparar resultados de Optuna y generar tabla comparativa")
        print("J. Salir del programa")

        opcion = input("Ingrese la letra de la opción deseada: ")

        if opcion == "A":
            print("-----5. Visualizando una muestra -----")
            visualizar_muestra(df)
            print("-----5. Muestra visualizada -----")
        elif opcion == "B":
            print("-----6. Extrayendo Tiles -----")
            extraer_tiles(df)
            print("-----6. Tiles extraidos -----")

        elif opcion == "C": 
            print("-----7. Generando manifiesto -----")
            generar_manifiesto(df)
            print("-----7. Manifiesto generado -----")

            print("-----8. Implementacion de Dataset y Dataloader-----")

            # Cargar el manifiesto que acabas de guardar
            inicio = time.time()
            ruta_manifiesto = Path('manifests/manifiesto_tiles.csv')
            manifiesto = pd.read_csv(ruta_manifiesto)

            print("Manifiesto cargado:", manifiesto.shape)

            dataset = DatasetTiles(manifiesto)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

            # Tomar un batch
            imagenes, ids, xs, ys = next(iter(dataloader))

            print("Batch de imágenes:", imagenes.shape)
            print("IDs:", ids)
            print("Coordenadas X:", xs)
            print("Coordenadas Y:", ys)

            # Mostrar el primer tile del batch
            plt.figure(figsize=(6, 6))
            tile = imagenes[0].permute(1, 2, 0).numpy()
            plt.imshow(tile)
            plt.axis("off")
            plt.show()
            fin = time.time()
            print(f"Tiempo total de ejecución: {fin - inicio:.4f} segundos")

            print("-----8. Dataset y Dataloader implementados -----")

        elif opcion == "D":
            print("-----9. Iniciando entrenamiento del baseline-----")
            modelo_rf, X_val, y_val, pred_val_rf = entrenar_baseline(manifiesto, df)
            print("-----9. Entrenamiento del baseline Finalizado-----")

        elif opcion == "E":
            print("-----10. Ejecutando tiling con el script de integración -----")
            ejecutar_tiling(ruta_csv="data/train.csv",dir_imagenes="data/train_images",dir_salida=".",tamano_tile=256,max_tiles=10)
            print("-----10. Tiling ejecutado -----")
        elif opcion == "F":
            print("-----11. Optimizacion y Prueba Grad-CAM")

            inicio = time.time()

            #agregar el dataset al path
            sys.path.append('src')

            #agregar la ruta de trabajo local
            sys.path.append('')

            #cargar optuna con los datos para la optimizacion
            estudio = optuna.create_study(direction="minimize")
            estudio.optimize(funcion_objetivo, n_trials=3)


            print("\n----Resultados de Optuna----")
            print(f"Mejor Learning Rate encontrado: {estudio.best_params['lr']:.6f}")

            # 5. Prueba de Explicabilidad
            print("\n----Grad Cam----")
            probar_grad_cam()

            fin = time.time()
            print(f"Tiempo total de ejecución: {fin - inicio:.4f} segundos")

            print("----- 11. Optimizacion y Prueba Grad-CAM Finalizada -----  ")
            
        elif opcion == "G":
            print("-----12. Ejecutando matriz de confusion para el random forest -----")
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            

            # Obtener las etiquetas verdaderas y las predicciones del modelo Random Forest
            y_true = y_val
            y_pred = pred_val_rf

            # Calcular la matriz de confusión
            cm = confusion_matrix(y_true, y_pred)

            # Obtener los nombres de las clases (ej. del 0 al 5 si esup_grade va de 0 a 5)
            # Asumiendo que isup_grade son enteros de 0 a 5
            class_names = sorted(list(np.unique(y_true)))

            # Crear la visualización de la matriz de confusión
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

            ax.set_title('Matriz de Confusión - Random Forest Classifier')
            ax.set_xlabel('Etiqueta Predicha')
            ax.set_ylabel('Etiqueta Verdadera')
            plt.tight_layout()
            plt.show()
            print("-----12. Matriz de confusion ejecutada -----")

        elif opcion == "H":
            import matplotlib.pyplot as plt
            from pathlib import Path

            # Sustituye '0.007430' por el valor de LR de tu mejor modelo (mira tus archivos .npy)
            mejor_lr = "0.000340"
            archivo_historial = Path(f"models/historial_loss_{mejor_lr}.pth").with_suffix('.npy')

            # Cargar y graficar
            historial = np.load(archivo_historial)

            plt.figure(figsize=(10, 5))
            plt.plot(historial, label='Loss Entrenamiento', marker='o', color='tab:blue')
            plt.title(f'Curva de Aprendizaje - Mejor Modelo (LR: {mejor_lr})')
            plt.xlabel('Épocas')
            plt.ylabel('Loss (CrossEntropy)') 
            plt.grid(True)
            plt.legend()
            plt.show()
            
        elif opcion == "I":
            import json

            # Cargar todos los resultados de los trials
            archivos_json = list(Path('.').glob('models/metrics_lr_*.json'))
            resultados = [json.load(open(f)) for f in archivos_json]

            # Crear DataFrame y mostrar
            df_res = pd.DataFrame(resultados)
            print("--- Tabla Comparativa: Rendimiento por Learning Rate ---")
            # Verificar si 'val_qwk' existe antes de intentar ordenar
            if 'val_qwk' in df_res.columns:
                print(df_res.sort_values(by='val_qwk', ascending=False))
            else:
                print("La columna 'val_qwk' no se encontró en el DataFrame de resultados. Verifique que los archivos de métricas se hayan generado correctamente.")
                print(df_res) # Mostrar el DataFrame sin ordenar para inspección

            df_res.to_csv("models/resultados_finales_etapa2.csv", index=False)
            
        elif opcion == "J":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intente nuevamente.")

if __name__ == "__main__":
    main()
