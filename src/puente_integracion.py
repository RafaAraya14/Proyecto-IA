from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .dataset_wsi import construir_dataloader_wsi, construir_manifiesto_tiles, construir_transformaciones_entrenamiento
from .particion_pacientes import guardar_particiones_paciente

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = RAIZ_PROYECTO / "./data"


def construir_ruta_archivo(directorio_base: str | Path, nombre_archivo: str) -> Path:
    """'Funcion para construir la ruta del archivo"""
    return Path(directorio_base) / nombre_archivo


def preparar_dataframe_base(
    ruta_csv: str | Path,
    directorio_imagenes: str | Path,
    directorio_mascaras: str | Path | None = None,
    extension_imagen: str = ".tiff",
    sufijo_mascara: str = "_mask.tiff",
) -> pd.DataFrame:
    """
    funcion para prerparar el dataframe con las rutas de los slides y de las mascaras 
    """

    df = pd.read_csv(ruta_csv)

    # validar una imagen 
    if "image_id" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'image_id'.")

    # copiar el dataset para no modificar el original 
    df = df.copy()
    #agrega la ruta de cada slide al dataframe las columnas de ruta_slide y ruta mascara
    df["ruta_slide"] = df["image_id"].apply(
        lambda image_id: str(Path(directorio_imagenes) / f"{image_id}{extension_imagen}")
    )

    if directorio_mascaras is not None:
        df["ruta_mascara"] = df["image_id"].apply(
            lambda image_id: str(Path(directorio_mascaras) / f"{image_id}{sufijo_mascara}")
        )

    return df


def generar_artefactos_datos(
    ruta_csv: str | Path,
    directorio_imagenes: str | Path,
    directorio_salida: str | Path,
    directorio_mascaras: str | Path | None = None,
    ruta_mapeo_paciente: str | Path | None = None,
    extension_imagen: str = ".tiff",
    columna_paciente: str = "patient_id",
    columna_etiqueta: str = "isup_grade",
    tamano_tile: int = 256,
    salto: int | None = None,
    nivel: int = 0,
    proporcion_minima_tejido: float = 0.3,
    umbral_blanco: int = 245,
    maximo_tiles_por_slide: int | None = 20,
) -> dict[str, Path]:
    """
    funcion para generar los artefactos de datos para el entrenamiento"""

    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)
    (directorio_salida / "splits").mkdir(parents=True, exist_ok=True)
    (directorio_salida / "manifests").mkdir(parents=True, exist_ok=True)

    # construye el dataset
    df_base = preparar_dataframe_base(
        ruta_csv=ruta_csv,
        directorio_imagenes=directorio_imagenes,
        directorio_mascaras=directorio_mascaras,
        extension_imagen=extension_imagen,
    )

    archivos_salida: dict[str, Path] = {}

    #so hay mapeo de pacientes
    if ruta_mapeo_paciente is not None:
        archivos_salida = guardar_particiones_paciente(
            ruta_csv=ruta_csv,
            directorio_salida=directorio_salida / "splits",
            columna_paciente=columna_paciente,
            columna_imagen="image_id",
            columna_etiqueta=columna_etiqueta,
            ruta_mapeo_paciente=ruta_mapeo_paciente,
        )
    #construye el manifiesto de tiles
    manifiesto = construir_manifiesto_tiles(
        df_slides=df_base,
        columna_slide="ruta_slide",
        columna_etiqueta=columna_etiqueta,
        columna_paciente=columna_paciente,
        directorio_mascaras=directorio_mascaras,
        tamano_tile=tamano_tile,
        salto=salto,
        nivel=nivel,
        proporcion_minima_tejido=proporcion_minima_tejido,
        umbral_blanco=umbral_blanco,
        maximo_tiles_por_slide=maximo_tiles_por_slide,
    )

    ruta_manifiesto = directorio_salida / "manifests" / "manifiesto_tiles.csv"
    manifiesto.to_csv(ruta_manifiesto, index=False)
    archivos_salida["manifiesto"] = ruta_manifiesto

    return archivos_salida


def construir_argumentos() -> argparse.Namespace:
    """
    funcion para construir los argumentos 
    """
    parser = argparse.ArgumentParser(description="Puente de integración Colab <-> proyecto local.")
    parser.add_argument("--ruta-csv", required=True, help="Ruta de train.csv")
    parser.add_argument("--directorio-imagenes", required=True, help="Carpeta de train_images")
    parser.add_argument("--directorio-salida", default=str(DIRECTORIO_DATOS), help="Carpeta base donde guardar CSVs")
    parser.add_argument("--directorio-mascaras", default=None, help="Carpeta de máscaras TIFF")
    parser.add_argument("--ruta-mapeo-paciente", default=None, help="CSV opcional con image_id y patient_id")
    parser.add_argument("--extension-imagen", default=".tiff")
    parser.add_argument("--columna-paciente", default="patient_id")
    parser.add_argument("--columna-etiqueta", default="isup_grade")
    parser.add_argument("--tamano-tile", type=int, default=256)
    parser.add_argument("--salto", type=int, default=0)
    parser.add_argument("--nivel", type=int, default=0)
    parser.add_argument("--proporcion-minima-tejido", type=float, default=0.3)
    parser.add_argument("--umbral-blanco", type=int, default=245)
    parser.add_argument("--maximo-tiles-por-slide", type=int, default=20)
    # Parámetros para probar el DataLoader al final del flujo
    parser.add_argument("--tamano-lote", type=int, default=8)
    parser.add_argument("--num-trabajadores", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """
    funcion principal que ejecuta el flujo 
    genera artefactos de datos, prueba el dataloader y guarda los resultados
    """
    args = construir_argumentos()
    directorio_datos = Path(args.directorio_salida)
    archivos = generar_artefactos_datos(
        ruta_csv=args.ruta_csv,
        directorio_imagenes=args.directorio_imagenes,
        directorio_salida=directorio_datos,
        directorio_mascaras=args.directorio_mascaras,
        ruta_mapeo_paciente=args.ruta_mapeo_paciente,
        extension_imagen=args.extension_imagen,
        columna_paciente=args.columna_paciente,
        columna_etiqueta=args.columna_etiqueta,
        tamano_tile=args.tamano_tile,
        salto=None if args.salto == 0 else args.salto,
        nivel=args.nivel,
        proporcion_minima_tejido=args.proporcion_minima_tejido,
        umbral_blanco=args.umbral_blanco,
        maximo_tiles_por_slide=args.maximo_tiles_por_slide,
    )
    print("Artefactos generados:")
    for nombre, ruta in archivos.items():
        #ciclo para imprimir los archivos generados
        print(f"- {nombre}: {ruta}")

    ruta_manifiesto = archivos["manifiesto"]
    transformacion = construir_transformaciones_entrenamiento(tamano_tile=args.tamano_tile)
    dataloader = construir_dataloader_wsi(
        manifiesto=ruta_manifiesto,
        tamano_lote=args.tamano_lote,
        mezclar=True,
        num_trabajadores=args.num_trabajadores,
        columna_slide="ruta_slide",
        columna_etiqueta=args.columna_etiqueta,
        columna_mascara="ruta_mascara" if args.directorio_mascaras is not None else None,
        tamano_tile=args.tamano_tile,
        nivel=args.nivel,
        transformacion=transformacion,
        devolver_metadatos=True,
    )

    lote = next(iter(dataloader))
    print(f"Primer lote listo. Tamaños: {[getattr(elemento, 'shape', type(elemento)) for elemento in lote]}")


if __name__ == "__main__":
    main()
