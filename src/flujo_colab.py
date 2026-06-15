from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset_wsi import construir_dataloader_wsi, construir_manifiesto_tiles, construir_transformaciones_entrenamiento
from .particion_pacientes import guardar_particiones_paciente


# rutas base del proyecto, por defecto guarda cosas en data/
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = RAIZ_PROYECTO / "data"


def construir_ruta_slide(identificador_imagen: str, directorio_imagenes: str | Path, extension: str = ".tiff") -> str:
    """
    funcion para construir la ruta completa del slide con image_id y extension
    """
    directorio_imagenes = Path(directorio_imagenes)
    return str(directorio_imagenes / f"{identificador_imagen}{extension}")


def preparar_dataframe_base(
    ruta_csv: str | Path,
    directorio_imagenes: str | Path,
    extension_imagen: str = ".tiff",
) -> pd.DataFrame:
    """
    carga el csv principal y le agrega la columna ruta_slide
    """
    df = pd.read_csv(ruta_csv)
    # valida que exista image_id porque sin eso no se puede crear la ruta
    if "image_id" not in df.columns:
        raise ValueError("El CSV  debe tener la columna 'image_id'.")

    df = df.copy()
    # crea la ruta de cada slide 
    df["ruta_slide"] = df["image_id"].apply(lambda image_id: construir_ruta_slide(image_id, directorio_imagenes, extension=extension_imagen))
    return df


def construir_manifiesto_desde_csv(
    ruta_csv: str | Path,
    directorio_imagenes: str | Path,
    directorio_mascaras: str | Path | None,
    directorio_salida: str | Path,
    ruta_mapeo_paciente: str | Path | None,
    extension_imagen: str = ".tiff",
    columna_paciente: str = "patient_id",
    columna_etiqueta: str = "isup_grade",
) -> Path:
    """
    construye el manifiesto de tiles desde un csv y lo guarda
    """
    directorio_salida = Path(directorio_salida)
    # crea carpeta de salida si no existe
    directorio_salida.mkdir(parents=True, exist_ok=True)

    df_base = preparar_dataframe_base(
        ruta_csv=ruta_csv,
        directorio_imagenes=directorio_imagenes,
        extension_imagen=extension_imagen,
    )

    #mapeo de pacientes
    if ruta_mapeo_paciente is not None:
        mapa = pd.read_csv(ruta_mapeo_paciente)
        if {"image_id", columna_paciente}.difference(mapa.columns):
            raise ValueError("El mapeo de pacientes debe tener 'image_id' y la columna de paciente indicada.")
        df_base = df_base.merge(mapa[["image_id", columna_paciente]], on="image_id", how="left", validate="one_to_one")

    # crea el manifiesto recorriendo los slides y guardando solo tiles validos
    manifiesto = construir_manifiesto_tiles(
        df_slides=df_base,
        columna_slide="ruta_slide",
        columna_etiqueta=columna_etiqueta,
        columna_paciente=columna_paciente,
        directorio_mascaras=directorio_mascaras,
        tamano_tile=256,
        salto=256,
        nivel=0,
        proporcion_minima_tejido=0.3,
        umbral_blanco=245,
    )

    # guarda el manifiesto en csv
    ruta_manifiesto = directorio_salida / "manifiesto_tiles.csv"
    manifiesto.to_csv(ruta_manifiesto, index=False)
    return ruta_manifiesto


def construir_dataloader_desde_manifiesto(
    ruta_manifiesto: str | Path,
    tamano_lote: int = 8,
    mezclar: bool = True,
    num_trabajadores: int = 2,
    tamano_tile: int = 256,
):
    """
    crea un dataloader desde el manifiesto.
    """
    # crea augmentations para entrenamiento
    transformacion = construir_transformaciones_entrenamiento(tamano_tile=tamano_tile)
    # devuelve el dataloader listo
    return construir_dataloader_wsi(
        manifiesto=ruta_manifiesto,
        tamano_lote=tamano_lote,
        mezclar=mezclar,
        num_trabajadores=num_trabajadores,
        pin_memory=True,
        columna_slide="ruta_slide",
        columna_etiqueta="isup_grade",
        tamano_tile=tamano_tile,
        nivel=0,
        transformacion=transformacion,
        devolver_metadatos=True,
    )


def construir_argumentos() -> argparse.Namespace:
    """
    define argumentos para ejecutar este flujo por cli.
    """
    parser = argparse.ArgumentParser(description="Flujo de Colab para tiles histopatológicos.")
    parser.add_argument("--ruta-csv", required=True)
    parser.add_argument("--directorio-imagenes", required=True)
    parser.add_argument("--directorio-mascaras", default=None)
    parser.add_argument("--directorio-salida", default=str(DIRECTORIO_DATOS))
    parser.add_argument("--ruta-mapeo-paciente", default=None)
    parser.add_argument("--extension-imagen", default=".tiff")
    parser.add_argument("--columna-paciente", default="patient_id")
    parser.add_argument("--columna-etiqueta", default="isup_grade")
    parser.add_argument("--tamano-lote", type=int, default=8)
    parser.add_argument("--num-trabajadores", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """
    ejecuta todo el flujo: splits opcionales, manifiesto y prueba del dataloader.
    """
    args = construir_argumentos()
    directorio_datos = Path(args.directorio_salida)

    # si hay mapeo de pacientes, genera los splits
    if args.ruta_mapeo_paciente is not None:
        guardar_particiones_paciente(
            ruta_csv=args.ruta_csv,
            directorio_salida=directorio_datos / "splits",
            columna_paciente=args.columna_paciente,
            columna_imagen="image_id",
            columna_etiqueta=args.columna_etiqueta,
            ruta_mapeo_paciente=args.ruta_mapeo_paciente,
        )

    # construye el manifiesto y lo guarda en manifests/
    ruta_manifiesto = construir_manifiesto_desde_csv(
        ruta_csv=args.ruta_csv,
        directorio_imagenes=args.directorio_imagenes,
        directorio_mascaras=args.directorio_mascaras,
        directorio_salida=directorio_datos / "manifests",
        ruta_mapeo_paciente=args.ruta_mapeo_paciente,
        extension_imagen=args.extension_imagen,
        columna_paciente=args.columna_paciente,
        columna_etiqueta=args.columna_etiqueta,
    )

    # crea dataloader para validar que todo salio bien
    dataloader = construir_dataloader_desde_manifiesto(
        ruta_manifiesto=ruta_manifiesto,
        tamano_lote=args.tamano_lote,
        num_trabajadores=args.num_trabajadores,
    )

    # prueba rapida: toma un batch e imprime las formas
    lote = next(iter(dataloader))
    imagenes = lote[0]
    etiquetas = lote[1]
    print(f"Dataloader listo. Imágenes: {tuple(imagenes.shape)}. Etiquetas: {tuple(etiquetas.shape)}.")


if __name__ == "__main__":
    main()
