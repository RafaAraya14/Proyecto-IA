from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = RAIZ_PROYECTO / "data"


def validar_proporciones(proporcion_entrenamiento: float, proporcion_validacion: float, proporcion_prueba: float) -> None:
    """
    funcion para validar que las proporciones sumen 1
    """
    total = proporcion_entrenamiento + proporcion_validacion + proporcion_prueba
    if not np.isclose(total, 1.0):
        raise ValueError(f"Las proporciones de partición deben sumar 1.0, pero suman {total:.4f}.")
    if min(proporcion_entrenamiento, proporcion_validacion, proporcion_prueba) < 0:
        raise ValueError("Las proporciones de partición no pueden ser negativas.")


def obtener_moda_o_primero(valores: pd.Series):
    """
    funcion para obtener la moda o el primer elemento de la serie de pandas
    """
    modas = valores.mode(dropna=True)
    if not modas.empty:
        return modas.iloc[0]
    return valores.iloc[0]


def repartir_cantidades(tamano: int, proporciones: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    funcion para repartir las cantidades de los pacientes en cada particion 
    """
    valores_brutos = np.asarray(proporciones, dtype=float) * tamano
    cantidades = np.floor(valores_brutos).astype(int)
    resto = int(tamano - cantidades.sum())
    if resto > 0: 
        orden = np.argsort(-(valores_brutos - cantidades))
        for indice in range(resto):
            cantidades[orden[indice % len(orden)]] += 1
    return int(cantidades[0]), int(cantidades[1]), int(cantidades[2])


def construir_tabla_particion_paciente(
    df: pd.DataFrame,
    columna_paciente: str = "patient_id",
    columna_imagen: str = "image_id",
    columna_etiqueta: str = "isup_grade",
    ruta_mapeo_paciente: str | Path | None = None,
    proporcion_entrenamiento: float = 0.7,
    proporcion_validacion: float = 0.15,
    proporcion_prueba: float = 0.15,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    funcion para construir la tabla de particion por paciente
    """
    validar_proporciones(proporcion_entrenamiento, proporcion_validacion, proporcion_prueba)

    
    if columna_paciente not in df.columns:#si no hay columna de paciente, se hace mapeo con el csv
        if ruta_mapeo_paciente is None:
            raise ValueError(
                f"No existe la columna '{columna_paciente}' en el CSV principal. "
                f"Debes pasar una tabla auxiliar con el mapeo de pacientes."
            )

        if columna_imagen not in df.columns:
            raise ValueError(f"Falta la columna de imagen '{columna_imagen}' en el CSV principal.")

        #lee el csv de la ruta de mapeo del paciente
        mapeo_paciente = pd.read_csv(ruta_mapeo_paciente)
        columnas_mapeo = {columna_imagen, columna_paciente}
        faltantes_mapeo = columnas_mapeo.difference(mapeo_paciente.columns)
        if faltantes_mapeo:
            raise ValueError(f"Faltan columnas en el mapeo de pacientes: {sorted(faltantes_mapeo)}")

        # hace un merge con el dataframe original 
        df = df.merge(mapeo_paciente[[columna_imagen, columna_paciente]], on=columna_imagen, how="left", validate="one_to_one")

    # valida las columnas necesarias
    columnas_requeridas = {columna_paciente, columna_etiqueta}
    faltantes = columnas_requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")

    # construir el resumen del paciente
    resumen_pacientes = (
        df[[columna_paciente, columna_etiqueta]]
        .dropna(subset=[columna_paciente])
        .groupby(columna_paciente, as_index=False)[columna_etiqueta]
        .agg(obtener_moda_o_primero)
    )

    generador = np.random.default_rng(semilla)
    asignaciones = []

    for _, grupo in resumen_pacientes.groupby(columna_etiqueta, dropna=False):
        #extraer los pacientes del grupo actual 
        pacientes = grupo[columna_paciente].to_numpy(copy=True)
        generador.shuffle(pacientes)

        cantidad_entrenamiento, cantidad_validacion, cantidad_prueba = repartir_cantidades(
            len(pacientes),
            (proporcion_entrenamiento, proporcion_validacion, proporcion_prueba),
        )

        # asignar los pacientes a cada particion 
        pacientes_entrenamiento = pacientes[:cantidad_entrenamiento]
        pacientes_validacion = pacientes[cantidad_entrenamiento : cantidad_entrenamiento + cantidad_validacion]
        pacientes_prueba = pacientes[cantidad_entrenamiento + cantidad_validacion : cantidad_entrenamiento + cantidad_validacion + cantidad_prueba]

        asignaciones.append(pd.DataFrame({columna_paciente: pacientes_entrenamiento, "particion": "entrenamiento"}))
        asignaciones.append(pd.DataFrame({columna_paciente: pacientes_validacion, "particion": "validacion"}))
        asignaciones.append(pd.DataFrame({columna_paciente: pacientes_prueba, "particion": "prueba"}))

    tabla_particion = pd.concat(asignaciones, ignore_index=True).drop_duplicates(subset=[columna_paciente])
    return tabla_particion


def dividir_dataframe_por_paciente(
    df: pd.DataFrame,
    columna_paciente: str = "patient_id",
    columna_imagen: str = "image_id",
    columna_etiqueta: str = "isup_grade",
    ruta_mapeo_paciente: str | Path | None = None,
    proporcion_entrenamiento: float = 0.7,
    proporcion_validacion: float = 0.15,
    proporcion_prueba: float = 0.15,
    semilla: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    funcion para dividir el dataframe por pacientes y devolver el df completo con la particion 
    """
    tabla_particion = construir_tabla_particion_paciente(
        df=df,
        columna_paciente=columna_paciente,
        columna_imagen=columna_imagen,
        columna_etiqueta=columna_etiqueta,
        ruta_mapeo_paciente=ruta_mapeo_paciente,
        proporcion_entrenamiento=proporcion_entrenamiento,
        proporcion_validacion=proporcion_validacion,
        proporcion_prueba=proporcion_prueba,
        semilla=semilla,
    )

    combinado = df.merge(tabla_particion, on=columna_paciente, how="left", validate="many_to_one")

    if combinado["particion"].isna().any():
        pacientes_sin_particion = combinado.loc[combinado["particion"].isna(), columna_paciente].dropna().unique().tolist()
        raise ValueError(f"Hay pacientes sin partición asignada: {pacientes_sin_particion[:10]}")

    entrenamiento = combinado.loc[combinado["particion"] == "entrenamiento"].drop(columns=["particion"])
    validacion = combinado.loc[combinado["particion"] == "validacion"].drop(columns=["particion"])
    prueba = combinado.loc[combinado["particion"] == "prueba"].drop(columns=["particion"])
    return combinado, entrenamiento, validacion, prueba


def guardar_particiones_paciente(
    ruta_csv: str | Path,
    directorio_salida: str | Path,
    columna_paciente: str = "patient_id",
    columna_imagen: str = "image_id",
    columna_etiqueta: str = "isup_grade",
    ruta_mapeo_paciente: str | Path | None = None,
    proporcion_entrenamiento: float = 0.7,
    proporcion_validacion: float = 0.15,
    proporcion_prueba: float = 0.15,
    semilla: int = 42,
) -> dict[str, Path]:
    
    """funcoon que genera las particiones por los pacientes y los gaurda en el csv"""

    ruta_csv = Path(ruta_csv)
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)
    # Leer CSV de entrada
    df = pd.read_csv(ruta_csv)

    combinado, entrenamiento, validacion, prueba = dividir_dataframe_por_paciente(
        df=df,
        columna_paciente=columna_paciente,
        columna_imagen=columna_imagen,
        columna_etiqueta=columna_etiqueta,
        ruta_mapeo_paciente=ruta_mapeo_paciente,
        proporcion_entrenamiento=proporcion_entrenamiento,
        proporcion_validacion=proporcion_validacion,
        proporcion_prueba=proporcion_prueba,
        semilla=semilla,
    )

    # rutas de salida 
    archivos_salida = {
        "todo_con_particion": directorio_salida / f"{ruta_csv.stem}_con_particion.csv",
        "entrenamiento": directorio_salida / f"{ruta_csv.stem}_entrenamiento.csv",
        "validacion": directorio_salida / f"{ruta_csv.stem}_validacion.csv",
        "prueba": directorio_salida / f"{ruta_csv.stem}_prueba.csv",
    }

    #escribir los archivos de salida
    combinado.to_csv(archivos_salida["todo_con_particion"], index=False)
    entrenamiento.to_csv(archivos_salida["entrenamiento"], index=False)
    validacion.to_csv(archivos_salida["validacion"], index=False)
    prueba.to_csv(archivos_salida["prueba"], index=False)
    return archivos_salida


def construir_argumentos() -> argparse.Namespace:

    """funcion para construir argumentos"""
    parser = argparse.ArgumentParser(description="Crear particiones seguras por paciente.")
    parser.add_argument("--ruta-csv", required=True, help="Ruta de train.csv")
    parser.add_argument(
        "--directorio-salida",
        default=str(DIRECTORIO_DATOS / "splits"),
        help="Directorio donde se guardarán los CSV particionados",
    )
    parser.add_argument("--columna-paciente", default="patient_id")
    parser.add_argument("--columna-imagen", default="image_id")
    parser.add_argument("--columna-etiqueta", default="isup_grade")
    parser.add_argument("--ruta-mapeo-paciente", default=None, help="Ruta opcional al CSV con image_id y patient_id")
    parser.add_argument("--proporcion-entrenamiento", type=float, default=0.7)
    parser.add_argument("--proporcion-validacion", type=float, default=0.15)
    parser.add_argument("--proporcion-prueba", type=float, default=0.15)
    parser.add_argument("--semilla", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = construir_argumentos()
    """
    funcion principal que ejecuta ej flujo de particion por cada clientel, generando los csv de cada particion y uno completo con la particion asignana
    
    """
    archivos = guardar_particiones_paciente(
        ruta_csv=args.ruta_csv,
        directorio_salida=args.directorio_salida,
        columna_paciente=args.columna_paciente,
        columna_imagen=args.columna_imagen,
        columna_etiqueta=args.columna_etiqueta,
        ruta_mapeo_paciente=args.ruta_mapeo_paciente,
        proporcion_entrenamiento=args.proporcion_entrenamiento,
        proporcion_validacion=args.proporcion_validacion,
        proporcion_prueba=args.proporcion_prueba,
        semilla=args.semilla,
    )
    for nombre, ruta in archivos.items():
        print(f"{nombre}: {ruta}")


if __name__ == "__main__":
    main()
