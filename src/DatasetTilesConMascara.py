from torch.utils.data import Dataset
import torch
import openslide
import numpy as np
import pandas as pd
from PIL import Image

class DatasetTilesConMascara(Dataset):
    def __init__(self, df, transform=None):
        #guardar el dataframe y definir las albumentaciones
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        #cantidad de tiles
        return len(self.df)

    def __getitem__(self, idx):
        fila = self.df.iloc[idx]
        ruta_slide = fila["ruta_slide"]
        ruta_mascara = fila["ruta_mascara"]
        x, y = int(fila["x"]), int(fila["y"])
        tamano_tile = int(fila["tamano_tile"])

        #cargar solo el espacio de x y y de la iamgen original
        slide = openslide.OpenSlide(ruta_slide)
        imagen = slide.read_region((x, y), 0, (tamano_tile, tamano_tile)).convert("RGB")
        imagen = np.array(imagen)
        slide.close()

        #cargar el tile de la mascarna
        Image.MAX_IMAGE_PIXELS = None
        mascara = np.array(Image.open(ruta_mascara))
        if mascara.ndim == 3: # convertir si es rgb
            mascara = mascara[..., 0]

        mascara = mascara[y:y + tamano_tile, x:x + tamano_tile]

        #aplicar transformaciones
        if self.transform is not None:
            salida = self.transform(image=imagen, mask=mascara)
            imagen = salida["image"]
            mascara = salida["mask"]
        else:
            #normalizar y permutar a pythorch C, H, W
            imagen = torch.tensor(imagen).permute(2, 0, 1).float() / 255.0
            mascara = torch.tensor(mascara).long()

        return imagen, mascara, fila["image_id"]