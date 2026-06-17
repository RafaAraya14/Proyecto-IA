# Cargar el manifiesto que acabas de guardar
from torch.utils.data import Dataset
import torch
import openslide
import numpy as np


class DatasetTiles(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fila = self.df.iloc[idx]

        slide = openslide.OpenSlide(fila["ruta_slide"])
        x = int(fila["x"])
        y = int(fila["y"])
        tamano_tile = int(fila["tamano_tile"])

        imagen = slide.read_region((x, y), 0, (tamano_tile, tamano_tile)).convert("RGB")
        imagen = np.array(imagen)
        slide.close()

        # Tensor CHW normalizado a [0,1]
        imagen = torch.tensor(imagen).permute(2, 0, 1).float() / 255.0

        return imagen, fila["image_id"], x, y

