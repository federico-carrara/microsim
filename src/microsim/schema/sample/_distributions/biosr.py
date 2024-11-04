import os
from typing import Literal

import numpy as np
from pydantic import model_validator

from microsim._data_array import ArrayProtocol, xrDataArray
from microsim.util import _read_mrc
from microsim._data_array import xrDataArray
from microsim.schema.backend import NumpyAPI
from microsim.schema.sample._distributions._base import _BaseDistribution

NIMGS = {
    "CCPs": 54,
    "F-actin": 51,
    "ER": 68,
    "Microtubules": 55,
    "F-actin_Nonlinear": 51,
}

class BioSR(_BaseDistribution): # FromDir --> take away bioSR related stuff
    root_dir: str
    label: Literal["CCPs", "ER", "F-actin", "Microtubules", "F-actin_Nonlinear"] # str
    img_idx: int | None = None
    # imreader: Callable
    
    @model_validator(mode="after")
    def _validate_img_idx(cls, value):
        if value.img_idx is not None:
            n_imgs = NIMGS[value.label]
            if value.img_idx < 0 or value.img_idx >= n_imgs:
                raise ValueError(
                    f"Invalid idx: {value.img_idx}. Must be in range [0, {n_imgs})."
                )
        return value
    
    @property
    def _idx(self):
        """Return the index of the image to load.
        
        If a specific index is not set through `self.img_idx`, a random index is
        returned.
        """
        if self.img_idx is None:
            n_imgs = NIMGS[self.label]
            return np.random.randint(n_imgs)
        else:
            return self.img_idx
        
    @classmethod
    def is_random(cls) -> bool:
        """Return True if this distribution generates randomized results."""
        return True
    
    def _load_data(self) -> np.ndarray:
        fname = os.path.join(
            self.label, 
            "GT_all.mrc" if "F-actin" not in self.label else f"GT_all_a.mrc"
        )
        fpath = os.path.join(self.root_dir, fname)
        imgs = _read_mrc(fpath) # shape: (X, Y, N)
        img = imgs[:, :, self._idx]
        return img[np.newaxis, ...]
    
    def _map_to_fp_distrib(
        self, 
        data: np.ndarray, 
        threshold: float = 0.1,
        gamma: float = 2.0
    ) -> np.ndarray:
        """Map data from 16bit img to flurophore distribution.
        
        Steps:
        1. Normalize data to [0, 1]
        2. Apply a threshold to get rid of background noise
        3. Apply gamma correction (exp) to enhance contrast (hist equaliz??)
        4. Map back to 8bit (0-255)
        
        NOTE: threshold is expressed in relative terms (0-1)
        """
        # data = np.power(data, gamma)
        data = (data - data.min()) / (data.max() - data.min())
        # data[data < threshold] = 0
        data = (data * 255).astype(np.uint8)
        return data
    
    @staticmethod
    def _crop_to_shape(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Center-crop data to a specific shape.
        
        Parameters
        ----------
        data : np.ndarray
            Data to crop. Shape: (1, H, W)
        shape : tuple[int, int]
            Target shape.
        
        Returns
        -------
        np.ndarray
            Cropped data.
        """
        h, w = shape
        _, h0, w0 = data.shape
        h0, w0 = h0 // 2, w0 // 2
        h0, w0 = h0 - h // 2, w0 - w // 2
        return data[:, h0:h0+h, w0:w0+w]
        
    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        data = self._load_data()
        if space.shape != data.shape:
            data = BioSR._crop_to_shape(data, space.shape[-2:])
        data = self._map_to_fp_distrib(data)        
        return space + xp.asarray(data).astype(space.dtype)