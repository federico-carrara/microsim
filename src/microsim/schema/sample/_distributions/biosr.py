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
}

class BioSR(_BaseDistribution): # FromDir --> take away bioSR related stuff
    root_dir: str
    label: Literal["CCPs", "ER", "F-actin", "Microtubules"] # str
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
            "GT_all.mrc" if self.label != "F-actin" else f"GT_all_a.mrc"
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
    
    def _crop_to_shape(self, data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        pass   
        
    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        data = self._load_data()
        if space.shape != data.shape:
            # TODO: implement cropping to shape
            raise ValueError(
                "This GroundTruth may only be used with simulation space of shape: "
                f"{data.shape}. Got: {space.shape}"
            )
        data = self._map_to_fp_distrib(data)        
        return space + xp.asarray(data).astype(space.dtype)