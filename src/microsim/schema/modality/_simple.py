from typing import Annotated, Any, Literal

import pint
from annotated_types import Ge
from pint import Quantity
from tqdm import tqdm

from microsim._data_array import ArrayProtocol, DataArray, xrDataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.dimensions import Axis
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings
from microsim.schema.space import SpaceProtocol


class _PSFModality(SimBaseModel):
    def psf(
        self,
        space: SpaceProtocol,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        em_wvl: Quantity | None = None,
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
            em_wvl=em_wvl,
        )

    def render(
        self,
        truth: xrDataArray,
        channel: OpticalConfig,
        retain_spectrum: bool,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> xrDataArray:
        # convolved = xp.zeros_like(truth.data)
        convolved: Any = 0
        ureg = pint.application_registry.get()  # type: ignore
        for fluor_idx in range(truth.sizes[Axis.F]):
            print(f"Obtaining optical image for fluorophore {fluor_idx}...")
            # convolved_fluor = xp.zeros_like(truth.data)
            convolved_fluor: Any = 0
            for bin_idx in tqdm(range(truth.sizes[Axis.W]), desc="Convolving PSF over spectral bands"):
                binned_flux = truth.isel({Axis.W: bin_idx, Axis.F: fluor_idx})
                if xp.isnan(xp.sum(binned_flux.data)):
                    # NOTE: there can be bins for which there is no data in one of the
                    #  fluorophores
                    print(f"Skipping bin {bin_idx} for fluorophore {fluor_idx}")
                    continue
                em_wvl = binned_flux[Axis.W].values.item().mid * ureg.nm
                psf = self.psf(
                    truth.attrs["space"],
                    channel,
                    objective_lens,
                    settings,
                    xp,
                    em_wvl=em_wvl,
                )
                curr_convolved = xp.fftconvolve(
                    binned_flux.isel({Axis.C: 0}), psf, mode="same"
                ) # shape (Z, Y, X)
                if retain_spectrum:
                    curr_convolved = xp.expand_dims(curr_convolved, axis=0)
                    if isinstance(convolved_fluor, int):
                        convolved_fluor = curr_convolved
                    else:
                        convolved_fluor = xp.concatenate(
                            [convolved_fluor, curr_convolved], axis=0
                        ) # shape (W, Z, Y, X) 
                else:
                    convolved_fluor += curr_convolved # shape (Z, Y, X)
            # Get spectrum for this fluorophore
            if retain_spectrum:
                convolved_fluor = xp.expand_dims(convolved_fluor, axis=1)
                if isinstance(convolved, int):
                    convolved = convolved_fluor
                else:
                    convolved = xp.concatenate(
                        [convolved, convolved_fluor], axis=1
                    ) # shape (W, F, Z, Y, X) 
            else: 
                convolved += convolved_fluor # shape (Z, Y, X)
        if retain_spectrum:
            convolved = convolved[:, xp.newaxis, ...]
            out = DataArray(
                convolved,
                dims=[Axis.W, Axis.C, Axis.F, Axis.Z, Axis.Y, Axis.X],
                coords={
                    Axis.W: truth.coords[Axis.W],
                    Axis.C: [channel],
                    Axis.F: truth.coords[Axis.F],
                    Axis.Z: truth.coords[Axis.Z],
                    Axis.Y: truth.coords[Axis.Y],
                    Axis.X: truth.coords[Axis.X],
                },
                attrs=truth.attrs,
            )
        else:
            out = DataArray(
                convolved[None],
                dims=[Axis.C, Axis.Z, Axis.Y, Axis.X],
                coords={
                    Axis.C: [channel],
                    Axis.Z: truth.coords[Axis.Z],
                    Axis.Y: truth.coords[Axis.Y],
                    Axis.X: truth.coords[Axis.X],
                },
                attrs=truth.attrs,
            )
        return out
        

class Confocal(_PSFModality):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def psf(
        self,
        space: SpaceProtocol,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        em_wvl: Quantity | None = None,
    ) -> ArrayProtocol:
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
            em_wvl=em_wvl,
        )


class Widefield(_PSFModality):
    type: Literal["widefield"] = "widefield"
