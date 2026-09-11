from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from microsim.schema.optical_config.config import LightSource, OpticalConfig
from microsim.schema.optical_config.filter import (
    Bandpass,
    Longpass,
    Placement,
    SpectrumFilter,
    bandpass,
)
from microsim.schema.spectrum import Spectrum

# https://www.chroma.com/products/sets/49000-et-dapi
DAPI = OpticalConfig(
    name="DAPI",
    filters=[
        Bandpass(bandcenter=350, bandwidth=50, placement="EX"),
        Longpass(cuton=400, placement="BS"),
        Bandpass(bandcenter=460, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49001-et-ecfp
ECFP = OpticalConfig(
    name="ECFP",
    filters=[
        Bandpass(bandcenter=436, bandwidth=20, placement="EX"),
        Longpass(cuton=455, placement="BS"),
        Bandpass(bandcenter=480, bandwidth=40, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49002-et-egfp-fitc-cy2
FITC = OpticalConfig(
    name="FITC",
    filters=[
        Bandpass(bandcenter=470, bandwidth=40, placement="EX"),
        Longpass(cuton=495, placement="BS"),
        Bandpass(bandcenter=525, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49003-et-eyfp
EYFP = OpticalConfig(
    name="EYFP",
    filters=[
        Bandpass(bandcenter=500, bandwidth=20, placement="EX"),
        Longpass(cuton=515, placement="BS"),
        Bandpass(bandcenter=535, bandwidth=30, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49004-et-cy3-tritc
TRITC = OpticalConfig(
    name="TRITC",
    filters=[
        Bandpass(bandcenter=545, bandwidth=25, placement="EX"),
        Longpass(cuton=565, placement="BS"),
        Bandpass(bandcenter=605, bandwidth=70, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49005-et-dsred-tritc-cy3
DSRED = OpticalConfig(
    name="DSRED",
    filters=[
        Bandpass(bandcenter=545, bandwidth=30, placement="EX"),
        Longpass(cuton=570, placement="BS"),
        Bandpass(bandcenter=620, bandwidth=60, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49006-et-cy5
CY5 = OpticalConfig(
    name="CY5",
    filters=[
        Bandpass(bandcenter=620, bandwidth=60, placement="EX"),
        Longpass(cuton=660, placement="BS"),
        Bandpass(bandcenter=700, bandwidth=75, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49007-et-cy7
CY7 = OpticalConfig(
    name="CY7",
    filters=[
        Bandpass(bandcenter=710, bandwidth=75, placement="EX"),
        Longpass(cuton=760, placement="BS"),
        Bandpass(bandcenter=810, bandwidth=90, placement="EM"),
    ],
)


# This is just a rough example... needs to be refined
def spectral_detector(
    bins: int,
    min_wave: float,
    max_wave: float,
    lasers: Sequence[int],
    laser_powers: Sequence[float],
    bp_bandwidth: float = 10,
) -> list[OpticalConfig]:
    """Create a spectral detector with a given number of bins and lasers.

    Parameters
    ----------
    bins : int
        Number of bins to use.
    min_wave : float
        Minimum wavelength to consider.
    max_wave : float
        Maximum wavelength to consider.
    lasers : Sequence[int]
        List of lasers to use.
    laser_powers: Sequence[float]
        List of laser powers.
    bp_bandwidth : float, optional
        Bandwidth of the bandpass filter, by default 10.

    Returns
    -------
    list[ms.OpticalConfig]
        List of optical configurations.

    Examples
    --------
    Create a spectral detector with 8 bins, wavelengths between 480 and 600 nm, and
    lasers at 488, 515, and 561 nm.

    >>> spectral_detector(8, 480, 600, [488, 515, 561])
    """
    lights = [
        LightSource.laser(laser, power) for laser, power in zip(lasers, laser_powers)
    ]
    waves = np.arange(min(lasers) - 50, max_wave + 50, 1)

    # create fake bandpass ... could also be something like 80/20
    laser0, *rest = lasers
    bp = bandpass(waves, center=laser0, bandwidth=bp_bandwidth, transmission=1)
    for laser in rest:
        bp = bp + bandpass(waves, center=laser, bandwidth=bp_bandwidth)
    bp = SpectrumFilter(
        transmission=Spectrum(wavelength=waves, intensity=1 - bp),
        placement=Placement.BS,
    )

    configs: list[OpticalConfig] = []
    edges = np.linspace(min_wave, max_wave, bins + 1)
    for i, (low, high) in enumerate(pairwise(edges)):
        mask = (waves >= low) & (waves < high)
        f = SpectrumFilter(
            transmission=Spectrum(wavelength=waves, intensity=mask),
            placement=Placement.EM_PATH,
        )
        oc = OpticalConfig(
            name=f"Channel {i}",
            lights=lights,
            filters=[bp, f],
        )
        configs.append(oc)
    return configs


def leica_stellaris_detector(
    bands: Sequence[tuple[float, float]],
    lasers: Sequence[float],
    laser_powers: Sequence[float],
    notch_width: float = 10,
) -> list[OpticalConfig]:
    """Create the optical configs of a Leica-like confocal (e.g. SP8/Stellaris).

    Each detection band iss one optical config.
    All bands are acquired simultaneously, so every band sees every laser.
    A notch around each laser line keeps laser light out of the emission path.
    This generalises `spectral_detector` to explicit, arbitrary band edges.

    Parameters
    ----------
    bands : Sequence[tuple[float, float]]
        Detection bands as ``(start, end)`` wavelengths in nm, with ``start``
        included and ``end`` excluded. Bands must be in ascending order and must not
        overlap. Adjacent bands, where one band's ``end`` equals the next band's
        ``start``, are allowed.
    lasers : Sequence[float]
        Laser wavelengths in nm.
    laser_powers : Sequence[float]
        Power of each laser in W/cm^2, in the same order as `lasers`.
    notch_width : float, optional
        Width in nm of the notch around each laser line, by default 10.

    Returns
    -------
    list[OpticalConfig]
        One optical config per band, in the order of `bands`. Each config has all
        the lasers, the notch filter (shared by all configs) and the band's emission
        filter.

    Raises
    ------
    ValueError
        If `bands` or `lasers` is empty, if `lasers` and `laser_powers` differ in
        length, if a band has ``start >= end``, or if the bands are not in ascending
        order or overlap.

    Notes
    -----
    `OpticalConfig.irradiance` normalises the summed laser spectrum to an area of 1
    and then multiplies it by `OpticalConfig.power`. Each config's power is set to
    ``sum(laser_powers)``, so each laser delivers its own power in W/cm^2.

    Examples
    --------
    Three bands, excited by lasers at 488, 561 and 638 nm.

    >>> configs = leica_optical_configs(
    ...     bands=[(500, 550), (570, 620), (650, 750)],
    ...     lasers=[488, 561, 638],
    ...     laser_powers=[1, 1, 1],
    ... )
    >>> [oc.name for oc in configs]
    ['Band 0 [500-550 nm]', 'Band 1 [570-620 nm]', 'Band 2 [650-750 nm]']
    """
    if len(bands) == 0:
        raise ValueError("`bands` must contain at least one band.")
    if len(lasers) == 0:
        raise ValueError("`lasers` must contain at least one laser.")
    if len(lasers) != len(laser_powers):
        raise ValueError(
            "`lasers` and `laser_powers` must have the same length, got "
            f"{len(lasers)} and {len(laser_powers)}."
        )
    for lo, hi in bands:
        if lo >= hi:
            raise ValueError(f"Band start must be smaller than end, got ({lo}, {hi}).")
    for (prev_lo, prev_hi), (lo, hi) in pairwise(bands):
        if lo < prev_lo:
            raise ValueError(
                "`bands` must be in ascending order, got "
                f"({prev_lo}, {prev_hi}) before ({lo}, {hi})."
            )
        if lo < prev_hi:
            raise ValueError(
                f"Bands must not overlap, got ({prev_lo}, {prev_hi}) and ({lo}, {hi})."
            )

    waves = np.arange(
        min(min(lasers), bands[0][0]) - 50, max(max(lasers), bands[-1][1]) + 50, 1
    )
    lights = [
        LightSource.laser(wl, p) for wl, p in zip(lasers, laser_powers, strict=True)
    ]

    # AOBS-like notch around every laser line. The product (rather than the sum used
    # in `spectral_detector`) keeps transmission in [0, 1] when notches overlap.
    notch = SpectrumFilter(
        transmission=Spectrum(
            wavelength=waves,
            intensity=np.prod(
                [
                    1 - bandpass(waves, center=wl, bandwidth=notch_width)
                    for wl in lasers
                ],
                axis=0,
            ),
        ),
        placement=Placement.BS,
    )

    configs: list[OpticalConfig] = []
    for i, (lo, hi) in enumerate(bands):
        band_filter = SpectrumFilter(
            transmission=Spectrum(
                wavelength=waves,
                intensity=((waves >= lo) & (waves < hi)).astype(float),
            ),
            placement=Placement.EM_PATH,
        )
        configs.append(
            OpticalConfig(
                name=f"Band {i} [{lo:g}-{hi:g} nm]",
                lights=lights,
                filters=[notch, band_filter],
                power=sum(laser_powers),
            )
        )
    return configs
