from collections.abc import Sequence

import numpy as np
import pytest

import microsim.schema as ms
from microsim.schema.optical_config.filter import Placement
from microsim.schema.optical_config.lib import (
    leica_stellaris_detector,
    spectral_detector,
)

WAVES = np.arange(300, 900, 1.0)
# band 1 is adjacent to band 0 and contains the 561 nm laser
BANDS = [(500, 550), (550, 600), (650, 700)]
LASERS = [488, 561, 638]
POWERS = [1.0, 2.0, 3.0]
NOTCH_WIDTH = 10


def _gaussian(center: float, sigma: float) -> ms.Spectrum:
    return ms.Spectrum(
        wavelength=WAVES,
        intensity=np.exp(-((WAVES - center) ** 2) / (2 * sigma**2)),
    )


# excited only by the 488 nm laser, emits in band 0
GREEN = ms.Fluorophore(
    name="green",
    excitation_spectrum=_gaussian(488, 8),
    emission_spectrum=_gaussian(525, 12),
    extinction_coefficient=50_000,
    quantum_yield=0.8,
)


def test_one_config_per_band() -> None:
    configs = leica_stellaris_detector(BANDS, LASERS, POWERS)

    assert [oc.name for oc in configs] == [
        "Band 0 [500-550 nm]",
        "Band 1 [550-600 nm]",
        "Band 2 [650-700 nm]",
    ]
    for oc in configs:
        assert [light.spectrum.peak_wavelength for light in oc.lights] == LASERS
        assert [light.power for light in oc.lights] == POWERS
        assert oc.power == sum(POWERS)


def test_emission_transmission() -> None:
    configs = leica_stellaris_detector(BANDS, LASERS, POWERS, notch_width=NOTCH_WIDTH)

    for oc, (lo, hi) in zip(configs, BANDS, strict=True):
        assert oc.emission is not None
        spectrum = oc.emission.spectrum
        waves = np.asarray(spectrum.wavelength)
        transmission = np.asarray(spectrum.intensity)

        inside = (waves >= lo) & (waves < hi)
        dist_to_laser = np.min(np.abs(waves[:, None] - np.array(LASERS)), axis=1)
        far_from_lasers = dist_to_laser > NOTCH_WIDTH

        assert np.all(transmission[~inside] == 0)
        assert np.any(inside & far_from_lasers)
        np.testing.assert_allclose(transmission[inside & far_from_lasers], 1)

    # the 561 nm laser falls inside band 1 and is notched out
    spectrum = configs[1].emission.spectrum  # type: ignore[union-attr]
    at_laser = np.asarray(spectrum.intensity)[np.asarray(spectrum.wavelength) == 561]
    assert at_laser.size == 1
    assert at_laser.item() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    ("bands", "lasers", "laser_powers"),
    [
        ([], LASERS, POWERS),
        (BANDS, [], []),
        (BANDS, [488, 561], [1.0]),
        ([(550, 500)], LASERS, POWERS),
        ([(500, 500)], LASERS, POWERS),
        ([(600, 650), (500, 550)], LASERS, POWERS),
        ([(500, 560), (550, 600)], LASERS, POWERS),
    ],
    ids=[
        "empty-bands",
        "empty-lasers",
        "lasers-powers-mismatch",
        "start-after-end",
        "start-equals-end",
        "not-ascending",
        "overlapping",
    ],
)
def test_invalid_inputs(
    bands: Sequence[tuple[float, float]],
    lasers: Sequence[float],
    laser_powers: Sequence[float],
) -> None:
    with pytest.raises(ValueError):
        leica_stellaris_detector(bands, lasers, laser_powers)


def _green_emission(laser_powers: Sequence[float]) -> float:
    oc = leica_stellaris_detector(BANDS, LASERS, laser_powers)[0]
    return float(oc.filtered_emission_rate(GREEN).sum())


def test_emission_scales_only_with_exciting_laser() -> None:
    base = _green_emission([1.0, 2.0, 3.0])
    assert base > 0

    # raising a laser that doesn't excite the fluorophore changes nothing
    assert _green_emission([1.0, 20.0, 3.0]) == pytest.approx(base, rel=1e-6)
    # doubling the exciting laser doubles the emission
    assert _green_emission([2.0, 2.0, 3.0]) == pytest.approx(2 * base, rel=1e-6)


def _band_filter(oc: ms.OpticalConfig) -> np.ndarray:
    (f,) = [f for f in oc.filters if f.placement == Placement.EM_PATH]
    return np.asarray(f.transmission.intensity)  # type: ignore[union-attr]


def test_spectral_detector_bins_do_not_overlap() -> None:
    # 15 nm bins: every bin edge falls on the 1 nm wavelength grid
    configs = spectral_detector(8, 480, 600, [488])
    coverage = np.sum([_band_filter(oc) for oc in configs], axis=0)
    waves = np.asarray(configs[0].filters[0].transmission.wavelength)  # type: ignore[union-attr]

    assert coverage.max() == 1  # an edge wavelength is not detected twice
    np.testing.assert_array_equal(coverage[(waves >= 480) & (waves < 600)], 1)


def test_spectral_detector_laser_outside_bins() -> None:
    # 405 nm is more than 100 nm below the first bin
    oc = spectral_detector(4, 560, 700, [405, 561])[0]
    assert oc.illumination is not None
    illum = oc.illumination.as_xarray()

    at_405 = float(illum.sel(w=slice(400, 410)).sum())
    at_561 = float(illum.sel(w=slice(556, 566)).sum())
    assert at_405 > 0
    assert at_405 == pytest.approx(at_561, rel=1e-6)


def test_spectral_detector_laser_powers() -> None:
    configs = spectral_detector(4, 500, 700, LASERS, laser_powers=POWERS)
    for oc in configs:
        assert [light.power for light in oc.lights] == POWERS
        assert oc.power == sum(POWERS)

    # without laser powers, nothing is set (as before)
    for oc in spectral_detector(4, 500, 700, LASERS):
        assert [light.power for light in oc.lights] == [None] * len(LASERS)
        assert oc.power is None

    with pytest.raises(ValueError):
        spectral_detector(4, 500, 700, LASERS, laser_powers=[1.0])
