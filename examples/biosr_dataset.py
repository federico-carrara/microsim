from typing import Literal

import numpy as np
import xarray as xr

from microsim import schema as ms


def create_custom_channel(
    min_wave: int = 300, 
    max_wave: int = 800,
) -> ms.optical_config.OpticalConfig:

    custom_spectrum = ms.Spectrum(
        wavelength=np.arange(min_wave, max_wave, 1),
        intensity=np.ones(max_wave - min_wave),
    )
    custom_filter = ms.optical_config.SpectrumFilter(transmission=custom_spectrum) # placement=ALL by default
    custom_channel = ms.optical_config.OpticalConfig(
        name="FEDERICO",
        filters=[custom_filter],
    )
    
    return custom_channel

def create_distribution(
    label: Literal["CCPs", "ER", "F-actin", "Microtubules"],
    fluorophore: str,
    root_dir: str,
    idx: int | None = None, 
) -> ms.FluorophoreDistribution:
    return ms.FluorophoreDistribution(
        distribution=ms.BioSR(root_dir=root_dir, label=label),
        fluorophore=fluorophore, 
        img_idx=idx,  
    )
    
def init_simulation(
    labels: list[Literal["CCPs", "ER", "F-actin", "Microtubules"]],
    fluorophores: list[str],
    root_dir: str,
    light_pwrs: list[float] = [1., 1., 1.],
    detector_qe: float = 0.8,
) -> ms.Simulation:
    assert len(labels) == len(fluorophores)
    
    custom_cache_settings = ms.settings.CacheSettings(
        read=False,
        write=False,
    )
    sample = ms.Sample(
        labels=[
            create_distribution(label, fp, root_dir) 
            for label, fp in zip(labels, fluorophores)
        ]
    )
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(1, 1004, 1004), scale=(0.02, 0.02, 0.02)),
        output_space={"downscale": (1, 4, 4)},
        sample=sample,
        channels=[create_custom_channel(min_wave=460, max_wave=550)],
        modality=ms.Confocal(pinhole_au=2),
        settings=ms.Settings(max_psf_radius_aus=2, cache=custom_cache_settings),
        detector=ms.CameraCCD(qe=detector_qe, read_noise=6, bit_depth=12),
        emission_bins=32,
        light_powers=light_pwrs
    )
    
def run_simulation(sim: ms.Simulation, detect_exposure: int = 100) -> xr.DataArray:
    gt = sim.ground_truth()
    print(f"Ground truth: {gt.sizes}") # (F, Z, Y, X)
    print("----------------------------------")
    em_img, _, _ = sim.spectral_emission_flux(gt, channel_idx=0, plot_hist=False)
    print(f"Emission image: {em_img.sizes}") # (W, C, F+1, Z, Y, X)
    print("----------------------------------")
    sim.spectral_image = True
    opt_img = sim.optical_image(em_img, channel_idx=0)
    print(f"Optical image: {opt_img.sizes}") # (W, C, F+1, Z, Y, X)
    print("----------------------------------")
    digital_img = sim.digital_image(opt_img, exposure_ms=detect_exposure)
    print(f"Digital image: {digital_img.sizes}") # (W, C, F+1, Z, Y, X)
    print("----------------------------------")
    
    return digital_img

def simulate_dataset(
    labels: list[Literal["CCPs", "ER", "F-actin", "Microtubules"]],
    fluorophores: list[str],
    num_simulations: int,
    root_dir: str,
    light_pwrs: list[float] = [1., 1., 1.],
    detector_qe: float = 0.8,
    detect_exposure: int = 100,
) -> list[xr.DataArray]:
    sim_imgs = []
    for i in range(num_simulations):
        print("----------------------------------")
        print(f"SIMULATING IMAGE {i+1}")
        print("----------------------------------")
        sim = init_simulation(labels, fluorophores, root_dir, light_pwrs, detector_qe)
        sim_imgs.append(run_simulation(sim, detect_exposure)) 
    return sim_imgs

def save_simulation_results(
    results: list[xr.DataArray],
    save_dir: str,
) -> None:
    ...

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT_DIR = "/group/jug/federico/careamics_training/data/BioSR"

    import matplotlib.pyplot as plt

    res = simulate_dataset(
        labels=["ER", "F-actin", "Microtubules"],
        fluorophores=["mTurquoise", "EGFP", "EYFP"],
        num_simulations=3,
        root_dir=ROOT_DIR,
        light_pwrs=[15., 3., 2.],
    )
    
    N, F = len(res), res[0].sizes["f"]
    
    fig, ax = plt.subplots(3, F, figsize=(15, 30))
    fig.suptitle("Some Examples of Simulated Images", fontsize=20)
    sp_bands_idxs = [4, 12, 20]
    for i, img in enumerate(res[:3]):
        for j in range(F):
            if j == F-1:
                ax[i, j].set_title(f"Mixed image - Sample {i+1}")
                ax[i, j].imshow(img[10, 0, j, 0, ...], cmap="gray")
            else:
                ax[i, j].set_title(f"{img.coords['f'].values[j].fluorophore.name} - Sample {i+1}")
                ax[i, j].imshow(img[sp_bands_idxs[j], 0, j, 0, ...], cmap="gray")
    plt.show()