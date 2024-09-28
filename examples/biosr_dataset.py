from typing import Literal, Any, Sequence
import os
import datetime
import json

import tifffile as tiff
import numpy as np
import xarray as xr

from microsim import schema as ms
from microsim.schema.optical_config.lib import spectral_detector

# --- Set simulation parameters
labels: str = ["ER", "F-actin", "Microtubules"]
"""The labels of the structures to simulate."""
fluorophores: str = ["mTurquoise", "EGFP", "EYFP"]
"""The fluorophores associated with the structures to simulate."""
num_bands: int = 32
"""The number of spectral bands to acquire (i.e., physically, the number of cameras)."""
light_wavelengths: Sequence[int] = [488, 561, 640]
"""List of lasers to use for excitation."""
light_power: float = 10
"""The power of the light source (applied in the optical config)."""
out_range: tuple[int, int] = (400, 700)
"""The range of wavelengths of the acquired spectrum in nm."""
exposure_ms: float = 100
"""The exposure time for the detector cameras in ms."""


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
    detector_qe: float = 0.8,
) -> ms.Simulation:
    assert len(labels) == len(fluorophores)
    
    custom_cache_settings = ms.settings.CacheSettings(
        read=False,
        write=False,
    )
    # create the GT sample
    sample = ms.Sample(
        labels=[
            create_distribution(label, fp, root_dir) 
            for label, fp in zip(labels, fluorophores)
        ]
    )
    # create the channels simulating the spectral detector
    # NOTE: this assumes excitation is done with lasers
    detect_channels = spectral_detector(
        bins=num_bands,
        min_wave=out_range[0],
        max_wave=out_range[1],
        lasers=light_wavelengths,
    )
    
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(1, 1004, 1004), scale=(0.02, 0.02, 0.02)),
        output_space={"downscale": (1, 4, 4)},
        sample=sample,
        channels=detect_channels,
        modality=ms.Confocal(pinhole_au=0.4),
        settings=ms.Settings(
            cache=custom_cache_settings, spectral_bins_per_emission_channel=1
        ),
        detector=ms.CameraCCD(qe=detector_qe, read_noise=6, bit_depth=12),
    )

    
def run_simulation(sim: ms.Simulation, detect_exposure: int = 100) -> xr.DataArray:
    gt = sim.ground_truth()
    print(f"Ground truth: {gt.sizes}") # (F, Z, Y, X)
    print("----------------------------------")
    em_img = sim.emission_flux()
    print(f"Emission image: {em_img.sizes}") # (C, F, Z, Y, X)
    print("----------------------------------")
    opt_img_per_fluor = sim.optical_image_per_fluor() # (C, F, Z, Y, X)
    opt_img = opt_img_per_fluor.sum("f")
    print(f"Optical image: {opt_img.sizes}") # (C, Z, Y, X)
    print("----------------------------------")
    digital_img = sim.digital_image(opt_img)
    # TODO: add digital GT (C, F, Z, Y, X)
    print(f"Digital image: {digital_img.sizes}") # (C, Z, Y, X)
    print("----------------------------------")    
    return digital_img


def simulate_dataset(
    labels: list[Literal["CCPs", "ER", "F-actin", "Microtubules"]],
    fluorophores: list[str],
    num_simulations: int,
    root_dir: str,
    detector_qe: float = 0.8,
    detect_exposure: int = 100,
) -> list[xr.DataArray]:
    sim_imgs = []
    for i in range(num_simulations):
        print("----------------------------------")
        print(f"SIMULATING IMAGE {i+1}")
        print("----------------------------------")
        sim = init_simulation(labels, fluorophores, root_dir, detector_qe)
        sim_imgs.append(run_simulation(sim, detect_exposure)) 
        
    # Create simulation metadata
    wave_range = [
       int(sim.channels[0].filters[0].transmission.wavelength[0].magnitude),
       int(sim.channels[0].filters[0].transmission.wavelength[-1].magnitude),
    ]
    sim_metadata = {
        "structures": labels, 
        "fluorophores": fluorophores,
        "shape": list(sim_imgs[0].shape[-3:]),
        "downscale": sim.output_space.downscale,
        "detect_exposure_ms": detect_exposure,
        "detect_quantum_eff": detector_qe,
        "wavelength_range": wave_range,
        "dtype": str(sim_imgs[0].dtype),
    }
    return sim_imgs, sim_metadata


def save_simulation_results(
    results: list[xr.DataArray],
    save_dir: str,
    dtype: Literal["8bit", "16bit"] = "16bit",
) -> None:
    print(f"Saving simulated data into {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(results):
        img = img.values
        img = normalize_image(img, dtype)
        fname = f"simulated_img_{i+1}.tif"
        tiff.imwrite(os.path.join(save_dir, fname), img.squeeze()) 

    
def save_metadata(
    img: xr.DataArray,
    sim_metadata: dict[str, Any],
    save_dir: str,
) -> None:
    """Save Metadata of the simulation."""
    from microsim.schema.dimensions import Axis
    print(f"Saving metadata into {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    w_bins = [
        (img.coords[Axis.W].values[i].left, img.coords[Axis.W].values[i].right)
        for i in range(len(img.coords[Axis.W].values))
    ]
    coords_info = {
        "x_coords": img.coords[Axis.X].values.tolist(),
        "y_coords": img.coords[Axis.Y].values.tolist(),
        "z_coords": img.coords[Axis.Z].values.tolist(),
        "w_bins": w_bins,
    }
    with open(os.path.join(save_dir, "sim_coords.json"), "w") as f:
        json.dump(coords_info, f)
    with open(os.path.join(save_dir, "sim_metadata.json"), "w") as f:
        json.dump(sim_metadata, f)


def get_save_path(root_dir: str) -> str:
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%y%m%d")
    current_dir = os.path.join(root_dir, formatted_date)
    return get_unique_directory_path(current_dir)


def get_unique_directory_path(base_path: str) -> str:
    version = 0
    new_path = f"{base_path}_v{version}"
    while os.path.exists(new_path):
        version += 1
        new_path = f"{base_path}_v{version}"
    return new_path


def normalize_image(
    img: np.ndarray, 
    dtype: Literal["8bit", "16bit"]
) -> np.ndarray:
    """Normalize an image for each flurophore separately.
    
    Image has shape (W, F, Z, Y, X).
    """ 
    min_vals = np.min(img, axis=(0, -3, -2, -1), keepdims=True)
    max_vals = np.max(img, axis=(0, -3, -2, -1), keepdims=True)
    img = (img - min_vals) / (max_vals - min_vals)
    if dtype == "8bit":
        return (img * 255).astype(np.uint8)
    elif dtype == "16bit":
        return (img * 65535).astype(np.uint16)
    else:
        ValueError(f"Invalid dtype: {dtype}. Available options are '8bit' or '16bit'.")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT_DIR = "/group/jug/federico/careamics_training/data/BioSR"
    SAVE_DIR = "/group/jug/federico/microsim/BIOSR_spectral_data"

    import matplotlib.pyplot as plt

    res, sim_metadata = simulate_dataset(
        labels=labels,
        fluorophores=fluorophores,
        num_simulations=100,
        root_dir=ROOT_DIR
    )
    
    # Saving results
    path_to_save_dir = get_save_path(SAVE_DIR)
    save_metadata(res[0], sim_metadata, path_to_save_dir)
    save_simulation_results(res, os.path.join(path_to_save_dir, "imgs"))
    
    # Display some results
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