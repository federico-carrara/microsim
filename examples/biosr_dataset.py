from typing import Literal, Any, Sequence
import os
import datetime
import json

import tifffile as tiff
import numpy as np
from tqdm import tqdm
import xarray as xr

from microsim import schema as ms
from microsim.schema.optical_config.lib import spectral_detector
from microsim.schema.dimensions import Axis

ImgStage = Literal["emission", "optical_pf", "digital_pf", "digital"]


# --- Set simulation parameters
n_simulations: int = 4
"""The number of images to simulate."""
labels: str = ["ER", "F-actin", "Microtubules"]
"""The labels of the structures to simulate."""
fluorophores: str = ["mTurquoise", "EGFP", "EYFP"]
"""The fluorophores associated with the structures to simulate."""
num_bands: int = 32
"""The number of spectral bands to acquire (i.e., physically, the number of cameras)."""
light_wavelengths: Sequence[int] = [435, 488, 514]
"""List of lasers to use for excitation."""
light_powers: Sequence[float] = [3., 3., 1.]
"""List of powers associate to each light source (work as scaling factors)."""
out_range: tuple[int, int] = (460, 550)
"""The range of wavelengths of the acquired spectrum in nm."""
exposure_ms: float = 5
"""The exposure time for the detector cameras in ms."""
detector_quantum_eff: float = 0.8
"""The quantum efficiency of the detector cameras."""


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
    channels: Sequence[ms.OpticalConfig],
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
    
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(1, 1004, 1004), scale=(0.02, 0.02, 0.02)),
        output_space={"downscale": (1, 4, 4)},
        sample=sample,
        channels=channels,
        modality=ms.Identity(),
        settings=ms.Settings(
            cache=custom_cache_settings, spectral_bins_per_emission_channel=1
        ),
        detector=ms.CameraCCD(qe=detector_qe, read_noise=6, bit_depth=12),
    )

    
def run_simulation(sim: ms.Simulation) -> tuple[xr.DataArray]:
    gt = sim.ground_truth() # (F, Z, Y, X)
    # print(f"Ground truth: {gt.sizes}")
    # print("----------------------------------")
    em_img = sim.emission_flux() # (C, F, Z, Y, X)
    # print(f"Emission image: {em_img.sizes}")
    # print("----------------------------------")
    opt_img_per_fluor = sim.optical_image_per_fluor() # (C, F, Z, Y, X)
    opt_img = opt_img_per_fluor.sum("f") # (C, Z, Y, X)
    # print(f"Optical image: {opt_img.sizes}")
    # print("----------------------------------")
    dig_img_per_fluor = sim.digital_image(opt_img_per_fluor) # (C, F, Z, Y, X)
    digital_img = sim.digital_image(opt_img) # (C, Z, Y, X)
    # print(f"Digital image: {digital_img.sizes}")
    # print("----------------------------------")    
    return em_img, opt_img_per_fluor, dig_img_per_fluor, digital_img


def simulate_dataset(
    labels: list[Literal["CCPs", "ER", "F-actin", "Microtubules"]],
    fluorophores: list[str],
    num_simulations: int,
    root_dir: str,
    detect_channels: Sequence[ms.OpticalConfig],
    detector_qe: float = 0.8,
) -> list[dict[str, xr.DataArray]]:
    sim_imgs = []
    for _ in tqdm(range(num_simulations), desc="Simulating images"):
        # print("----------------------------------")
        # print(f"SIMULATING IMAGE {i+1}")
        # print("----------------------------------")
        sim = init_simulation(
            labels=labels,
            fluorophores=fluorophores,
            root_dir=root_dir,
            channels=detect_channels,
            detector_qe=detector_qe
        )
        em_pf, opt_pf, dig_pf, dig = run_simulation(sim)
        em_pf = em_pf.sum(Axis.C)
        opt_pf = opt_pf.sum(Axis.C)
        dig_pf = dig_pf.sum(Axis.C)
        sim_imgs.append({
            "emission_pf": em_pf, 
            "optical_pf": opt_pf, 
            "digital_pf": dig_pf, 
            "digital": dig
        })
        
    # Create simulation metadata
    sim_metadata = {
        "structures": labels, 
        "fluorophores": fluorophores,
        "shape": list(sim_imgs[0]["digital"].shape[2:]),
        "downscale": sim.output_space.downscale,
        "detect_exposure_ms": exposure_ms,
        "detect_quantum_eff": detector_qe,
        "light_powers": light_powers,
        "light_wavelengths": light_wavelengths,
        "wavelength_range": out_range,
        "dtype": str(sim_imgs[0]["digital"].dtype),
    }
    return sim_imgs, sim_metadata


def save_simulation_results(
    results: list[dict[xr.DataArray]],
    save_dir: str,
    stages: list[ImgStage] = ["optical_pf", "digital_pf", "digital"],
    dtype: Literal["8bit", "16bit"] = "16bit",
) -> None:
    print(f"Saving images into {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    for stage in stages:
        print("----------------------------------")
        print(f"Saving {stage} images...")
        for i, img_dict in enumerate(results):
            img = img_dict[stage].values
            img = normalize_image(img, dtype)
            fname = f"{stage}_img_{i+1}.tif"
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
    coords_info = {
        "x_coords": img.coords[Axis.X].values.tolist(),
        "y_coords": img.coords[Axis.Y].values.tolist(),
        "z_coords": img.coords[Axis.Z].values.tolist(),
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
    
    # define spectral detector
    detect_channels = spectral_detector(
        bins=num_bands,
        min_wave=out_range[0],
        max_wave=out_range[1],
        lasers=light_wavelengths,
        powers=light_powers,
        exposure_ms=exposure_ms,
        beam_splitter=False,
    )
    # run simulations
    res, sim_metadata = simulate_dataset(
        labels=labels,
        fluorophores=fluorophores,
        num_simulations=n_simulations,
        root_dir=ROOT_DIR,
        detect_channels=detect_channels,
        detector_qe=detector_quantum_eff,
    )
    
    # Saving results
    path_to_save_dir = get_save_path(SAVE_DIR)
    save_metadata(
        img=res[0]["digital"], sim_metadata=sim_metadata, save_dir=path_to_save_dir
    )
    save_simulation_results(
        results=res, 
        save_dir=os.path.join(path_to_save_dir, "imgs"),
        stages=["optical_pf", "digital_pf", "digital"],
        dtype="16bit",
    )
    
    # # Display some results
    # N, F = len(res), res[0]["optical_pf"].sizes["f"]
    # fig, ax = plt.subplots(3, 5, figsize=(30, 6))
    # fig.suptitle("Some Examples of Simulated Images", fontsize=20)
    # sp_bands_idxs = [4, 12, 20, 24, 28]
    # for i, img_dict in enumerate(res[:3]):
    #     for j, sp_bands_idx in enumerate(sp_bands_idxs):
    #         curr_img = img_dict["digital"][sp_bands_idx, 0, ...].values
    #         ax[i, j].set_title(f"Digital Mixed image - Sample {i+1} - Band {sp_bands_idx}")
    #         ax[i, j].imshow(curr_img, cmap="gray")
    # plt.show()