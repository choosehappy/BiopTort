#%%
import openslide
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

#%%
path = "/media/jackson/backup/dp_data/tortuosity_study/tortuosity_test_set/"
output_path = "/media/jackson/backup/dp_data/tortuosity_study/tortuosity_test_set/pngs_with_scale/"




def get_slide_paths(directory: str) -> list[str]:
    return glob.glob(os.path.join(directory, "*.svs"))

def overlay_scale_on_slide(slide_path: str, output_path: str):
    slidename = os.path.basename(slide_path)
    slide = openslide.OpenSlide(slide_path)
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)    # microns per pixel in x direction
    width = slide.dimensions[0]
    height = slide.dimensions[1]

    pixels_per_mm = 1000 / float(mpp_x)  # pixels per mm

    downsample_factor = 10  # Factor to downsample the image
    downsample_width = width // downsample_factor
    downsample_height = height // downsample_factor
    downsample_pixels_per_mm = pixels_per_mm / downsample_factor  # Adjusted pixels per mm after downsampling
    thumbnail = slide.get_thumbnail((downsample_width, downsample_height))

    plt.figure(figsize=(10, 10))
    plt.imshow(thumbnail)
    scalebar_length_mm = 1  # Length of the scale bar in mm
    scalebar_length_pixels = scalebar_length_mm * downsample_pixels_per_mm

    plt.plot(
        [50, 50 + scalebar_length_pixels],  # x-coordinates of the scale bar
        [thumbnail.size[1] - 50, thumbnail.size[1] - 50],  # y-coordinates of the scale bar
        color="black",
        linewidth=3,
    )
    plt.text(
        50,
        thumbnail.size[1] - 75,
        f"{scalebar_length_mm} mm",
        # color="white",
        fontsize=12,
        # backgroundcolor="black",
    )
    plt.axis("off")
    plt.savefig(os.path.join(output_path, slidename.replace('.svs', '.png')), bbox_inches='tight', pad_inches=0.1)



# %%

if not os.path.exists(output_path):
    os.makedirs(output_path)

slide_paths = get_slide_paths(path)
for slide_path in tqdm(slide_paths):
    # print(f"Processing {slide_path}")
    overlay_scale_on_slide(slide_path, output_path)


# %%
