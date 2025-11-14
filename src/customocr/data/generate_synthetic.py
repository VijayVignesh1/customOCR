import json
import os

from tqdm import tqdm
from trdg.generators import GeneratorFromRandom
from trdg.generators import GeneratorFromStrings


def random_strings(
    output_dir: str = "data/synthetic/train",
    **kwargs,
):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(
            f"[INFO] Dataset already exists at {output_dir}. \n"
            f"Skipping generation."
        )
        return
    os.makedirs(output_dir, exist_ok=True)
    generator = GeneratorFromRandom(**kwargs)

    # generate as json
    with open(os.path.join(output_dir, "labels.json"), "w") as lf:
        label_dict = {}
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            label_dict[img_path] = lbl
        lf.write(json.dumps(label_dict))


def predefined_strings(
    output_dir: str = "data/synthetic/train",
    **kwargs,
):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(
            f"[INFO] Dataset already exists at {output_dir}. \n"
            f"Skipping generation."
        )
        return
    os.makedirs(output_dir, exist_ok=True)
    # Remove count parameter since GeneratorFromStrings uses strings list length
    generator = GeneratorFromStrings(**kwargs)
    with open(os.path.join(output_dir, "labels.json"), "w") as lf:
        label_dict = {}
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            label_dict[img_path] = lbl
        lf.write(json.dumps(label_dict))
