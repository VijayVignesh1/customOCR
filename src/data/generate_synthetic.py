import os
import json
from trdg.generators import GeneratorFromRandom, GeneratorFromStrings
from tqdm import tqdm

def random_strings(output_dir: str = "data/synthetic/train", 
                            # count: int = 100, 
                            # length: int = 10,
                            # size: int = 100,
                            # language: str = "en",
                            # text_color: str = "#000000",
                            # skewing_angle: int = 3,
                            **kwargs
                            ):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"[INFO] Dataset already exists at {output_dir}. Skipping generation.")
        return
    os.makedirs(output_dir, exist_ok=True)
    generator = GeneratorFromRandom(
        # output_dir=output_dir,
        # count,
        # blur=0,
        # random_blur=False,
        # skewing_angle=skewing_angle,
        # background_type=1,           # plain white
        # text_color=text_color,
        # length=length,
        # size=size,
        # language=language,
        **kwargs
    )

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
                                # strings: list = ['Test1', 'Test2', 'Test3'],
                                # size: int = 100,
                                # language: str = "en",
                                # text_color: str = "#000000",
                                # skewing_angle: int = 3,
                                **kwargs
                                ):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"[INFO] Dataset already exists at {output_dir}. Skipping generation.")
        return
    os.makedirs(output_dir, exist_ok=True)
    # Remove count parameter since GeneratorFromStrings uses strings list length
    generator = GeneratorFromStrings(
        # output_dir=output_dir,
        # strings,  # Pass strings as positional argument
        # blur=0,
        # random_blur=False,
        # skewing_angle=skewing_angle,
        # background_type=1,           # plain white
        # text_color=text_color,
        # size=size,
        # language=language,
        **kwargs
    )
    with open(os.path.join(output_dir, "labels.json"), "w") as lf:
        label_dict = {}
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            label_dict[img_path] = lbl
        lf.write(json.dumps(label_dict))
