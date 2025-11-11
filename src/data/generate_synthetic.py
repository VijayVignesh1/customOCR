import os
from trdg.generators import GeneratorFromRandom, GeneratorFromStrings
from tqdm import tqdm

def random_strings(output_dir: str = "data/synthetic/train", 
                            count: int = 100, 
                            length: int = 10,
                            size: int = 100,
                            language: str = "en",
                            text_color: str = "#000000",
                            skewing_angle: int = 3,
                            **kwargs
                            ):
    os.makedirs(output_dir, exist_ok=True)
    generator = GeneratorFromRandom(
        count,
        blur=0,
        random_blur=False,
        skewing_angle=skewing_angle,
        background_type=1,           # plain white
        text_color=text_color,
        length=length,
        size=size,
        language=language,
        **kwargs
    )
    with open(os.path.join(output_dir, "labels.txt"), "w") as lf:
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            lf.write(f"{img_path}\t{lbl}\n")

def predefined_strings(output_dir: str = "data/synthetic/train", 
                                strings: list = ['Test1', 'Test2', 'Test3'],
                                length: int = 10,
                                size: int = 100,
                                language: str = "en",
                                text_color: str = "#000000",
                                skewing_angle: int = 3,
                                **kwargs
                                ):
    os.makedirs(output_dir, exist_ok=True)
    generator = GeneratorFromStrings(
        strings=strings,
        count = 100,
        blur=0,
        random_blur=False,
        skewing_angle=skewing_angle,
        background_type=1,           # plain white
        text_color=text_color,
        size=size,
        language=language,
        **kwargs
    )
    with open(os.path.join(output_dir, "labels.txt"), "w") as lf:
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            lf.write(f"{img_path}\t{lbl}\n")

