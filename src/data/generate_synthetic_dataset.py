import os
from trdg.generators import GeneratorFromRandom
from tqdm import tqdm

def generate_synthetic_trdg(output_dir: str = "data/synthetic/train", 
                            count: int = 5000, 
                            length: int = 10,
                            size: int = 32,
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
        **kwargs
    )
    with open(os.path.join(output_dir, "labels.txt"), "w") as lf:
        for i, (img, lbl) in enumerate(tqdm(generator)):
            img_filename = f"{i:06d}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            lf.write(f"{img_path}\t{lbl}\n")

if __name__ == "__main__":
    generate_synthetic_trdg(
        output_dir="../../data/synthetic/train",
        count=5000,
        text_color="#000000",
        skewing_angle=5
    )
