from customocr.data.generate_synthetic import random_strings, predefined_strings

def get_generator(name: str):
    """Factory method to get string generator functions."""
    generators = {
        "random_strings": random_strings,
        "predefined_strings": predefined_strings
    }
    if name not in generators:
        raise ValueError(f"Generator '{name}' not found. Available generators: {list(generators.keys())}")
    return generators[name]