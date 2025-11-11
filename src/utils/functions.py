import random

def generate_random_word():
    """Generate a random word consisting of lowercase letters and digits."""
    letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
    word_length = random.randint(5, 10)
    return ''.join(random.choice(letters) for _ in range(word_length))