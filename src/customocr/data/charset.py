# ---------------------------
# Character mapping
# ---------------------------
CHAR_TO_IDX = {
    "<blank>": 0,  # CTC blank token
    " ": 1,  # space
}

# Add lowercase letters
for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
    CHAR_TO_IDX[char] = i + 2

# Add uppercase letters (optional)
for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CHAR_TO_IDX[char] = i + 28

# Add digits
for i, char in enumerate("0123456789"):
    CHAR_TO_IDX[char] = i + 56

# Optional unknown token
CHAR_TO_IDX["<unk>"] = len(CHAR_TO_IDX)

IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
