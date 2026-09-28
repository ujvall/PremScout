import unicodedata

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    return " ".join(norm.lower().split())

name = "Emiliano Mart\xc3\xadnez"
print("Decoded:", name)
print("Normalized:", normalize_name(name))
print("Equal?", normalize_name(name) == "emiliano martinez")
