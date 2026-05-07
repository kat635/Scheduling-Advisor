""" Loader for general education requirements. 
    Loads and indexes general education requirements."""

from pathlib import Path
import json

def load_gen_ed(path="data/gen_ed.json"):
    """
    Loader for general education requirements data from JSON file.

    Args:
        path: Path to the gen_ed JSON file (default: "data/gen_ed.json")

    Returns:
        Parsed JSON data as a Python dictionary
    """
    path = Path(path)

    with path.open() as f:
        return json.load(f)

def build_gen_ed_index(data):
    """
    Build an index mapping requirement IDs to requirement objects.

    Args:
        data: Parsed JSON data as a Python dictionary

    Returns:
        Dictionary mapping requirement IDs to requirement objects
    """
    # Create index mapping requirement IDs to requirement objects
    index = {req["id"]: req for req in data.get("requirements", [])}
    return index

def get_gen_ed_requirements(path="data/gen_ed.json"):
    """
    Load and index general education requirements data.

    Args:
        path: Path to the gen_ed JSON file (default: "data/gen_ed.json")

    Returns:
        Dictionary containing raw data and index of general education requirements
    """
    raw = load_gen_ed(path)
    
    # Create index mapping requirement IDs to requirement objects
    index = build_gen_ed_index(raw)
    
    return {
        "raw": raw,
        "index": index
    }