""" Loader for major requirements data.
    Returns raw and indexed data for major requirements."""

import json
from loaders.major_requirements_index import build_index

def load_major_req():
    """Loader for major requirements data from JSON file.
    
    Returns:
        A dictionary containing the major requirements data.
    """

    with open("data/major_requirements.json", "r") as f:
        return json.load(f)

def get_major_req():
    """Retrieves the major requirements data and builds an index.

    Returns:
        A dictionary containing the raw major requirements data and the built index.
    """
    raw = load_major_req()
    index = build_index(raw)
    
    return {
        "raw": raw,
        "index": index
    }