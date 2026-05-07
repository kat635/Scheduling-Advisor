""" Main program data module.
    Loads and provides access to all program-related data, including general education and major requirements."""

from loaders.program_loader import load_program_data

_PROGRAM_DATA = load_program_data()

def get_program_data():
    return _PROGRAM_DATA

def get_gen_ed():
    return _PROGRAM_DATA["gen_ed"]

def get_major():
    return _PROGRAM_DATA["major"]