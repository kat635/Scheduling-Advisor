"""Main loader for program data, including general education and major requirements."""

from loaders.gen_ed_loader import get_gen_ed_requirements
from loaders.major_requirements_loader import get_major_req

def load_program_data():
    return {
        "gen_ed": get_gen_ed_requirements(),
        "major": get_major_req()
    }