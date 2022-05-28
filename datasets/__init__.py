from .vidstg import build as build_vidstg
from .hcstvg import build as build_hcstvg
from .ego4d import build as build_ego4d

def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "vidstg":
        return build_vidstg(image_set, args)
    if dataset_file == "hcstvg":
        return build_hcstvg(image_set, args)
    if dataset_file == "ego4d":
        return build_ego4d(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
