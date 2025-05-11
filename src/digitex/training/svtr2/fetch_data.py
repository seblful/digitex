import os

from digitex.training.svtr2.components.fetchers import PubChemFetcher

HOME = os.getcwd()
SVTR_DATA_DIR = os.path.join(HOME, "src/digitex/training/svtr2/data")

TRAIN_SYNTH_DIR = os.path.join(SVTR_DATA_DIR, "train", "synthtiger")
TRAIN_PUBCHEM_DIR = os.path.join(TRAIN_SYNTH_DIR, "pubchem")

elements_info_json_path = os.path.join(TRAIN_PUBCHEM_DIR, "elements_info.json")
element_cids_txt_path = os.path.join(TRAIN_PUBCHEM_DIR, "element_cids.txt")
substructure_cids_txt_path = os.path.join(TRAIN_PUBCHEM_DIR, "substructure_cids.txt")
all_mfs_txt_path = os.path.join(TRAIN_PUBCHEM_DIR, "all_mfs.txt")


def main() -> None:
    pubchem_fetcher = PubChemFetcher(elements_info_json_path)
    pubchem_fetcher.create_mf_unicode_txts(all_mfs_txt_path, TRAIN_PUBCHEM_DIR)


if __name__ == "__main__":
    main()
