import os

from digitex.training.svtr2.components.augmenters import (
    SpecCharsAugmenter,
    TestsAugmenter,
)
from digitex.training.svtr2.components.creators import TestsCreator, FormulasCreator

HOME = os.getcwd()
SVTR_DATA_DIR = os.path.join(HOME, "src/digitex/training/svtr2/data")

TRAIN_RES_DIR = os.path.join(SVTR_DATA_DIR, "train", "synthtiger", "resources")
TRAIN_CORPUS_DIR = os.path.join(TRAIN_RES_DIR, "corpus")
TRAIN_CHARSET_DIR = os.path.join(TRAIN_RES_DIR, "charset")

FINETUNE_RES_DIR = os.path.join(SVTR_DATA_DIR, "finetune", "synthtiger", "resources")
FINETUNE_CORPUS_DIR = os.path.join(FINETUNE_RES_DIR, "corpus")
FINETUNE_CHARSET_DIR = os.path.join(FINETUNE_RES_DIR, "charset")


def augment_train_words(n_words_aug: int) -> None:
    spec_chars_txt_path = os.path.join(TRAIN_CHARSET_DIR, "spec_chars.txt")
    corpus_augmenter = SpecCharsAugmenter(spec_chars_txt_path=spec_chars_txt_path)
    for txt_path in os.listdir(TRAIN_CORPUS_DIR):
        skip_keywords = ["_aug", "formulas", "ions"]
        if any(keyword in txt_path for keyword in skip_keywords):
            continue
        full_path = os.path.join(TRAIN_CORPUS_DIR, txt_path)
        corpus_augmenter.augment(input_txt_path=full_path, n_words_aug=n_words_aug)


def augment_finetune_words(n_words_aug: int) -> None:
    forms_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "forms.txt")
    tests_augmenter = TestsAugmenter()
    tests_augmenter.augment(input_txt_path=forms_txt_path, n_words_aug=n_words_aug)


def create_finetune_tests() -> None:
    tests_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "testing.txt")
    tests_creator = TestsCreator()
    tests_creator.create(output_txt_path=tests_txt_path)


def create_finetune_formulas(n_formulas: int) -> None:
    elements_txt_path = os.path.join(FINETUNE_CHARSET_DIR, "elements.txt")
    train_ions_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "ions.txt")
    formulas_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "formulas_aug.txt")

    formulas_creator = FormulasCreator(elements_txt_path, train_ions_txt_path)
    formulas_creator.create(output_txt_path=formulas_txt_path, n_formulas=n_formulas)


def main() -> None:
    augment_train_words(200000)
    augment_finetune_words(50000)
    create_finetune_tests()
    create_finetune_formulas(100000)


if __name__ == "__main__":
    main()
