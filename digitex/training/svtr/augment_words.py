import os

from components.words import WordsCreator, CorpusAugmenter, WordsAugmenter

HOME = os.getcwd()
DATA = os.path.join(HOME, "data")

TRAIN_RAW_DIR = os.path.join(DATA, "train", "raw-data")
TRAIN_SYNTH_DIR = os.path.join(TRAIN_RAW_DIR, "synthtiger")
TRAIN_CORPUS_DIR = os.path.join(TRAIN_SYNTH_DIR, "corpus")
TRAIN_CHARSET_DIR = os.path.join(TRAIN_SYNTH_DIR, "charset")

FINETUNE_RAW_DIR = os.path.join(DATA, "finetune", "raw-data")
FINETUNE_SYNTH_DIR = os.path.join(FINETUNE_RAW_DIR, "synthtiger")
FINETUNE_CORPUS_DIR = os.path.join(FINETUNE_SYNTH_DIR, "corpus")

symbols_txt_path = os.path.join(TRAIN_CHARSET_DIR, "numpunct.txt")

finetune_words_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "testing.txt")
finetune_forms_txt_path = os.path.join(FINETUNE_CORPUS_DIR, "forms.txt")


def main() -> None:
    # Create WordsCreator instance and create words
    words_creator = WordsCreator()
    words_creator.create(output_txt_path=finetune_words_txt_path)

    # Augment corpus for training
    corpus_augmenter = CorpusAugmenter(symbols_txt_path=symbols_txt_path)
    for corpus_txt_path in os.listdir(TRAIN_CORPUS_DIR):
        if "_aug" not in corpus_txt_path:
            corpus_txt_path = os.path.join(TRAIN_CORPUS_DIR, corpus_txt_path)
            corpus_augmenter.augment(corpus_txt_path=corpus_txt_path,
                                     n_words_aug=50000)

    # Augment corpus for finetuning
    words_augmenter = WordsAugmenter()
    words_augmenter.augment(input_txt_path=finetune_forms_txt_path,
                            n_words_aug=25000)


if __name__ == "__main__":
    main()
