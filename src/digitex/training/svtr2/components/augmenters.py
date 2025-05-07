import os
import random

from digitex.core.processors.file import FileProcessor


class CorpusAugmenter:
    def __init__(self, symbols_txt_path: str) -> None:
        self.symbols_txt_path = symbols_txt_path
        self.__symbols = None

        self.aug_methods = [
            self.augment_prefix,
            self.augment_postfix,
            self.augment_both,
            self.augment_surround,
            self.augment_middle,
            self.augment_double_middle,
        ]

    @property
    def symbols(self) -> str:
        if self.__symbols is None:
            self.__symbols = FileProcessor.read_txt(self.symbols_txt_path, strip=True)[
                0
            ]

        return self.__symbols

    def augment_prefix(self, word: str) -> str:
        symbol = random.choice(self.symbols)
        word = f"{symbol}{word}"

        return word

    def augment_postfix(self, word: str) -> str:
        symbol = random.choice(self.symbols)
        word = f"{word}{symbol}"

        return word

    def augment_both(self, word: str) -> str:
        prefix = random.choice(self.symbols)
        suffix = random.choice(self.symbols)
        word = f"{prefix}{word}{suffix}"

        return word

    def augment_surround(self, word: str) -> str:
        symbol = random.choice(self.symbols)
        word = f"{symbol}{word}{symbol}"

        return word

    def augment_middle(self, word: str) -> str:
        symbol = random.choice(self.symbols)

        if len(word) <= 1:
            word = f"{word}{symbol}"

        else:
            pos = random.randint(1, len(word) - 1)
            word = f"{word[:pos]}{symbol}{word[pos:]}"

        return word

    def augment_double_middle(self, word: str) -> str:
        symbol1 = random.choice(self.symbols)
        symbol2 = random.choice(self.symbols)

        if len(word) <= 1:
            word = f"{word}{symbol1}{symbol2}"

        else:
            pos = random.randint(1, len(word) - 1)
            word = f"{word[:pos]}{symbol1}{symbol2}{word[pos:]}"

        return word

    def augment(self, corpus_txt_path: str, n_words_aug: int = 100000) -> None:
        corpus = FileProcessor.read_txt(corpus_txt_path, strip=True)

        all_words = []

        for _ in range(n_words_aug):
            rnd_word = random.choice(corpus)
            rnd_method = random.choice(self.aug_methods)
            word = rnd_method(rnd_word)
            word = word.strip()
            all_words.append(word)

        # Write words to txt
        output_dir = os.path.dirname(corpus_txt_path)
        input_name, ext = os.path.splitext(os.path.basename(corpus_txt_path))
        output_name = input_name + "_aug" + ext
        output_txt_path = os.path.join(output_dir, output_name)
        FileProcessor.write_txt(output_txt_path, lines=all_words, newline=True)


class WordsAugmenter:
    def __init__(self) -> None:
        self.cmn_lower_letters = "абвгдеж"
        self.cmn_upper_letters = "AБВГДЕЖ"
        self.cmn_letters = self.cmn_lower_letters + self.cmn_upper_letters

        self.nat_numbers = "123456789"

        self.cmn_puncts = ".,;"
        self.count_puncts = ".)"
        self.pre_puncts = '«("'
        self.post_puncts = '".,:;?!)»-'
        self.out_puncts = ["«»", "()", '""', "[]"]

        self.aug_methods = [
            self.augment_number,
            self.augment_letter,
            self.augment_pre_puncts,
            self.augment_post_puncts,
            self.augment_out_puncts,
        ]

    def postfix_punct(self, word: str) -> str:
        if random.random() > 0.5:
            word += random.choice(self.cmn_puncts)

        return word

    def augment_number(self, word: str) -> str:
        number = random.choice(self.nat_numbers)
        punct = random.choice(self.count_puncts)

        word = number + punct + word
        word = self.postfix_punct(word)

        return word

    def augment_letter(self, word: str) -> str:
        letter = random.choice(self.cmn_letters)
        punct = random.choice(self.count_puncts)

        word = letter + punct + word
        word = self.postfix_punct(word)

        return word

    def augment_pre_puncts(self, word: str) -> str:
        punct = random.choice(self.pre_puncts)

        word = punct + word

        return word

    def augment_post_puncts(self, word: str) -> str:
        punct = random.choice(self.post_puncts)

        word = word + punct
        word = self.postfix_punct(word)

        return word

    def augment_out_puncts(self, word: str) -> str:
        punct = random.choice(self.out_puncts)
        pre_punct, post_punct = punct

        word = pre_punct + word + post_punct
        word = self.postfix_punct(word)

        return word

    def augment(self, input_txt_path: str, n_words_aug: int = 25000) -> None:
        corpus = FileProcessor.read_txt(input_txt_path, strip=True)

        all_words = []

        for _ in range(n_words_aug):
            rnd_word = random.choice(corpus)
            rnd_method = random.choice(self.aug_methods)
            word = rnd_method(rnd_word)
            all_words.append(word)

        # Write words to txt
        output_dir = os.path.dirname(input_txt_path)
        input_name, ext = os.path.splitext(os.path.basename(input_txt_path))
        output_name = input_name + "_aug" + ext
        output_txt_path = os.path.join(output_dir, output_name)
        FileProcessor.write_txt(output_txt_path, lines=all_words, newline=True)
