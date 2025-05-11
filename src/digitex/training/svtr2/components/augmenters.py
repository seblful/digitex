import os
import random

from digitex.core.processors.file import FileProcessor


class BaseAugmenter:
    def __init__(self) -> None:
        self.textcases = ["lower", "upper", "capitalize", "mixed"]
        self.greek_transl = str.maketrans({"Α": "α", "Β": "β"})

    def random_textcase(self, word: str) -> str:
        style = random.choice(self.textcases)

        if style == "upper":
            word = word.upper()
        elif style == "lower":
            word = word.lower()
        elif style == "capitalize":
            word = word.capitalize()
        elif style == "mixed":
            word = "".join(random.choice((str.upper, str.lower))(char) for char in word)

        word = word.translate(self.greek_transl)

        return word

    def augment(self, input_txt_path: str, n_words_aug) -> None:
        corpus = FileProcessor.read_txt(input_txt_path, strip=True)

        all_words = []

        for _ in range(n_words_aug):
            rnd_word = random.choice(corpus)
            rnd_method = random.choice(self.aug_methods)

            word = self.random_textcase(rnd_word)
            word = rnd_method(word)
            word = word.strip()
            all_words.append(word)

        # Write words to txt
        output_dir = os.path.dirname(input_txt_path)
        input_name, ext = os.path.splitext(os.path.basename(input_txt_path))
        output_name = input_name + "_aug" + ext
        output_txt_path = os.path.join(output_dir, output_name)
        FileProcessor.write_txt(output_txt_path, lines=all_words, newline=True)


class SpecCharsAugmenter(BaseAugmenter):
    def __init__(self, spec_chars_txt_path: str) -> None:
        super().__init__()
        self.spec_chars_txt_path = spec_chars_txt_path
        self.__spec_chars = None

        self.aug_methods = [
            self.augment_prefix,
            self.augment_postfix,
            self.augment_both,
            self.augment_surround,
            self.augment_middle,
            self.augment_double_middle,
        ]

    @property
    def spec_chars(self) -> str:
        if self.__spec_chars is None:
            self.__spec_chars = FileProcessor.read_txt(
                self.spec_chars_txt_path, strip=True
            )[0]

        return self.__spec_chars

    def augment_prefix(self, word: str) -> str:
        char = random.choice(self.spec_chars)
        word = f"{char}{word}"

        return word

    def augment_postfix(self, word: str) -> str:
        char = random.choice(self.spec_chars)
        word = f"{word}{char}"

        return word

    def augment_both(self, word: str) -> str:
        prefix = random.choice(self.spec_chars)
        suffix = random.choice(self.spec_chars)
        word = f"{prefix}{word}{suffix}"

        return word

    def augment_surround(self, word: str) -> str:
        char = random.choice(self.spec_chars)
        word = f"{char}{word}{char}"

        return word

    def augment_middle(self, word: str) -> str:
        char = random.choice(self.spec_chars)

        if len(word) <= 1:
            word = f"{word}{char}"

        else:
            pos = random.randint(1, len(word) - 1)
            word = f"{word[:pos]}{char}{word[pos:]}"

        return word

    def augment_double_middle(self, word: str) -> str:
        char1 = random.choice(self.spec_chars)
        char2 = random.choice(self.spec_chars)

        if len(word) <= 1:
            word = f"{word}{char1}{char2}"

        else:
            pos = random.randint(1, len(word) - 1)
            word = f"{word[:pos]}{char1}{char2}{word[pos:]}"

        return word


class TestsAugmenter(BaseAugmenter):
    def __init__(self) -> None:
        super().__init__()

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
