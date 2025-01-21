import random


class WordsCreator:
    def __init__(self,
                 n_words_cat: int = 2500) -> None:
        self.n_words_cat = n_words_cat

        self.lower_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.upper_alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.alphabet = self.lower_alphabet + self.upper_alphabet

        self.cmn_lower_letters = "абвгдеж"
        self.cmn_upper_letters = "AБВГДЕЖ"

        self.nat_numbers = "123456789"

        self.puncts = ",.:;)"
        self.cmn_puncts = ".,;"

        self.roman_lookup = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                             (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

    @staticmethod
    def write_txt(txt_path: str,
                  lines: list[str]) -> None:
        with open(txt_path, 'w', encoding="utf-8") as txt_file:
            for line in lines:
                txt_file.write(line + "\n")

        return None

    def postfix_punct(self, word: str, mode="common") -> str:
        puncts = self.cmn_puncts if mode == "common" else self.puncts
        if random.random() > 0.5:
            word += random.choice(puncts)

        return word

    def create_short_numbers(self, words: list[str]) -> None:
        for i in range(1000):
            words.append(str(i))
            words.append("(" + str(i) + ")")

            for punct in self.puncts:
                words.append(str(i) + punct)

    def create_long_numbers(self,
                            words: list[str],
                            start: int = 1000,
                            end: int = 100000) -> None:
        for _ in range(self.n_words_cat):
            word = str(random.randint(start, end))
            word = self.postfix_punct(word)

            words.append(word)

    def create_roman_numbers(self,
                             words: list[str]) -> None:
        for n in range(1, 1000):
            number = ''
            for value, numeral in self.roman_lookup:
                count, n = divmod(n, value)
                number += numeral * count

            number = self.postfix_punct(number, mode="puncts")

            words.append(number)

    def create_letters(self, words: list[str]) -> None:
        for letter in self.alphabet:
            words.append(letter)
            for punct in self.puncts:
                words.append(letter + punct)

    def create_question_nums(self, words: list[str]) -> None:
        for i in range(1, 51):
            words.append("А" + str(i) + ".")
            words.append("Б" + str(i) + ".")

    def create_short_a_answers(self,
                               words: list[str],
                               max_length: int = 4) -> None:
        for _ in range(self.n_words_cat):

            for num in self.nat_numbers:
                word = num

                length = random.randint(1, max_length)
                letters = sorted(random.sample(self.cmn_lower_letters, length))

                for letter in letters:
                    word += letter

                word = self.postfix_punct(word)

                words.append(word)

    def create_long_a_answers(self,
                              words: list[str],
                              max_length: int = 4) -> None:
        for _ in range(self.n_words_cat):
            word = ""

            length = random.randint(1, max_length)
            space = random.choice(["", " "])

            letters = sorted(random.sample(self.cmn_lower_letters, length))

            for i, letter in enumerate(letters):
                word += letter

                if i + 1 != length:
                    word += "," + space
                else:
                    word += random.choice(self.cmn_puncts)

            words.append(word)

    def create_b_answers(self, words: list[str]) -> None:
        for _ in range(self.n_words_cat):

            word = ""

            length = random.randint(1, len(self.cmn_upper_letters))

            for i in range(length):
                word += (self.cmn_upper_letters[i] +
                         random.choice(self.nat_numbers))

            word = self.postfix_punct(word)

            words.append(word)

    def create(self, output_txt_path: str) -> None:
        all_words = []

        # Create different categories of words
        self.create_short_numbers(all_words)
        self.create_long_numbers(all_words)
        self.create_roman_numbers(all_words)
        self.create_question_nums(all_words)
        self.create_letters(all_words)
        self.create_question_nums(all_words)
        self.create_short_a_answers(all_words)
        self.create_long_a_answers(all_words)
        self.create_b_answers(all_words)

        all_words = sorted(list(set(all_words)))

        # Write words to txt
        self.write_txt(output_txt_path, lines=all_words)


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

        self.aug_methods = [self.augment_number, self.augment_letter,
                            self.augment_pre_puncts, self.augment_post_puncts,
                            self.augment_out_puncts]

    @staticmethod
    def read_txt(txt_path) -> list[str]:
        corpus = []
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            for word in txt_file.readlines():
                corpus.append(word.strip())

        return corpus

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

    def augment(self,
                corpus_txt_path: str,
                output_txt_path: str,
                n_words_aug: int = 25000) -> None:
        corpus = self.read_txt(corpus_txt_path)

        all_words = []

        for _ in range(n_words_aug):
            rnd_word = random.choice(corpus)
            rnd_method = random.choice(self.aug_methods)
            word = rnd_method(rnd_word)
            all_words.append(word)

        # Write words to txt
        WordsCreator.write_txt(output_txt_path, lines=all_words)
