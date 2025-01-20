import random


class WordsCreator:
    def __init__(self,
                 n_words_cat: int = 2500) -> None:
        self.n_words_cat = n_words_cat

        self.lower_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.upper_alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.alphabet = self.lower_alphabet + self.upper_alphabet

        self.cmn_lower_letters = "абвгдежзи"
        self.cmn_upper_letters = "AБВГДЕЖЗИ"

        self.nat_numbers = "123456789"

        self.puncts = ",.:;)"
        self.cmn_puncts = ".,;"

    @staticmethod
    def write_txt(txt_path: str,
                  lines: list[str]) -> None:
        with open(txt_path, 'w', encoding="utf-8") as txt_file:
            for line in lines:
                txt_file.write(line + "\n")

        return None

    def postfix_punct(self, word: str) -> str:
        if random.random() > 0.5:
            word += random.choice(self.cmn_puncts)

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
    pass
