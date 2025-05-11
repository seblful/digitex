import random
import re
import math

from digitex.core.processors.file import FileProcessor


class TestsCreator:
    def __init__(self, n_words_cat: int = 2500) -> None:
        self.n_words_cat = n_words_cat

        self.lower_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.upper_alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.alphabet = self.lower_alphabet + self.upper_alphabet

        self.cmn_lower_letters = "абвгдеж"
        self.cmn_upper_letters = "AБВГДЕЖ"

        self.nat_numbers = "123456789"

        self.puncts = ",.:;)"
        self.cmn_puncts = ".,;"

        self.roman_lookup = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

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

    def create_long_numbers(
        self, words: list[str], start: int = 1000, end: int = 100000
    ) -> None:
        for _ in range(self.n_words_cat):
            word = str(random.randint(start, end))
            word = self.postfix_punct(word)

            words.append(word)

    def create_roman_numbers(self, words: list[str]) -> None:
        for n in range(1, 1000):
            number = ""
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

    def create_short_a_answers(self, words: list[str], max_length: int = 4) -> None:
        for _ in range(self.n_words_cat):
            for num in self.nat_numbers:
                word = num

                length = random.randint(1, max_length)
                letters = sorted(random.sample(self.cmn_lower_letters, length))

                for letter in letters:
                    word += letter

                word = self.postfix_punct(word)

                words.append(word)

    def create_long_a_answers(self, words: list[str], max_length: int = 4) -> None:
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
                word += self.cmn_upper_letters[i] + random.choice(self.nat_numbers)

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
        FileProcessor.write_txt(output_txt_path, lines=all_words, newline=True)


class FormulasCreator:
    def __init__(self, elements_txt_path: str, ions_txt_path: str) -> None:
        self.elements_txt_path = elements_txt_path
        self.ions_txt_path = ions_txt_path

        self.scripts = "0123456789"
        self.superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        self.subscripts = "₀₁₂₃₄₅₆₇₈₉"

        self.__elements = None
        self.__ions = None
        self.__cations = None
        self.__anions = None

    @property
    def elements(self) -> list[str]:
        if self.__elements is None:
            self.__elements = FileProcessor.read_txt(self.elements_txt_path, strip=True)
        return self.__elements

    @property
    def ions(self) -> list[str]:
        if self.__ions is None:
            self.__ions = FileProcessor.read_txt(self.ions_txt_path, strip=True)
        return self.__ions

    @property
    def anions(self) -> list[str]:
        if self.__anions is None:
            self.__anions = [ion for ion in self.ions if ion.endswith("⁻")]
        return self.__anions

    @property
    def cations(self) -> list[str]:
        if self.__cations is None:
            self.__cations = [ion for ion in self.ions if ion.endswith("⁺")]
        return self.__cations

    def num_to_subscript(self, text) -> str:
        subscript_map = str.maketrans(self.scripts, self.subscripts)
        return text.translate(subscript_map)

    def num_to_superscript(self, text) -> str:
        superscript_map = str.maketrans(self.scripts, self.superscripts)
        return text.translate(superscript_map)

    def subscript_to_num(self, text) -> str:
        subscript_map = str.maketrans(self.subscripts, self.scripts)
        return text.translate(subscript_map)

    def superscript_to_num(self, text) -> str:
        superscript_map = str.maketrans(self.superscripts, self.scripts)
        return text.translate(superscript_map)

    def get_ion_with_charge(self, ion) -> tuple[str, int]:
        # Regex to find all charge patterns
        charge_pattern = r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+[⁺⁻]|[⁺⁻]|\d+[+-]|[+-])"
        matches = re.findall(charge_pattern, ion)

        total_charge = 0
        for pattern in matches:
            # Check if it's superscript with digits
            if re.fullmatch(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+[⁺⁻]", pattern):
                digits_part = pattern[:-1]
                sign = pattern[-1]
                charge = 0
                for c in digits_part:
                    charge = charge * 10 + int(self.superscript_to_num(c))
                total_charge += charge if sign == "⁺" else -charge

            # Check if it's a superscript sign alone
            elif pattern in ("⁺", "⁻"):
                total_charge += 1 if pattern == "⁺" else -1

            # Check if it's regular digits with sign
            elif re.fullmatch(r"\d+[+-]", pattern):
                digits_part = pattern[:-1]
                sign = pattern[-1]
                charge = int(digits_part)
                total_charge += charge if sign == "+" else -charge

            # Check if it's a regular sign alone
            elif pattern in ("+", "-"):
                total_charge += 1 if pattern == "+" else -1

        # Remove all charge patterns to get ion_part
        ion_part = re.sub(charge_pattern, "", ion)
        return (ion_part, total_charge)

    def is_single_element(self, formula: str) -> bool:
        if len(formula) not in (1, 2):
            return False
        if not formula[0].isupper():
            return False
        if len(formula) == 2 and not formula[1].islower():
            return False
        return True

    def format_part(self, formula, divisor) -> str:
        if divisor == 1:
            return formula
        if self.is_single_element(formula):
            return f"{formula}{self.num_to_subscript(str(divisor))}"
        else:
            return f"({formula}){self.num_to_subscript(str(divisor))}"

    def create(self, output_txt_path: str, n_formulas: int) -> None:
        all_formulas = []

        for _ in range(n_formulas):
            # Randomly select a cation and anion
            cation = random.choice(self.cations)
            anion = random.choice(self.anions)

            # Get ion base and charge
            cat_base, cat_charge = self.get_ion_with_charge(cation)
            ani_base, ani_charge = self.get_ion_with_charge(anion)

            # Find greatest common divisor and ions subscripts
            cat_charge = abs(cat_charge)
            ani_charge = abs(ani_charge)
            gcd_val = math.gcd(cat_charge, ani_charge)

            cation_subscript = ani_charge // gcd_val
            anion_subscript = cat_charge // gcd_val

            # Create ions
            cation_part = self.format_part(cat_base, cation_subscript)
            anion_part = self.format_part(ani_base, anion_subscript)
            all_formulas.append(cation_part + anion_part)

        all_formulas = sorted(list(set(all_formulas)))

        # Write formulas to txt
        FileProcessor.write_txt(output_txt_path, lines=all_formulas, newline=True)
