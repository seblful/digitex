from digitex.core.processors.file import FileProcessor


def remove_lines_with_invalid_chars(input_txt_path, chars_path) -> None:
    allowed_chars = set("".join(FileProcessor.read_txt(chars_path)).strip())
    lines = FileProcessor.read_txt(input_txt_path, strip=False)
    valid_lines = [
        line for line in lines if all(c in allowed_chars or c == "\n" for c in line)
    ]
    FileProcessor.write_txt(input_txt_path, valid_lines, newline=False)


def print_lines_with_invalid_chars(input_txt_path, chars_path) -> None:
    allowed_chars = set("".join(FileProcessor.read_txt(chars_path)).strip())
    for line in FileProcessor.read_txt(input_txt_path, strip=False):
        line_stripped = line.rstrip("\n")
        if any(char not in allowed_chars for char in line_stripped):
            print(line_stripped)
