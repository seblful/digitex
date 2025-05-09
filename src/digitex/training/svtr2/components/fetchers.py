import requests

from digitex.core.processors.file import FileProcessor


class PubChemFetcher:
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/oxygen/property/Title/json"

    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/cid/5359268/cids/json"
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/5359268,9989226/property/MolecularFormula/json"

    def __init__(self, elements_info_json: str) -> None:
        self.__elements_info_json = elements_info_json

        self.__symbols2names = None
        self.__element_symbols = None
        self.__element_names = None

    @property
    def symbols2names(self) -> dict[str, str]:
        if self.__symbols2names is not None:
            return self.__symbols2names

        # Parse the JSON string
        data = FileProcessor.read_json(self.__elements_info_json)
        columns = data["Table"]["Columns"]["Column"]
        rows = data["Table"]["Row"]

        # Find the indices for Symbol and Name
        symbol_idx = columns.index("Symbol")
        name_idx = columns.index("Name")

        # Build the mapping
        symbols2names = {}
        for row in rows:
            cell = row["Cell"]
            symbol = cell[symbol_idx]
            name = cell[name_idx]
            symbols2names[symbol] = name

        self.__symbols2names = symbols2names
        return self.__symbols2names

    @property
    def element_symbols(self) -> list[str]:
        if self.__element_symbols is None:
            self.__element_symbols = list(self.symbols2names.keys())

        return self.__element_symbols

    @property
    def element_names(self) -> list[str]:
        if self.__element_names is None:
            self.__element_names = list(self.symbols2names.values())

        return self.__element_names

    def create_elements_to_cid(self, elements_json: str):
        pass
