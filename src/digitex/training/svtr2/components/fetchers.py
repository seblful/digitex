import requests

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor


class PubChemFetcher:
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/cid/5359268/cids/json"

    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/5359268,9989226/property/MolecularFormula/json"

    def __init__(self, elements_info_json_path: str) -> None:
        self.__elements_info_json_path = elements_info_json_path

        self.__symbols2names = None
        self.__element_symbols = None
        self.__element_names = None

    @property
    def symbols2names(self) -> dict[str, str]:
        if self.__symbols2names is not None:
            return self.__symbols2names

        # Parse the JSON string
        data = FileProcessor.read_json(self.__elements_info_json_path)
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

    def get_response_json(self, url: str) -> dict | None:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data from {url}.")
            return None

    def create_element_cids_txt(self, output_path: str) -> None:
        cids = []
        for name in self.element_names:
            # Get json data with the CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/Title/json"
            data = self.get_response_json(url)
            if data is None:
                continue

            # Extract the CID from the json and add it to the list
            cid = data["PropertyTable"]["Properties"][0]["CID"]
            cids.append(str(cid))

        FileProcessor.write_txt(output_path, cids, newline=True)

    def create_substructure_cids_txt(
        self, element_cids_txt_path: str, output_path: str
    ) -> None:
        # Read the element CIDs from the file
        element_cids = FileProcessor.read_txt(element_cids_txt_path, strip=True)

        # Create a list to store the substructure CIDs
        substructure_cids = set()

        for cid in tqdm(element_cids, desc="Fetching substructure CIDs", unit="cid"):
            # Get json data with the substructure CIDs
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/cid/{cid}/cids/json"
            data = self.get_response_json(url)
            if data is None:
                continue

            # Extract the substructure CIDs from the json and add them to the set
            cids = data["IdentifierList"]["CID"]
            cids = list(map(str, cids))
            substructure_cids.update(cids)

        substructure_cids = sorted(substructure_cids)
        FileProcessor.write_txt(output_path, substructure_cids, newline=True)
