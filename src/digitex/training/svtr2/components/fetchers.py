import os
import re
import requests
from json import JSONDecodeError

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor


class PubChemFetcher:
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

    def _get_response_json(self, url: str) -> dict | None:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}")
        except JSONDecodeError:
            print(f"Failed to decode JSON from {url}.")
        return None

    def _mf_to_unicode(self, mf: str) -> str:
        # Convert numbers to subscripts, charges to superscripts
        sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        super_map = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

        # Replace numbers after element symbols with subscripts
        def subscript_numbers(match):
            return match.group(1) + match.group(2).translate(sub_map)

        mf = re.sub(r"([A-Za-z\)])(\d+)", subscript_numbers, mf)

        # Replace charge at the end (e.g., 2+, 3-, +, -, 2+5, 4-7) with superscript
        mf = re.sub(
            r"^(.*?)([+-]\d*|\d+[+-]\d*)$",
            lambda m: m.group(1) + m.group(2).translate(super_map),
            mf,
        )
        return mf

    def _categorize_mf(self, mf: str) -> str:
        # Organic: contains C and H, possibly with O, N, S, etc.
        # Ion: ends with + or -
        is_ion = bool(re.search(r"[+-]\d*$|\d+[+-]$", mf))
        is_organic = bool(re.match(r"C\d*H\d*", mf))
        if is_organic and is_ion:
            return "org_ions"
        elif is_organic:
            return "org_formulas"
        elif is_ion:
            return "in_ions"
        else:
            return "in_formulas"

    def create_element_cids_txt(self, output_path: str) -> None:
        cids = []
        for name in self.element_names:
            # Get json data with the CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/Title/json"
            data = self._get_response_json(url)
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
            data = self._get_response_json(url)
            if data is None:
                continue

            # Extract the substructure CIDs from the json and add them to the set
            cids = data["IdentifierList"]["CID"]
            cids = list(map(str, cids))
            substructure_cids.update(cids)

        substructure_cids = sorted(substructure_cids)
        FileProcessor.write_txt(output_path, substructure_cids, newline=True)

    def create_all_mfs_txt(
        self,
        substructure_cids_txt_path: str,
        output_path: str,
        start_idx: int = 0,
        batch_size: int = 10,
    ) -> None:
        # Read the substructure CIDs from the file
        substructure_cids = FileProcessor.read_txt(
            substructure_cids_txt_path, strip=True
        )
        substructure_cids = substructure_cids[start_idx:]

        # Open the output file in append mode
        with open(output_path, "a") as output_file:
            for i in tqdm(
                range(0, len(substructure_cids), batch_size),
                desc="Fetching molecular formulas",
                unit="batch",
            ):
                # Get a batch of CIDs
                batch_cids = substructure_cids[i : i + batch_size]
                cids_str = ",".join(batch_cids)

                # Get json data with the molecular formulas
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/MolecularFormula/json"

                data = self._get_response_json(url)
                if data is None:
                    continue

                # Extract the molecular formulas from the json and write them to the file
                for prop in data["PropertyTable"]["Properties"]:
                    output_file.write(f"{prop['MolecularFormula']}\n")
