"""Tests for reading where Label Studio says an image is.

Two URI shapes and three column names, none of them ours. They moved here with
the code: the parsing used to sit in `domain` because the dataset builder read
exports directly, and it does not any more.
"""

from __future__ import annotations

import pytest

from digitex.labeling.uris import local_file_path, task_image_path


class TestLocalFilePath:
    @pytest.mark.parametrize(
        ("uri", "name"),
        [
            ("/data/local-files/?d=training%5Cdata%5Cpage.jpg", "page.jpg"),
            ("/data/local-files/?file=training/data/page.jpg", "page.jpg"),
            ("/data/local-files/?d=images%5Cmy%20file.jpg", "my file.jpg"),
        ],
        ids=["d-parameter", "file-parameter", "url-encoded-space"],
    )
    def test_the_filename_is_recovered_from_either_parameter(
        self, uri: str, name: str
    ) -> None:
        path = local_file_path(uri)

        assert path is not None
        assert path.name == name

    def test_a_backslash_uri_splits_on_every_platform(self) -> None:
        """The separators are the Label Studio host's, not this machine's.

        Asserting ``.name`` alone passed on Windows while the whole URI stayed
        one filename on Linux, which is how this reached CI unnoticed.
        """
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cpage.jpg"

        path = local_file_path(uri)

        assert path is not None
        assert path.parts == ("training", "data", "images", "page.jpg")

    @pytest.mark.parametrize(
        "uri",
        ["", "http://example.com/image.jpg", "/data/local-files/?other=x"],
        ids=["empty", "remote-url", "no-local-file-parameter"],
    )
    def test_a_uri_naming_no_local_file_has_no_path(self, uri: str) -> None:
        """The predictor skips such a task rather than failing the run."""
        assert local_file_path(uri) is None


class TestTaskImagePath:
    """A task's image is not always filed under ``image``.

    A sync from a storage of blob URLs files it under ``$undefined$``, and a
    reader that only knows ``image`` skips every task of such a project without
    saying why — which is how a prediction run over 1069 tasks predicted none.
    """

    URI = "/data/local-files/?d=var%5Ctraining%5Cpage.jpg"

    @pytest.mark.parametrize("key", ["image", "$undefined$"], ids=["named", "unnamed"])
    def test_the_image_is_found_under_either_key(self, key: str) -> None:
        path = task_image_path({key: self.URI})

        assert path is not None
        assert path.name == "page.jpg"

    def test_the_key_the_label_config_names_wins(self) -> None:
        """Both keys present means one of them is a leftover; ``image`` is not."""
        path = task_image_path(
            {"$undefined$": "/data/local-files/?d=stale.jpg", "image": self.URI}
        )

        assert path is not None
        assert path.name == "page.jpg"

    @pytest.mark.parametrize(
        "data",
        [{}, {"image": ""}, {"text": "a question"}, {"image": 7}],
        ids=["empty", "empty-uri", "no-image-field", "not-a-string"],
    )
    def test_a_task_with_no_local_file_has_no_path(self, data: dict) -> None:
        assert task_image_path(data) is None
