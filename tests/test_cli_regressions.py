import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

from homr.main import (
    ProcessingConfig,
    _write_xml_atomically,
    main,
    process_image,
    run_homr,
)
from homr.music_xml_generator import XmlGeneratorArguments


class TestCliRegressions(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProcessingConfig(False, False, False, False, -1, False, False, False, False)
        self.xml_args = XmlGeneratorArguments(False, None, None)

    def test_failed_image_propagates_exception(self) -> None:
        with patch("homr.main.process_image", side_effect=AssertionError("bad duration")):
            with self.assertRaisesRegex(AssertionError, "bad duration"):
                run_homr(["page.png"], self.config, self.xml_args)

    def test_empty_image_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No images"):
            run_homr([], self.config, self.xml_args)

    def test_failed_regeneration_preserves_previous_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "page.png"
            prior = image.with_suffix(".musicxml")
            prior.write_text("previous checked result")
            with patch("homr.main.detect_staffs_in_image", side_effect=ValueError("unreadable")):
                with self.assertRaises(ValueError):
                    process_image(str(image), self.config, self.xml_args)
            self.assertEqual(prior.read_text(), "previous checked result")

    def test_serialization_failure_keeps_existing_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "page.musicxml"
            output.write_text("previous")
            invalid = ET.Element("score")
            cast(Any, invalid).text = object()
            with self.assertRaises(TypeError):
                _write_xml_atomically(invalid, str(output))
            self.assertEqual(output.read_text(), "previous")
            self.assertEqual(list(Path(temporary).iterdir()), [output])

    def test_atomic_write_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "page.musicxml"
            _write_xml_atomically(ET.Element("score-partwise"), str(output))
            root = ET.parse(output).getroot()  # noqa: S314 - trusted, locally generated test XML
            self.assertEqual(root.tag, "score-partwise")

    def test_merge_writes_beside_input_and_retains_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            images = [str(Path(temporary) / name) for name in ("one.png", "two.png")]
            pages = [str(Path(temporary) / name) for name in ("one.musicxml", "two.musicxml")]
            for page in pages:
                Path(page).write_text("original page")
            with (
                patch("homr.main.process_image", side_effect=pages),
                patch(
                    "homr.main.process_concat",
                    return_value=(ET.Element("score-partwise"), None, None),
                ),
            ):
                run_homr(images, self.config, self.xml_args)
            self.assertTrue((Path(temporary) / "one_merged.musicxml").is_file())
            self.assertTrue(all(Path(p).is_file() for p in pages))

    def test_merge_name_collision_preserves_every_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            names = ("one", "one_merged", "one_merged_2")
            images = [str(Path(temporary) / f"{name}.png") for name in names]
            pages = [str(Path(temporary) / f"{name}.musicxml") for name in names]
            originals = [f"original page {index}" for index in range(len(pages))]
            for page, content in zip(pages, originals, strict=True):
                Path(page).write_text(content)
            with (
                patch("homr.main.process_image", side_effect=pages),
                patch(
                    "homr.main.process_concat",
                    return_value=(ET.Element("score-partwise"), None, None),
                ),
            ):
                run_homr(images, self.config, self.xml_args)
            self.assertEqual([Path(page).read_text() for page in pages], originals)
            merged = Path(temporary) / "one_merged_3.musicxml"
            root = ET.parse(merged).getroot()  # noqa: S314 - trusted, locally generated test XML
            self.assertEqual(root.tag, "score-partwise")

    def test_title_failure_does_not_discard_music(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = str(Path(temporary) / "page.png")
            title: Future[str] = Future()
            title.set_exception(ValueError("empty OCR region"))
            debug = Mock()
            with (
                patch(
                    "homr.main.detect_staffs_in_image",
                    return_value=([], None, debug, title, None, None),
                ),
                patch("homr.main.parse_staffs", return_value=[Mock()]),
                patch(
                    "homr.main.generate_xml", return_value=ET.Element("score-partwise")
                ) as generator,
            ):
                output = process_image(image, self.config, self.xml_args)
            self.assertTrue(Path(output).is_file())
            self.assertEqual(generator.call_args.args[2], "")

    def test_directory_partial_failure_returns_nonzero_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            files = [str(Path(temporary) / "bad.png"), str(Path(temporary) / "good.png")]
            processor = Mock(side_effect=[ValueError("bad"), "good.musicxml"])
            with (
                patch.object(sys, "argv", ["homr", temporary, "--gpu", "no", "--no-title"]),
                patch("homr.main.download_weights"),
                patch("homr.main.get_all_image_files_in_folder", return_value=files),
                patch("homr.main.process_image", processor),
            ):
                with self.assertRaises(SystemExit) as exit_result:
                    main()
            self.assertEqual(exit_result.exception.code, 1)
            self.assertEqual(processor.call_count, 2)


if __name__ == "__main__":
    unittest.main()
