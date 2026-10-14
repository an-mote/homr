"""连音清理必须保留多小节休止符数量。"""

import json
import unittest
from unittest.mock import patch

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import (
    EncodedSymbol,
    _fix_over_eager_tuplets,
    _get_duration_of_measure,
)


class TestVocabularyRegressions(unittest.TestCase):
    def test_multimeasure_rest_count_is_never_a_tuplet_denominator(self) -> None:
        for count in range(2, 11):
            with self.subTest(count=count):
                symbol = EncodedSymbol(f"rest_{count}m", position="upper")
                self.assertFalse(symbol.is_tuplet())
                cleaned = symbol.remove_tuplet()
                self.assertEqual(f"rest_{count}m", cleaned.rhythm)
                self.assertEqual(symbol.get_duration(), cleaned.get_duration())

    def test_short_multirest_placeholder_survives_full_measure_cleanup(self) -> None:
        for count in range(2, 11):
            with self.subTest(count=count):
                multirest = [
                    [EncodedSymbol(f"rest_{count}m", position="upper")],
                    [EncodedSymbol("barline")],
                ]
                long_measure = [
                    [EncodedSymbol("note_2.", "C4", position="upper")],
                    [EncodedSymbol("note_2.", "D4", position="upper")],
                    [EncodedSymbol("barline")],
                ]
                self.assertLess(
                    _get_duration_of_measure(multirest), _get_duration_of_measure(long_measure)
                )
                result = _fix_over_eager_tuplets(multirest + long_measure + long_measure)
                self.assertEqual(f"rest_{count}m", result[0][0].rhythm)

    def test_normal_tuplet_cleanup_still_converts_notes_and_rests(self) -> None:
        for prefix in ("note", "rest"):
            for original, expected in ((12, 8), (24, 16), (20, 16), (28, 16)):
                with self.subTest(prefix=prefix, original=original):
                    symbol = EncodedSymbol(f"{prefix}_{original}", "C4", position="upper")
                    self.assertTrue(symbol.is_tuplet())
                    self.assertEqual(f"{prefix}_{expected}", symbol.remove_tuplet().rhythm)
                    self.assertEqual(f"{prefix}_{original}", symbol.rhythm)

    def test_regular_rhythms_and_controls_are_preserved(self) -> None:
        for rhythm in ("note_4", "note_8.", "rest_0", "rest_4", "clef_G2", "barline"):
            with self.subTest(rhythm=rhythm):
                symbol = EncodedSymbol(rhythm, "C4", position="upper")
                self.assertFalse(symbol.is_tuplet())
                self.assertEqual(rhythm, symbol.remove_tuplet().rhythm)

    def test_multimeasure_rests_never_generate_time_modification(self) -> None:
        for count in range(2, 11):
            with self.subTest(count=count):
                symbols = [
                    EncodedSymbol(f"rest_{count}m", position="upper"),
                    EncodedSymbol("barline"),
                ]
                xml = generate_xml(XmlGeneratorArguments(), [symbols], "")
                self.assertEqual(count, len(xml.findall("part/measure")))
                self.assertEqual([], xml.findall(".//time-modification"))
                self.assertEqual([], xml.findall(".//tuplet"))

    def test_tuplet_cleanup_noop_does_not_log_removing_music(self) -> None:
        short = [[EncodedSymbol("note_4", "C4", position="upper")], [EncodedSymbol("barline")]]
        full = [[EncodedSymbol("note_1", "D4", position="upper")], [EncodedSymbol("barline")]]
        with patch("homr.transformer.vocabulary.eprint") as logger:
            result = _fix_over_eager_tuplets(short + full + full)
        logger.assert_not_called()
        self.assertEqual("note_4", result[0][0].rhythm)

    def test_tuplet_cleanup_log_preserves_original_tokens_and_locations(self) -> None:
        note = EncodedSymbol("note_12", "C4", "#", slur="slurStart", position="upper")
        note.image_coordinates = (10.5, 20.25)
        partner = EncodedSymbol("note_24", "E3", position="lower")
        short = [
            [EncodedSymbol("clef_G2", position="upper")],
            [note, partner],
            [EncodedSymbol("barline")],
        ]
        full = [[EncodedSymbol("note_1", "D4", position="upper")], [EncodedSymbol("barline")]]
        with patch("homr.transformer.vocabulary.eprint") as logger:
            result = _fix_over_eager_tuplets(short + full + full)
        logger.assert_called_once()
        self.assertEqual("Tuplet cleanup changes:", logger.call_args.args[0])
        changes = json.loads(logger.call_args.args[1])
        self.assertEqual(2, len(changes))
        self.assertEqual(
            {
                "measure": 1,
                "chord_index": 1,
                "symbol_index": 0,
                "before": str(note),
                "after": str(result[1][0]),
                "position": "upper",
                "image_coordinates": [10.5, 20.25],
            },
            changes[0],
        )
        self.assertEqual(1, changes[1]["symbol_index"])
        self.assertIsNone(changes[1]["image_coordinates"])
        self.assertEqual("note_12", note.rhythm)
        self.assertEqual("note_8", result[1][0].rhythm)
        self.assertEqual("note_16", result[1][1].rhythm)
        recovered = [chord.copy() for chord in result]
        for change in changes:
            fields = change["before"].split()
            recovered[change["chord_index"]][change["symbol_index"]] = EncodedSymbol(*fields)
        self.assertEqual(
            [[str(symbol) for symbol in chord] for chord in short],
            [[str(symbol) for symbol in chord] for chord in recovered[: len(short)]],
        )
