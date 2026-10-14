"""Runtime-only regressions for preserving recognized music during XML export."""

import unittest
from fractions import Fraction
from xml.etree import ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol


def tokens(lines: str) -> list[EncodedSymbol]:
    result: list[EncodedSymbol] = []
    for line in lines.strip().splitlines():
        for index, entry in enumerate(line.split("&")):
            if index:
                result.append(EncodedSymbol("chord"))
            rhythm, pitch, lift, articulation, slur, position = entry.strip().split()
            result.append(EncodedSymbol(rhythm, pitch, lift, articulation, slur, position))
    return result


def export(lines: str) -> list[ET.Element]:
    root = generate_xml(XmlGeneratorArguments(), [tokens(lines)], "")
    return root.findall("part/measure")


def require_element(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise AssertionError(f"Missing expected XML element: {path}")
    return element


def onsets(measure: ET.Element) -> list[tuple[str, Fraction]]:
    division = int(measure.findtext("attributes/divisions", "1"))
    cursor, previous = Fraction(0), Fraction(0)
    result = []
    for child in measure:
        duration = Fraction(int(child.findtext("duration", "0")), division)
        if child.tag == "backup":
            cursor -= duration
        elif child.tag == "forward":
            cursor += duration
        elif child.tag == "note":
            start = previous if child.find("chord") is not None else cursor
            if child.find("chord") is None:
                previous = cursor
                cursor += duration
            if child.find("pitch") is not None:
                name = child.findtext("pitch/step", "") + child.findtext("pitch/octave", "")
                result.append((name, start))
    return result


class TestMusicXmlRegressions(unittest.TestCase):
    def test_pure_multimeasure_rest_expands_count_and_full_duration(self) -> None:
        measures = export("""
rest_3m _ _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 3)
        self.assertEqual(measures[0].findtext("attributes/measure-style/multiple-rest", ""), "3")
        for measure in measures:
            note = require_element(measure, "note")
            self.assertEqual(require_element(note, "rest").get("measure"), "yes")
            self.assertEqual(note.findtext("duration", ""), "4")

    def test_multimeasure_rest_keeps_surrounding_music_and_inferred_meter(self) -> None:
        measures = export("""
timeSignature/4 . . . . .
note_2. C4 _ _ _ upper
barline . . . . .
rest_3m _ _ _ _ upper
barline . . . . .
note_2. D4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 5)
        self.assertEqual(measures[0].findtext("attributes/time/beats", ""), "3")
        self.assertEqual(measures[0].findtext("note/pitch/step", ""), "C")
        self.assertEqual(measures[4].findtext("note/pitch/step", ""), "D")
        for measure in measures[1:4]:
            self.assertEqual(measure.findtext("note/duration", ""), "3")

    def test_grandstaff_multimeasure_rests_share_each_measure_time(self) -> None:
        measures = export("""
clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
rest_2m _ _ _ _ upper&rest_2m _ _ _ _ lower
barline . . . . .
""")
        self.assertEqual(len(measures), 2)
        for measure in measures:
            self.assertEqual([n.findtext("staff", "") for n in measure.findall("note")], ["1", "2"])
            self.assertEqual(measure.findtext("backup/duration", ""), "4")

    def test_inconsistent_multimeasure_rest_reports_location(self) -> None:
        with self.assertRaisesRegex(ValueError, "measure 1") as error:
            export("rest_2m _ _ _ _ upper&rest_3m _ _ _ _ lower")
        self.assertIn("rest_2m", str(error.exception))
        self.assertIn("rest_3m", str(error.exception))

    def test_repeat_end_after_barline_belongs_to_previous_music(self) -> None:
        measures = export("""
note_1 C4 _ _ _ upper
barline . . . . .
repeatEnd . . . . .
note_1 D4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 2)
        self.assertEqual(
            require_element(measures[0], "barline/repeat").get("direction"), "backward"
        )
        self.assertIsNone(measures[1].find("barline/repeat"))

    def test_multimeasure_rest_and_repeat_end_preserve_final_boundary(self) -> None:
        measures = export("""
rest_2m _ _ _ _ upper
barline . . . . .
repeatEndStart . . . . .
note_1 D4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 3)
        self.assertEqual(
            require_element(measures[1], "barline/repeat").get("direction"), "backward"
        )
        self.assertEqual(require_element(measures[2], "barline/repeat").get("direction"), "forward")
        self.assertEqual(require_element(measures[2], "barline").get("location"), "left")

    def test_invalid_boundary_in_note_group_does_not_shift_later_measures(self) -> None:
        measures = export("""
note_4 C4 _ _ _ upper
note_8 G4 _ _ _ upper&barline C5 _ _ _ upper
barline . . . . .
note_4 D4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 2)
        self.assertEqual(
            [n.findtext("pitch/step", "") for n in measures[0].findall("note")], ["C", "G"]
        )
        unresolved = measures[0].findtext("direction/direction-type/words", "")
        self.assertIn("barline C5", unresolved)
        self.assertEqual(measures[1].findtext("note/pitch/step", ""), "D")

    def test_standalone_boundary_with_extra_branch_fields_keeps_boundary(self) -> None:
        measures = export("""
note_1 C4 _ _ _ upper
barline F4 . . . .
note_1 D4 _ _ _ upper
barline . _ _ _ upper
note_1 E4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 3)
        self.assertEqual([m.findtext("note/pitch/step", "") for m in measures], ["C", "D", "E"])
        self.assertIn("barline F4", measures[0].findtext("direction/direction-type/words", ""))
        self.assertIn("barline . _", measures[1].findtext("direction/direction-type/words", ""))

    def test_trailing_invalid_control_preserves_original_location_without_empty_measure(
        self,
    ) -> None:
        for control, position, staff in [("barline", "upper", "1"), ("clef_G2", "lower", "2")]:
            with self.subTest(control=control):
                voice = tokens(
                    f"note_4 C4 _ _ _ upper\nbarline . . . . .\n" f"{control} C5 . . . {position}"
                )
                voice[-1].image_coordinates = (123.4, 456.7)
                root = generate_xml(XmlGeneratorArguments(), [voice], "")
                measures = root.findall("part/measure")
                self.assertEqual(len(measures), 1)
                self.assertEqual(measures[0].findtext("note/pitch/step", ""), "C")
                directions = measures[0].findall("direction")
                self.assertEqual(len(directions), 1)
                direction = directions[0]
                words = direction.findtext("direction-type/words", "")
                self.assertIn(f"{control} C5", words)
                self.assertIn("原始识别小节 2", words)
                self.assertEqual(direction.findtext("staff", ""), staff)
                self.assertIn("imgpos: 123, 457", ET.tostring(direction, encoding="unicode"))

    def test_consecutive_trailing_invalid_controls_are_each_retained_once(self) -> None:
        measures = export("""
note_4 C4 _ _ _ upper
barline . . . . .
barline C5 . . . upper
clef_G2 D5 . . . lower
barline . . . . .
""")
        self.assertEqual(len(measures), 1)
        words = measures[0].findall("direction/direction-type/words")
        self.assertEqual(len(words), 2)
        self.assertIn("barline C5", (words[0].text or ""))
        self.assertIn("clef_G2 D5", (words[1].text or ""))
        self.assertTrue(all("原始识别小节 2" in (word.text or "") for word in words))

    def test_input_without_music_fails_with_unresolved_location(self) -> None:
        for control in ["barline", "clef_G2"]:
            with self.subTest(control=control):
                voice = tokens(f"{control} C5 . . . lower")
                voice[-1].image_coordinates = (123.4, 456.7)
                with self.assertRaisesRegex(ValueError, "No musical content") as error:
                    generate_xml(XmlGeneratorArguments(), [voice], "")
                message = str(error.exception)
                self.assertIn("measure 1", message)
                self.assertIn(f"{control} C5", message)
                self.assertIn("lower", message)
                self.assertIn("123.4", message)
                self.assertIn("456.7", message)
        with self.assertRaisesRegex(ValueError, "No musical content"):
            export("clef_G2 _ _ _ _ upper\nbarline . . . . .")

    def test_unresolved_control_in_triplet_preserves_tuplet_start_stop(self) -> None:
        voice = tokens("""
note_12 C4 _ _ _ upper
note_12 D4 _ _ _ upper
barline C5 . . . upper&note_12 E4 _ _ _ upper
barline . . . . .
""")
        voice[2].image_coordinates = (123.4, 456.7)
        root = generate_xml(XmlGeneratorArguments(), [voice], "")
        measures = root.findall("part/measure")
        self.assertEqual(len(measures), 1)
        notes = measures[0].findall("note")
        self.assertEqual([n.findtext("pitch/step", "") for n in notes], ["C", "D", "E"])
        marks = [[tuplet.get("type") for tuplet in n.iter("tuplet")] for n in notes]
        self.assertEqual(marks, [["start"], [], ["stop"]])
        for note in notes:
            self.assertEqual(note.findtext("time-modification/actual-notes", ""), "3")
            self.assertEqual(note.findtext("time-modification/normal-notes", ""), "2")
        direction = require_element(measures[0], "direction")
        self.assertIn("barline C5", direction.findtext("direction-type/words", ""))
        self.assertEqual(direction.findtext("staff", ""), "1")
        self.assertIn("imgpos: 123, 457", ET.tostring(direction, encoding="unicode"))

    def test_clef_in_note_group_preserves_notes_and_attribute(self) -> None:
        measures = export("""
clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
note_8 E4 # _ _ upper&clef_G2 _ _ _ _ lower&note_8 C5 _ _ _ lower
note_8 F4 # _ _ upper&note_8 D5 _ _ _ lower
barline . . . . .
""")
        self.assertEqual(len(measures), 1)
        self.assertEqual(len(measures[0].findall("note")), 4)
        self.assertEqual(
            onsets(measures[0]),
            [
                ("E4", Fraction(0)),
                ("C5", Fraction(0)),
                ("F4", Fraction(1, 2)),
                ("D5", Fraction(1, 2)),
            ],
        )
        self.assertEqual(measures[0].findall("attributes/clef")[-1].findtext("sign", ""), "G")

    def test_leading_repeat_is_left_barline_without_empty_measure(self) -> None:
        measures = export("""
clef_G2 _ _ _ _ upper
keySignature_0 . . . . .
repeatStart . . . . .
timeSignature/4 . . . . .
note_1 C4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(len(measures), 1)
        repeat = require_element(measures[0], "barline/repeat")
        self.assertIsNotNone(repeat)
        self.assertEqual(require_element(measures[0], "barline").get("location"), "left")
        self.assertEqual(repeat.get("direction"), "forward")

    def test_repeat_ending_stays_with_music_before_following_barline(self) -> None:
        measures = export("""
voltaStart . . . . .
note_1 C4 _ _ _ upper
repeatEnd . . . . .
voltaStop . . . . .
barline . . . . .
voltaStart . . . . .
note_1 D4 _ _ _ upper
voltaDiscontinue . . . . .
barline . . . . .
newline . . . . .
""")
        self.assertEqual(len(measures), 2)
        endings = [m.findall("barline/ending") for m in measures]
        self.assertEqual(
            [(e.get("type"), e.get("number")) for e in endings[0]], [("start", "1"), ("stop", "1")]
        )
        self.assertEqual(
            [(e.get("type"), e.get("number")) for e in endings[1]],
            [("start", "2"), ("discontinue", "2")],
        )

    def test_meter_uses_all_sounding_notes(self) -> None:
        measures = export("""
clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_4 E3 _ _ _ lower
note_4 D5 _ _ _ upper
note_8 G3 _ _ _ lower
note_2 E5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .
""")
        self.assertEqual(measures[0].findall("attributes/time")[-1].findtext("beats", ""), "4")

    def test_meter_change_infers_each_region_separately(self) -> None:
        measures = export("""
timeSignature/4 . . . . .
note_1 C4 _ _ _ upper
barline . . . . .
note_1 C4 _ _ _ upper
barline . . . . .
note_1 C4 _ _ _ upper
barline . . . . .
timeSignature/4 . . . . .
note_2. D4 _ _ _ upper
barline . . . . .
note_2. D4 _ _ _ upper
barline . . . . .
""")
        self.assertEqual(measures[0].findall("attributes/time")[-1].findtext("beats", ""), "4")
        self.assertEqual(measures[3].findtext("attributes/time/beats", ""), "3")

    def test_tuplet_normal_type_and_dots_are_preserved(self) -> None:
        measures = export("""
note_12. C4 _ _ _ upper
note_12. D4 _ _ _ upper
note_12. E4 _ _ _ upper
barline . . . . .
""")
        for note in measures[0].findall("note"):
            modification = require_element(note, "time-modification")
            self.assertEqual(modification.findtext("normal-type", ""), "eighth")
            self.assertEqual(len(modification.findall("normal-dot")), 1)


if __name__ == "__main__":
    unittest.main()
