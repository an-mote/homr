import unittest
from unittest.mock import patch

import numpy as np

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import (
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    parse_staffs,
)
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol


def make_staff(number: int, is_grandstaff: bool = False) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    staff = Staff([StaffPoint(0.0, y_points, 0)])
    staff.is_grandstaff = is_grandstaff
    return staff


def make_row(number: int, is_grandstaff: bool = False) -> MultiStaff:
    return MultiStaff([make_staff(number, is_grandstaff)], [])


def _layouts(result: list[MultiStaff]) -> list[list[bool]]:
    return [[s.is_grandstaff for s in row.staffs] for row in result]


class TestStaffParsing(unittest.TestCase):
    def test_ensure_same_number_of_staffs_already_uniform(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))
        self.assertEqual(1, _get_number_of_voices(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern(self) -> None:
        # A solo staff followed by a piano grand staff, repeated for every system,
        # as in a vocal score with piano accompaniment.
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))
        self.assertEqual(2, _get_number_of_voices(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern_grand_staff_first(self) -> None:
        # Same pattern, but the piano grand staff comes first in every system
        # instead of the solo staff, e.g. an instrumental introduction before
        # the voice enters for the rest of the piece.
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i, is_grandstaff=True))
            staffs.append(make_row(i + 1))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, False]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_absorbs_inconsistently_pre_merged_row(self) -> None:
        # A solo-staff-plus-grand-staff pair that an earlier detection step
        # (staffs sharing a bar line) happened to merge into one row, while
        # the same kind of pair elsewhere on the page stayed as two separate
        # rows -- purely because of how cleanly a bar line lined up, not
        # because the underlying layout differs. The pre-merged row must be
        # recognized as fitting the repeating pattern, not trimmed away as an
        # outlier just because its row shape looks different from the rest.
        pre_merged = MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], [])
        staffs = [pre_merged]
        for i in range(2, 8, 2):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_rejects_discarding_leading_staff(self) -> None:
        # A different introduction is real music too; layout alone cannot
        # establish that an edge staff is safe to discard.
        staffs = [make_row(0)] + [make_row(i, is_grandstaff=True) for i in range(1, 5)]

        with self.assertRaisesRegex(ValueError, "1 leading and 0 trailing"):
            _ensure_same_number_of_staffs(staffs)
        self.assertEqual(5, len(staffs))

    def test_ensure_same_number_of_staffs_rejects_discarding_trailing_staff(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)] + [make_row(4)]

        with self.assertRaisesRegex(ValueError, "0 leading and 1 trailing"):
            _ensure_same_number_of_staffs(staffs)
        self.assertEqual(5, len(staffs))

    def test_ensure_same_number_of_staffs_falls_back_to_break_apart(self) -> None:
        # No consistent per-system layout and no clean repeating pattern at
        # any period/trim: the only safe option left is to treat every raw
        # staff as its own system, including exploding rows that an earlier
        # step had bundled together.
        staffs = [
            MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], []),
            MultiStaff([make_staff(2, is_grandstaff=True)], []),
            MultiStaff([make_staff(3), make_staff(4)], []),
            MultiStaff([make_staff(5)], []),
            MultiStaff([make_staff(6, is_grandstaff=True)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(
            [[False], [True], [True], [False], [False], [False], [True]], _layouts(result)
        )
        for row in result:
            self.assertEqual(1, len(row.staffs))

    def test_parse_staffs_rejects_empty_detection_with_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "No staff systems"):
            parse_staffs(None, [], np.zeros((200, 200)), Config())

    def test_parse_staffs_rejects_selection_outside_detected_systems(self) -> None:
        with self.assertRaisesRegex(ValueError, "Selected staff 1.*1 detected"):
            parse_staffs(None, [make_row(0)], np.zeros((200, 200)), Config(), selected_staff=1)

    def test_parse_staffs_does_not_skip_an_unrecognized_middle_system(self) -> None:
        note = EncodedSymbol("note_4", "C4", position="upper")
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch(
                "homr.staff_parsing.parse_staff_image", side_effect=[[note], [], [note]]
            ) as inference,
        ):
            with self.assertRaisesRegex(ValueError, "Staff 1.*system 1.*no notes or rests"):
                parse_staffs(None, [make_row(i) for i in range(3)], np.zeros((400, 200)), Config())
            self.assertEqual(2, inference.call_count)

    def test_parse_staffs_rejects_metadata_without_recognized_music(self) -> None:
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch("homr.staff_parsing.parse_staff_image", return_value=[EncodedSymbol("clef_G2")]),
        ):
            with self.assertRaisesRegex(ValueError, "no notes or rests"):
                parse_staffs(None, [make_row(0)], np.zeros((200, 200)), Config())

    def test_parse_staffs_preserves_rest_only_result(self) -> None:
        rest = EncodedSymbol("rest_1", position="upper")
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch("homr.staff_parsing.parse_staff_image", return_value=[rest]),
        ):
            result = parse_staffs(None, [make_row(0)], np.zeros((200, 200)), Config())
        self.assertEqual(["rest_1", "newline"], [symbol.rhythm for symbol in result[0]])

    def test_parse_staffs_preserves_multimeasure_rest_result(self) -> None:
        rest = EncodedSymbol("rest_4m", position="upper")
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch("homr.staff_parsing.parse_staff_image", return_value=[rest]),
        ):
            result = parse_staffs(None, [make_row(0)], np.zeros((200, 200)), Config())
        self.assertEqual(["rest_4m", "newline"], [symbol.rhythm for symbol in result[0]])

    def test_selected_staff_uses_system_index_for_every_voice(self) -> None:
        systems = [MultiStaff([make_staff(2 * i), make_staff(2 * i + 1)], []) for i in range(2)]
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch(
                "homr.staff_parsing.parse_staff_image",
                side_effect=lambda debug, index, *args: [EncodedSymbol("rest_1", position="upper")],
            ) as inference,
        ):
            result = parse_staffs(None, systems, np.zeros((400, 200)), Config(), selected_staff=1)
        self.assertEqual([1, 3], [call.args[1] for call in inference.call_args_list])
        self.assertEqual(2, len(result))
        self.assertTrue(all(voice[0].rhythm == "rest_1" for voice in result))

    def test_parse_staffs_adds_staff_location_to_inference_failure(self) -> None:
        with (
            patch("homr.staff_parsing.StaffRegions"),
            patch(
                "homr.staff_parsing.parse_staff_image", side_effect=RuntimeError("decoder failed")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Staff 0.*system 0.*decoder failed"):
                parse_staffs(None, [make_row(0)], np.zeros((200, 200)), Config())
