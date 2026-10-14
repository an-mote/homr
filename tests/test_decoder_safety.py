import unittest
from types import SimpleNamespace

import numpy as np

from homr.transformer.decoder_inference import ScoreDecoder
from homr.transformer.vocabulary import EncodedSymbol


class FakeBinding:
    def __init__(self, steps: list[list[np.ndarray]]) -> None:
        self.steps = steps
        self.index = -1

    def bind_cpu_input(self, *args: object) -> None:
        pass

    def bind_ortvalue_input(self, *args: object) -> None:
        pass

    def bind_output(self, *args: object) -> None:
        pass

    def get_outputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(numpy=lambda value=value: value) for value in self.steps[self.index]
        ]


class FakeSession:
    def __init__(self, steps: list[list[np.ndarray]]) -> None:
        self.binding = FakeBinding(steps)

    def io_binding(self) -> FakeBinding:
        return self.binding

    def run_with_iobinding(self, iobinding: FakeBinding) -> None:
        iobinding.index += 1


def outputs(rhythm: int = 0) -> list[np.ndarray]:
    rhythm_logits = np.zeros((1, 1, 3), dtype=np.float32)
    rhythm_logits[0, 0, rhythm] = 1
    return [rhythm_logits] + [np.ones((1, 1, 1), dtype=np.float32) for _ in range(6)]


def make_decoder(steps: list[list[np.ndarray]]) -> ScoreDecoder:
    config = SimpleNamespace(
        max_seq_len=len(steps),
        eos_token=2,
        rhythm_vocab={"note_4": 0, "<s>": 1, "</s>": 2},
        pitch_vocab={"C4": 0},
        lift_vocab={"_": 0},
        articulation_vocab={"_": 0},
        slur_vocab={"_": 0},
        position_vocab={"upper": 0},
        decoder_depth=0,
        decoder_heads=1,
        decoder_dim=1,
    )
    return ScoreDecoder(FakeSession(steps), False, False, config)


def generate(decoder: ScoreDecoder) -> list[EncodedSymbol]:
    return decoder.generate(
        np.array([[1]], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        context=np.zeros((1, 2, 1), dtype=np.float32),
    )


class TestDecoderSafety(unittest.TestCase):
    def test_eos_at_limit_preserves_completed_prediction(self) -> None:
        result = generate(make_decoder([outputs(), outputs(2)]))
        self.assertEqual(["note_4"], [symbol.rhythm for symbol in result])

    def test_sequence_limit_is_not_reported_as_completed_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "end token.*2.*incomplete"):
            generate(make_decoder([outputs(), outputs()]))

    def test_nonfinite_logits_identify_head_and_step(self) -> None:
        for head, name in enumerate(
            ("rhythm", "pitch", "lift", "position", "articulation", "slur", "attention")
        ):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(head=name, invalid=invalid):
                    step = outputs()
                    step[head][0, 0, 0] = invalid
                    with self.assertRaisesRegex(ValueError, f"{name}.*step 0.*non-finite"):
                        generate(make_decoder([step, outputs(2)]))

    def test_immediate_eos_is_empty_and_not_truncated(self) -> None:
        self.assertEqual([], generate(make_decoder([outputs(2)])))
