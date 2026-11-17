from __future__ import annotations

import pytest

from vigia.data.pyrosdis import yolo_to_xyxy


def test_single_box_exact() -> None:
    assert yolo_to_xyxy("0 0.5 0.5 0.2 0.4") == [pytest.approx([0.4, 0.3, 0.6, 0.7])]


def test_multiple_boxes_keep_order() -> None:
    text = "0 0.25 0.25 0.1 0.1\n0 0.75 0.5 0.5 0.2\n"
    assert yolo_to_xyxy(text) == [
        pytest.approx([0.2, 0.2, 0.3, 0.3]),
        pytest.approx([0.5, 0.4, 1.0, 0.6]),
    ]


def test_full_image_box() -> None:
    assert yolo_to_xyxy("0 0.5 0.5 1 1") == [[0.0, 0.0, 1.0, 1.0]]


def test_clipping_to_unit_square() -> None:
    assert yolo_to_xyxy("0 0.05 0.95 0.2 0.2") == [pytest.approx([0.0, 0.85, 0.15, 1.0])]
    assert yolo_to_xyxy("0 0.5 0.5 3 3") == [[0.0, 0.0, 1.0, 1.0]]


@pytest.mark.parametrize("text", ["", "\n", "  \n\t\n", "\r\n"])
def test_blank_input_means_annotated_without_boxes(text: str) -> None:
    assert yolo_to_xyxy(text) == []


def test_blank_lines_and_surrounding_whitespace_are_skipped() -> None:
    text = "\n  0 0.5 0.5 0.2 0.2  \n\n\t0\t0.1\t0.1\t0.2\t0.2\r\n"
    assert yolo_to_xyxy(text) == [
        pytest.approx([0.4, 0.4, 0.6, 0.6]),
        pytest.approx([0.0, 0.0, 0.2, 0.2]),
    ]


def test_output_is_plain_floats() -> None:
    (box,) = yolo_to_xyxy("0 0.5 0.5 0.5 0.5")
    assert all(type(v) is float for v in box)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("0 0.5 0.5 0.2", "5 fields"),
        ("0 0.5 0.5 0.2 0.2 0.9", "5 fields"),
        ("smoke 0.5 0.5 0.2 0.2", "integer"),
        ("0.0 0.5 0.5 0.2 0.2", "integer"),
        ("1 0.5 0.5 0.2 0.2", "class"),
        ("0 a 0.5 0.2 0.2", "numbers"),
        ("0 0.5 0.5 nan 0.2", "finite"),
        ("0 inf 0.5 0.2 0.2", "finite"),
        ("0 0.5 0.5 -0.1 0.2", ">= 0"),
        ("0 0.5 0.5 0.2 0.2\n0 0.5", "line 2"),
    ],
)
def test_malformed_lines_raise(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        yolo_to_xyxy(text)
