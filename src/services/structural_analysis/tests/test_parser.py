"""Parser smoke testleri.

Gerçek SAP2000 fixture'ı (`fixtures/simple_beam.s2k`) gelince bu dosya
parametrize edilip genişletilecek. Şimdilik inline minimal .s2k metni ile
tokenizer → tables → parser zincirini doğrularız.
"""

from __future__ import annotations

from textwrap import dedent

from services.structural_analysis.model import ElementType
from services.structural_analysis.parser import (
    S2KParser,
    extract_tables,
    parse_row,
)


MINIMAL_S2K = dedent(
    """\
    $ Minimal .s2k — tek açıklık kiriş, 2 düğüm, 1 frame
    TABLE:  "JOINT COORDINATES"
       Joint=1   CoordSys=GLOBAL   CoordType=Cartesian   XorR=0   Y=0   Z=0
       Joint=2   CoordSys=GLOBAL   CoordType=Cartesian   XorR=6   Y=0   Z=0

    TABLE:  "CONNECTIVITY - FRAME"
       Frame=1   JointI=1   JointJ=2

    END TABLE DATA
    """
)


def test_parse_row_basic_and_quoted():
    row = parse_row('Frame=1   Section="W14X22"   JointI=2')
    assert row == {"Frame": "1", "Section": "W14X22", "JointI": "2"}


def test_parse_row_comma_decimal():
    # SAP Windows yerel ayarı virgül ondalığı üretebilir.
    row = parse_row("X=3,8   Y=0")
    assert row["X"] == "3,8"


def test_extract_tables_routes_rows_to_names():
    tables = extract_tables(MINIMAL_S2K)
    assert "JOINT COORDINATES" in tables
    assert "CONNECTIVITY - FRAME" in tables
    assert len(tables["JOINT COORDINATES"]) == 2
    assert len(tables["CONNECTIVITY - FRAME"]) == 1


def test_parser_builds_nodes_and_frames():
    model = S2KParser().parse(MINIMAL_S2K)
    assert set(model.nodes) == {1, 2}
    assert model.nodes[1].x == 0.0
    assert model.nodes[2].x == 6.0
    assert set(model.frame_elements) == {1}
    el = model.frame_elements[1]
    assert el.nodes == [1, 2]
    assert el.type == ElementType.FRAME_3D


def test_parser_skips_frame_with_unknown_joint():
    # JointJ=99 tanımsız → eleman atlanmalı
    bad = dedent(
        """\
        TABLE:  "JOINT COORDINATES"
           Joint=1   XorR=0   Y=0   Z=0

        TABLE:  "CONNECTIVITY - FRAME"
           Frame=1   JointI=1   JointJ=99

        END TABLE DATA
        """
    )
    model = S2KParser().parse(bad)
    assert set(model.nodes) == {1}
    assert model.frame_elements == {}
