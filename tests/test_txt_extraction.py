import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audiobook_converter import QwenAudiobookConverter


def _converter():
    # Bypass __init__ (it contacts the TTS API); _extract_txt only uses
    # _clean_text, which needs no instance state.
    return QwenAudiobookConverter.__new__(QwenAudiobookConverter)


def test_extract_txt_strips_utf8_bom(tmp_path):
    # Windows Notepad saves UTF-8 with a BOM by default; it must not leak
    # into the text sent for narration.
    target = tmp_path / 'bom.txt'
    target.write_bytes(b'\xef\xbb\xbfHello world.')
    assert _converter()._extract_txt(target) == 'Hello world.'


def test_extract_txt_plain_utf8_unchanged(tmp_path):
    target = tmp_path / 'plain.txt'
    target.write_text('Hello world.', encoding='utf-8')
    assert _converter()._extract_txt(target) == 'Hello world.'
