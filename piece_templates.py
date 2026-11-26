"""Screen-recognition references from the Cburnett pieces shipped in python-chess."""
from functools import lru_cache

import chess
import chess.svg
import numpy as np
from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter
from PySide6.QtSvg import QSvgRenderer


@lru_cache(maxsize=1)
def template_sets(extractor):
    """Build per-symbol features with the same background removal used on screen."""
    symbols = "PNBRQKpnbrqk"
    renderers = {
        symbol: QSvgRenderer(QByteArray(chess.svg.piece(chess.Piece.from_symbol(symbol)).encode()))
        for symbol in symbols
    }
    references = {symbol: [] for symbol in symbols}
    # The sheet contains each piece on both common square colors, at several
    # rendering scales. This covers browser padding and antialiased edges.
    cell = 64
    backgrounds = (QColor("#f0d9b5"), QColor("#769656"))
    for scale in (.90, 1.0, 1.08):
        sheet = QImage(cell * 8, cell * 8, QImage.Format.Format_RGB888)
        sheet.fill(backgrounds[0])
        with QPainter(sheet) as painter:
            for index in range(len(symbols) * 2):
                row, col = divmod(index, 8)
                x, y = col * cell, row * cell
                tile = QRectF(x, y, cell, cell)
                painter.fillRect(tile, backgrounds[index // len(symbols)])
                side = cell * scale
                pad = (cell - side) / 2
                painter.setClipRect(tile)
                renderers[symbols[index % len(symbols)]].render(
                    painter, QRectF(x + pad, y + pad, side, side)
                )
                painter.setClipping(False)
        pixels = np.frombuffer(sheet.constBits(), dtype=np.uint8).reshape(cell * 8, cell * 8, 3)
        features = extractor(pixels)
        for index in range(len(symbols) * 2):
            references[symbols[index % len(symbols)]].append(features[index])
    return (references,)


def import_reference_image(filename, black_bottom=False, profile_path=None):
    """Learn a piece set from a separate picture of its standard starting board."""
    from board_locator import BoardLocator
    from tracker import BoardTracker, CalibrationError

    reader = QImageReader(str(filename))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        raise CalibrationError(f'Referans resmi açılamadı: {reader.errorString()}')
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    height, width = image.height(), image.width()
    scanlines = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(height, image.bytesPerLine())
    pixels = scanlines[:, :width * 3].reshape(height, width, 3)

    bank = None
    if width == height and min(width, height) >= 160:
        try:
            bank = BoardTracker.learn_reference(pixels, black_bottom)
        except CalibrationError:
            pass  # A square screenshot can still contain a border or surrounding UI.
    if bank is None:
        region = BoardLocator().find(pixels)
        if region is None:
            raise CalibrationError('Referans resminde tahta bulunamadı. Resmi 8×8 tahtaya kırp.')
        x, y, s = region.x, region.y, region.size
        bank = BoardTracker.learn_reference(pixels[y:y+s, x:x+s], black_bottom)
    try:
        BoardTracker.save_profile(profile_path, bank)
    except OSError as exc:
        raise CalibrationError(f'Taş teması kaydedilemedi: {exc}') from exc
    return bank
