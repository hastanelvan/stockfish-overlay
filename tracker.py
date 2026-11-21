"""Screen-square occupancy recognition + legal-move tracking.

Training-friendly MVP: calibrate ONCE with the board in the standard starting
position. No FEN is entered by the user. A single screenshot cannot reveal
castling/en-passant state, so all moves are tracked from the start of the game.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import cv2


class CalibrationError(ValueError):
    pass


@dataclass
class Recognition:
    board: chess.Board
    moves: list[str]
    message: str


class BoardTracker:
    def __init__(self, image: np.ndarray, black_bottom: bool = False):
        self.black_bottom = black_bottom
        self.board = chess.Board()
        self.templates = {}
        self.streak = 0
        self.last_candidate: tuple[bool, ...] | None = None
        self.missed = 0
        self.scoring_threshold = 27.0
        scores = self.scores(image)
        self.empty_thresholds: dict[int, float] = {}
        initial = self.expected(self.board)
        for color in (0, 1):
            occupied = [x for i, x in enumerate(scores) if initial[i] and self.color(i) == color]
            empty = [x for i, x in enumerate(scores) if not initial[i] and self.color(i) == color]
            highest_empty, lowest_piece = max(empty), min(occupied)
            if lowest_piece - highest_empty < 0.016:
                raise CalibrationError(
                    'Tahta okunamadı. Başlangıç konumunu açıp kareleri tam seç. '
                    f'{"Açık" if color else "Koyu"} karelerde taş/zemin ayrımı düşük '
                    f'({lowest_piece:.3f} / {highest_empty:.3f}).'
                )
            self.empty_thresholds[color] = (highest_empty + lowest_piece) / 2
        features = self.features(image)
        for i in range(64):
            piece = self.board.piece_at(self.square_at(i))
            if piece:
                self.templates.setdefault(piece.symbol(), []).append(features[i])
        self.resync_ready = False
        self.transition_cache = {}
        if self.read(image) != initial:
            raise CalibrationError('İlk görüntü standart başlangıç dizilişiyle eşleşmiyor.')

    @staticmethod
    def color(index: int) -> int:
        row, col = divmod(index, 8)
        return (row + col + 1) % 2  # a8 = light; 0=dark, 1=light

    @staticmethod
    def scores(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] < 3:
            raise CalibrationError('Ekran görüntüsü RGB olmalı.')
        h, w = image.shape[:2]
        if min(h, w) < 160:
            raise CalibrationError('Tahta görüntüsü çok küçük.')
        if abs(w - h) > max(w, h) * 0.10:
            raise CalibrationError('Seçilen tahta kare olmalı; kenardaki koordinatları dahil etme.')
        pixels = image[:, :, :3].astype(np.float32)
        result = []
        for row in range(8):
            for col in range(8):
                y0 = round(row * h / 8)
                y1 = round((row + 1) * h / 8)
                x0 = round(col * w / 8)
                x1 = round((col + 1) * w / 8)
                tile = pixels[y0:y1, x0:x1]
                th, tw = tile.shape[:2]
                edge = max(2, int(min(th, tw) * 0.08))
                corners = np.concatenate(
                    [tile[:edge, :edge].reshape(-1, 3),
                     tile[:edge, -edge:].reshape(-1, 3),
                     tile[-edge:, :edge].reshape(-1, 3),
                     tile[-edge:, -edge:].reshape(-1, 3)], axis=0
                )
                background = np.median(corners, axis=0)
                margin_y, margin_x = max(1, round(th * .13)), max(1, round(tw * .13))
                core = tile[margin_y:th-margin_y, margin_x:tw-margin_x]
                deviation = np.max(np.abs(core - background), axis=2)
                # Fraction of central pixels visibly different from the tile color.
                score = float(np.mean(deviation > 27))
                result.append(score)
        return np.array(result, dtype=float)

    def square_at(self, index: int) -> chess.Square:
        row, col = divmod(index, 8)
        if self.black_bottom:
            return chess.square(7 - col, row)
        return chess.square(col, 7 - row)

    def expected(self, board: chess.Board) -> tuple[bool, ...]:
        return tuple(board.piece_at(self.square_at(i)) is not None for i in range(64))

    def read(self, image: np.ndarray) -> tuple[bool, ...]:
        return self.occupancy(self.features(image))

    def costs(self, features):
        if getattr(self, '_cached_features', None) is features:
            return self._cached_costs
        if not hasattr(self, '_template_vectors'):
            pairs=[(symbol,t) for symbol,samples in self.templates.items() for t in samples]
            self._template_symbols=[symbol for symbol,_ in pairs]
            templates=np.stack([t for _,t in pairs]).astype(np.float32)
            self._template_vectors=(templates*np.sqrt([.6,.25])).reshape(len(pairs),-1).astype(np.float32)
            self._template_brightness=np.array([self.brightness(t) for t in templates],dtype=np.float32)
        data=np.stack(features).astype(np.float32)
        vectors=(data*np.sqrt([.6,.25])).reshape(len(data),-1).astype(np.float32)
        matrix=(np.sum(vectors*vectors,axis=1)[:,None]+np.sum(self._template_vectors**2,axis=1)[None,:]
                -2*np.einsum('ik,jk->ij',vectors,self._template_vectors))/576
        brightness=data[:,:,:,1].sum(axis=(1,2))/np.maximum(data[:,:,:,0].sum(axis=(1,2)),1.)
        matrix += .35*(brightness[:,None]-self._template_brightness[None,:])**2
        costs={symbol:matrix[:,[i for i,s in enumerate(self._template_symbols) if s==symbol]].min(axis=1)
               for symbol in self.templates}
        self._cached_features=features
        self._cached_costs=costs
        return costs

    def occupancy(self, features):
        costs=self.costs(features)
        occupied=np.stack(list(costs.values())).min(axis=0)
        data=np.stack(features)
        empty=np.mean(data**2,axis=(1,2,3))
        return tuple(bool(v) for v in ((data[:,:,:,0].mean(axis=(1,2))>.06)&(occupied<empty)))

    @staticmethod
    def brightness(feature):
        mask = feature[:,:,0]
        return float(feature[:,:,1].sum()/max(mask.sum(),1.))

    @classmethod
    def distance(cls, feature, template):
        silhouette = float(np.mean((feature[:,:,0]-template[:,:,0])**2))
        intensity = float(np.mean((feature[:,:,1]-template[:,:,1])**2))
        # Silhouettes alone cannot separate otherwise identical white/black pieces.
        color = (cls.brightness(feature)-cls.brightness(template))**2
        return .60*silhouette + .25*intensity + .35*color

    def ranked_pieces(self, feature):
        return sorted((min(self.distance(feature,t) for t in samples),symbol)
                      for symbol,samples in self.templates.items())

    @staticmethod
    def features(image):
        """Background-independent silhouettes and foreground brightness, 24x24."""
        h, w = image.shape[:2]
        features = []
        for row in range(8):
            for col in range(8):
                tile = image[round(row*h/8):round((row+1)*h/8),
                             round(col*w/8):round((col+1)*w/8), :3].astype(float)
                th, tw = tile.shape[:2]
                edge = max(2, int(min(th, tw)*.08))
                corners = np.concatenate([tile[:edge,:edge].reshape(-1,3),
                    tile[:edge,-edge:].reshape(-1,3), tile[-edge:,:edge].reshape(-1,3),
                    tile[-edge:,-edge:].reshape(-1,3)])
                background = np.median(corners, axis=0)
                # Remove tiny isolated decorations (coordinates, legal-move dots).
                deviation = np.max(np.abs(tile-background),axis=2)
                mask_full = (deviation > 27).astype(np.uint8)
                margin_y, margin_x = max(2,round(th*.12)), max(2,round(tw*.12))
                mask_full[:margin_y] = mask_full[-margin_y:] = 0
                mask_full[:,:margin_x] = mask_full[:,-margin_x:] = 0
                n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
                keep = np.zeros(n, dtype=np.uint8)
                for label in range(1,n):
                    keep[label] = stats[label,cv2.CC_STAT_AREA] >= th*tw*.012
                mask_full = keep[labels].astype(float)
                mask = cv2.resize(mask_full,(24,24),interpolation=cv2.INTER_AREA)
                brightness = cv2.resize(mask_full*tile.mean(axis=2)/255,(24,24),interpolation=cv2.INTER_AREA)
                features.append(np.stack([mask,brightness],axis=-1))
        return features

    def appearance_error(self, board, features):
        costs=self.costs(features)
        total = 0.
        for i,feature in enumerate(features):
            piece=board.piece_at(self.square_at(i))
            total += float(costs[piece.symbol()][i]) if piece else float(np.mean(feature**2))
        return total

    def _matching_transitions(self, occupancy, depth):
        # Reuse transitions while animations produce different intermediate frames.
        if depth not in self.transition_cache:
            indexed = {}
            def walk(board, moves, remaining):
                if remaining == 0:
                    indexed.setdefault(board.occupied, []).append(tuple(moves))
                    return
                for move in list(board.legal_moves):
                    board.push(move)
                    walk(board, moves + [move], remaining - 1)
                    board.pop()
            walk(self.board.copy(), [], depth)
            self.transition_cache[depth] = indexed
        result = []
        mask = sum(chess.BB_SQUARES[self.square_at(i)] for i, occupied in enumerate(occupancy) if occupied)
        for moves in self.transition_cache[depth].get(mask, []):
            board = self.board.copy()
            for move in moves:
                board.push(move)
            result.append((moves, board))
        return result

    def resync(self, image, turn, castling='-'):
        """Rebuild from learned piece templates; hidden chess state is explicit."""
        features = self.features(image)
        occupancy = self.occupancy(features)
        board = chess.Board(None)
        for i, occupied in enumerate(occupancy):
            if not occupied:
                continue
            ranked = self.ranked_pieces(features[i])
            own_color = ranked[0][1].isupper()
            opposite = min(error for error,symbol in ranked if symbol.isupper() != own_color)
            if opposite - ranked[0][0] < .012:
                raise CalibrationError(f'{chess.square_name(self.square_at(i))}: taş rengi belirsiz; yeniden deneniyor.')
            if ranked[0][0] > .07 or ranked[1][0] - ranked[0][0] < .0015:
                raise CalibrationError(f'{chess.square_name(self.square_at(i))}: taş net tanınamadı; yeniden okut.')
            board.set_piece_at(self.square_at(i), chess.Piece.from_symbol(ranked[0][1]))
        board.turn = turn
        try:
            board.set_castling_fen(castling)
        except ValueError as exc:
            raise CalibrationError('Rok hakları KQkq veya - olmalı.') from exc
        if not board.is_valid():
            raise CalibrationError('Tanınan konum geçerli değil. Hamle sırasını ve görüntüyü kontrol et.')
        self.board = board
        self.last_observed = occupancy
        self.transition_cache.clear()
        self.streak = 0
        self.last_candidate = None
        return board.copy()

    def observe(self, image: np.ndarray) -> Recognition | None:
        """Return Recognition only when a *new* stable legal position is observed.

        Requires two consecutive matching captures; otherwise waits for animation
        to finish. Can catch up by two plies if screenshots skip both moves.
        """
        features = self.features(image)
        observed = self.occupancy(features)
        self.last_observed = observed
        if observed == self.expected(self.board):
            self.streak = 0
            self.last_candidate = None
            self.missed = 0
            return None
        if observed != self.last_candidate:
            self.last_candidate = observed
            self.streak = 1
            return None
        self.streak += 1
        if self.streak < 2:
            return None
        # Retry with fresh visual evidence while occupancy stays unchanged.
        # A failed animation frame must never permanently lock this position.
        if self.streak > 2 and self.streak % 3 != 2:
            return None
        candidates = self._matching_transitions(observed, 1)
        if not candidates:
            candidates = self._matching_transitions(observed, 2)
        if not candidates:
            self.missed += 1
            return Recognition(self.board.copy(), [], 'Pozisyon eşleşmedi; animasyon ya da algılama hatası. Tahtayı kontrol et.')
        # Compare piece identities as well as occupancy. In particular, captures
        # with the same source square can otherwise be indistinguishable.
        ranked = sorted((self.appearance_error(board, features), n, moves, board)
                        for n, (moves, board) in enumerate(candidates))
        best = ranked[0]
        changed = [i for i in range(64) if self.board.piece_at(self.square_at(i)) != best[3].piece_at(self.square_at(i))]
        for i in changed:
            piece = best[3].piece_at(self.square_at(i))
            if piece is None:
                continue
            ranked_pieces = self.ranked_pieces(features[i])
            own = min(error for error,symbol in ranked_pieces if symbol.isupper() == piece.color)
            opponent = min(error for error,symbol in ranked_pieces if symbol.isupper() != piece.color)
            if opponent + .012 < own:
                return Recognition(self.board.copy(), [], 'Taş rengi konumla uyuşmuyor; net görüntü bekleniyor.')
        if best[0] / 64 > .035:
            return Recognition(self.board.copy(), [], "Taş görüntüleri kalibrasyonla uyuşmuyor; net görüntü bekleniyor.")
        distinct = next((r for r in ranked[1:] if r[3].fen() != best[3].fen()), None)
        if distinct is not None and distinct[0] - best[0] < .0015:
            return Recognition(self.board.copy(), [], 'Görüntü belirsiz; taşların netleşmesi bekleniyor.')
        candidates = [(best[2], best[3])]
        moves, self.board = candidates[0]
        self.streak = 0
        self.last_candidate = None
        self.missed = 0
        self.transition_cache.clear()
        return Recognition(self.board.copy(), [m.uci() for m in moves], f'{len(moves)} hamle algılandı.')
