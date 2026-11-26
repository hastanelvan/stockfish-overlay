"""Recognize the visible chess position and track subsequent legal moves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

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
    def __init__(self, image: np.ndarray, black_bottom: bool = False, *, turn=None,
                 castling=None, templates=None, profile_path=None):
        self.black_bottom = black_bottom
        self.board = chess.Board()
        self.streak = 0
        self.last_candidate = None
        self.missed = 0
        self.turn_assumed = False
        self.transition_cache = {}
        self.last_position_matches = False
        features = self.features(image)
        from piece_templates import template_sets
        saved = self.load_profile(profile_path)
        banks = (templates,) if templates is not None else template_sets(self.features)
        if saved:
            banks = (saved, *banks)
        matches = []
        last_error = None
        for bank in banks:
            self._set_templates(bank)
            try:
                placement, error = self._read_placement(features)
                matches.append((error, placement, bank))
            except CalibrationError as exc:
                last_error = exc
        if matches:
            _, placement, bank = min(matches, key=lambda m: m[0])
            self._set_templates(bank)
        else:
            bank = self._learn_start(features, black_bottom)
            if bank is None:
                raise CalibrationError(str(last_error or 'Taş teması tanınamadı.'))
            self._set_templates(bank)
            placement = chess.Board()
            try:
                self.save_profile(profile_path, bank)
            except OSError:
                pass  # Continue tracking the current game if the cache is read-only.
        self._adopt(placement, image, turn, castling)

    def _set_templates(self, bank):
        self.templates = bank
        self._cached_features = None
        self.__dict__.pop('_template_vectors', None)

    @staticmethod
    def load_profile(path):
        if path is None:
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                symbols = 'pnbrqkPNBRQK'
                if set(data.files) != set(symbols):
                    return None
                bank = {s: data[s] for s in symbols}
                if all(a.ndim == 4 and 1 <= len(a) <= 8 and a.shape[1:] == (24, 24, 2)
                       and np.isfinite(a).all() for a in bank.values()):
                    return bank
        except (OSError, ValueError, EOFError):
            pass
        return None

    @staticmethod
    def save_profile(path, bank):
        if path is None:
            return
        path = Path(path)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                temporary = Path(f.name)
                np.savez_compressed(f, **{s: np.asarray(items, dtype=np.float32)
                                         for s, items in bank.items()})
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def learn_reference(cls, image, black_bottom=False):
        """Learn a custom theme from an explicitly supplied starting-board image."""
        bank = cls._learn_start(cls.features(image), black_bottom)
        if bank is None:
            raise CalibrationError('Örnek görüntüdeki taşlar seçilen yön ve başlangıç dizilişiyle eşleşmedi.')
        return bank

    @classmethod
    def _learn_start(cls, features, black_bottom):
        # Unknown themes can be learned from an intact starting formation.
        # Require the pawn rows, repeated back-rank pieces and colors to agree.
        standard = chess.Board()
        bank = {}
        for i, feature in enumerate(features):
            square = cls._image_square(i, black_bottom)
            piece = standard.piece_at(square)
            if bool(piece) != bool(feature[:, :, 0].mean() > .06):
                return None
            if piece:
                bank.setdefault(piece.symbol(), []).append(feature)
        if np.mean([cls.brightness(f) for f in bank['P']]) <= np.mean([cls.brightness(f) for f in bank['p']]):
            return None
        for symbol, samples in bank.items():
            rivals = [f for other, items in bank.items() if other != symbol for f in items]
            for n, feature in enumerate(samples):
                other_type = min(cls.distance(feature, rival) for rival in rivals)
                if other_type < .0015:
                    return None
                if len(samples) > 1:
                    own_type = min(cls.distance(feature, other) for j, other in enumerate(samples) if j != n)
                    if own_type > other_type + .001:
                        return None
        return bank

    @staticmethod
    def _image_square(index, black_bottom):
        row, col = divmod(index, 8)
        if black_bottom:
            return chess.square(7 - col, row)
        return chess.square(col, 7 - row)

    def square_at(self, index: int) -> chess.Square:
        return self._image_square(index, self.black_bottom)
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

    @staticmethod
    def _foreground(tile, background, edge):
        h, w = tile.shape[:2]
        # A capture destination tints only the four corners. Recover the base
        # from the edge midpoints when the two colors fit Chessground's overlay.
        mid_y, mid_x = h // 2, w // 2
        sides = [tile[:edge, mid_x-edge:mid_x+edge], tile[-edge:, mid_x-edge:mid_x+edge],
                 tile[mid_y-edge:mid_y+edge, :edge], tile[mid_y-edge:mid_y+edge, -edge:]]
        side = np.median(np.concatenate([s.reshape(-1, 3) for s in sides]), axis=0)
        if np.max(np.abs(side-background)) > 15:
            for tint, opacity in ((np.array([20,85,0]), .3), (np.array([20,30,85]), .2)):
                if np.max(np.abs(side*(1-opacity) + tint*opacity - background)) < 9:
                    background = side
                    break

        mask = np.max(np.abs(tile-background), axis=2) > 27
        # A red check halo extends past both sides of the king. Remove red
        # background pixels only on tiles with this symmetric color signature.
        red = ((tile[:,:,0] - tile[:,:,1] > max(18, background[0]-background[1]+12))
               & (tile[:,:,1] < background[1]-6) & (tile[:,:,2] < background[2]-6))
        band = red[round(.42*h):round(.58*h)]
        if (band[:,round(.02*w):round(.08*w)].mean() > .65 and
                band[:,round(.92*w):round(.98*w)].mean() > .65):
            mask &= ~red

        return mask.astype(np.uint8), background

    @staticmethod
    def features(image):
        """Background-independent silhouettes and foreground brightness, 24x24."""
        if image.ndim != 3 or image.shape[2] < 3:
            raise CalibrationError('Ekran görüntüsü RGB olmalı.')
        h, w = image.shape[:2]
        if min(h, w) < 160 or abs(w - h) > max(w, h) * .10:
            raise CalibrationError('Tahta görüntüsü kare olmalı ve en az 160 piksel olmalı.')
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
                mask_full, background = BoardTracker._foreground(tile, background, edge)
                margin_y, margin_x = max(2,round(th*.12)), max(2,round(tw*.12))
                mask_full[:margin_y] = mask_full[-margin_y:] = 0
                mask_full[:,:margin_x] = mask_full[:,-margin_x:] = 0
                n, labels, stats, centers = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
                keep = np.zeros(n, dtype=np.uint8)
                for label in range(1,n):
                    _, _, width, height, area = stats[label]
                    if area < th*tw*.012:
                        continue
                    cx, cy = centers[label]
                    if (abs(cx/tw-.5) < .04 and abs(cy/th-.5) < .04
                            and .24 <= width/tw <= .35 and .24 <= height/th <= .35
                            and .68 <= area/(width*height) <= .88):
                        # Chessground legal (green) and premove (blue) dots.
                        center = tile[round(cy), round(cx)]
                        if any(np.max(np.abs(center-(background+tint)*.5)) < 10
                               for tint in (np.array([20,85,30]), np.array([20,30,85]))):
                            continue
                    keep[label] = 1
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

    def _matching_transitions(self, occupancy, depth, turn):
        # Reuse transitions while animations produce different intermediate frames.
        key = (depth, turn)
        if key not in self.transition_cache:
            indexed = {}
            def walk(board, moves, remaining):
                if remaining == 0:
                    indexed.setdefault(board.occupied, []).append(tuple(moves))
                    return
                for move in list(board.legal_moves):
                    board.push(move)
                    walk(board, moves + [move], remaining - 1)
                    board.pop()
            base = self.board.copy()
            base.turn = turn
            walk(base, [], depth)
            self.transition_cache[key] = indexed
        result = []
        mask = sum(chess.BB_SQUARES[self.square_at(i)] for i, occupied in enumerate(occupancy) if occupied)
        for moves in self.transition_cache[key].get(mask, []):
            board = self.board.copy()
            board.turn = turn
            for move in moves:
                board.push(move)
            result.append((moves, board))
        return result

    def _read_placement(self, features):
        occupancy = self.occupancy(features)
        costs = self.costs(features)
        board = chess.Board(None)
        errors = []
        for i, occupied in enumerate(occupancy):
            if not occupied:
                if features[i][:,:,0].mean() > .08:
                    raise CalibrationError(f'{chess.square_name(self.square_at(i))}: taş veya işaret net okunamadı.')
                continue
            ranked = sorted((values[i], symbol) for symbol, values in costs.items())
            own_color = ranked[0][1].isupper()
            opposite = min(error for error,symbol in ranked if symbol.isupper() != own_color)
            if opposite - ranked[0][0] < .012:
                raise CalibrationError(f'{chess.square_name(self.square_at(i))}: taş rengi belirsiz; yeniden deneniyor.')
            if ranked[0][0] > .07 or ranked[1][0] - ranked[0][0] < .0015:
                raise CalibrationError(f'{chess.square_name(self.square_at(i))}: taş net tanınamadı; yeniden okut.')
            board.set_piece_at(self.square_at(i), chess.Piece.from_symbol(ranked[0][1]))
            errors.append(ranked[0][0])
        for turn in (chess.WHITE, chess.BLACK):
            board.turn = turn
            if board.is_valid():
                return board, float(np.mean(errors))
        raise CalibrationError('Tanınan konum geçerli değil. Tahta yönünü ve taş temasını kontrol et.')

    def _highlighted_move(self, image, board):
        """Read a plausible last move from two distinctly recolored squares."""
        h, w = image.shape[:2]
        backgrounds = []
        for row in range(8):
            for col in range(8):
                tile = image[round(row*h/8):round((row+1)*h/8),
                             round(col*w/8):round((col+1)*w/8), :3]
                n = max(2, round(min(tile.shape[:2]) * .08))
                corners = [tile[:n,:n], tile[-n:,:n], tile[:n,-n:], tile[-n:,-n:]]
                backgrounds.append(np.median(np.concatenate(corners).reshape(-1,3), axis=0))
        backgrounds = np.asarray(backgrounds)
        parity = np.indices((8,8)).sum(axis=0).ravel() % 2
        colors = np.stack([np.median(backgrounds[parity == i], axis=0) for i in (0, 1)])
        marked = np.flatnonzero(np.abs(backgrounds - colors[parity]).max(axis=1) > 24)
        if len(marked) != 2:
            return None
        source, target = [self.square_at(i) for i in marked]
        if board.piece_at(source):
            source, target = target, source
        piece = board.piece_at(target)
        if board.piece_at(source) or not piece:
            return None
        before = board.copy(stack=False)
        before.turn = piece.color
        before.remove_piece_at(target)
        before.set_piece_at(source, piece)
        move = chess.Move(source, target)
        home_rank = 0 if piece.color else 7
        if (piece.piece_type == chess.KING and source == chess.square(4, home_rank)
                and target in (chess.square(2, home_rank), chess.square(6, home_rank))):
            kingside = chess.square_file(target) == 6
            rook_home = chess.square(7 if kingside else 0, home_rank)
            rook_target = chess.square(5 if kingside else 3, home_rank)
            rook = chess.Piece(chess.ROOK, piece.color)
            if before.piece_at(rook_home) is None and before.piece_at(rook_target) == rook:
                before.remove_piece_at(rook_target)
                before.set_piece_at(rook_home, rook)
                before.castling_rights = chess.BB_SQUARES[rook_home]
        if before.is_valid() and before.is_legal(move):
            return move
        # A capture can have blocked a check in the prior position.
        before.set_piece_at(target, chess.Piece(chess.KNIGHT, not piece.color))
        if before.is_valid() and before.is_legal(move):
            return move
        if (chess.square_rank(target) == 7 - home_rank and piece.piece_type in
                (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
            move = chess.Move(source, target, promotion=piece.piece_type)
            before.set_piece_at(source, chess.Piece(chess.PAWN, piece.color))
            if chess.square_file(source) == chess.square_file(target):
                before.remove_piece_at(target)
            if before.is_valid() and before.is_legal(move):
                return move
        return None

    def _adopt(self, board, image, turn, castling):
        standard = board.board_fen() == chess.STARTING_BOARD_FEN
        if castling is None:
            castling = 'KQkq' if standard else '-'
        board.ep_square = None
        try:
            board.set_castling_fen(castling)
        except ValueError as exc:
            raise CalibrationError('Rok hakları KQkq veya - olmalı.') from exc
        last_move = self._highlighted_move(image, board)
        last_piece = board.piece_at(last_move.to_square) if last_move else None
        assumed = False
        if turn is None:
            valid = []
            for color in (chess.WHITE, chess.BLACK):
                board.turn = color
                if board.is_valid():
                    valid.append(color)
            if len(valid) == 1:
                turn = valid[0]
            elif last_piece:
                turn = not last_piece.color
            else:
                turn = chess.WHITE
                assumed = not standard
        board.turn = turn
        if (last_piece and last_piece.color != turn and last_piece.piece_type == chess.PAWN
                and abs(last_move.from_square - last_move.to_square) == 16):
            board.ep_square = (last_move.from_square + last_move.to_square) // 2
        if not board.is_valid():
            raise CalibrationError('Tanınan konum geçerli değil. Hamle sırasını ve görüntüyü kontrol et.')
        self.board = board
        self.turn_assumed = assumed
        self.last_observed = self.expected(board)
        self.last_position_matches = True
        self.missed = 0
        self.transition_cache.clear()
        self.streak = 0
        self.last_candidate = None
        return board.copy()

    def resync(self, image, turn=None, castling=None):
        """Rebuild from visible pieces and any available last-move evidence."""
        board, _ = self._read_placement(self.features(image))
        return self._adopt(board, image, turn, castling)

    def observe(self, image: np.ndarray) -> Recognition | None:
        """Return Recognition only when a *new* stable legal position is observed.

        Requires two consecutive matching captures; otherwise waits for animation
        to finish. Can catch up by two plies if screenshots skip both moves.
        """
        features = self.features(image)
        observed = self.occupancy(features)
        self.last_observed = observed
        costs = self.costs(features)
        symbols = tuple(min(costs, key=lambda symbol: costs[symbol][i]) if full else None
                        for i, full in enumerate(observed))

        def matches(board):
            if observed != self.expected(board):
                return False
            for i, symbol in enumerate(symbols):
                if symbol is not None:
                    expected = board.piece_at(self.square_at(i)).symbol()
                    if costs[expected][i] > min(.07, costs[symbol][i] + .0015):
                        return False
                elif features[i][:,:,0].mean() > .08:
                    return False
            return True

        self.last_position_matches = matches(self.board)
        if self.last_position_matches:
            self.streak = 0
            self.last_candidate = None
            self.missed = 0
            return None
        if symbols != self.last_candidate:
            self.last_candidate = symbols
            self.streak = 1
            self.missed = 0
            return None
        self.streak += 1
        if self.streak < 2:
            return None
        # Retry with fresh visual evidence while occupancy stays unchanged.
        # A failed animation frame must never permanently lock this position.
        if self.streak > 2 and self.streak % 3 != 2:
            return None
        turns = [self.board.turn]
        if self.turn_assumed:
            turns.append(not self.board.turn)
        candidates = []
        for depth in (1, 2):
            for turn in turns:
                candidates.extend((moves, board) for moves, board
                                  in self._matching_transitions(observed, depth, turn)
                                  if matches(board))
            if candidates:
                break
        if not candidates:
            self.missed += 1
            return Recognition(self.board.copy(), [], 'Pozisyon eşleşmedi; animasyon ya da algılama hatası. Tahtayı kontrol et.')
        # Compare piece identities as well as occupancy. In particular, captures
        # with the same source square can otherwise be indistinguishable.
        ranked = sorted((self.appearance_error(board, features), n, moves, board)
                        for n, (moves, board) in enumerate(candidates))
        best = ranked[0]
        distinct = next((r for r in ranked[1:] if r[3].fen() != best[3].fen()), None)
        if distinct is not None and distinct[0] - best[0] < .0015:
            self.missed += 1
            return Recognition(self.board.copy(), [], 'Görüntü belirsiz; taşların netleşmesi bekleniyor.')
        moves, self.board = best[2], best[3]
        self.last_position_matches = True
        self.turn_assumed = False
        self.streak = 0
        self.last_candidate = None
        self.missed = 0
        self.transition_cache.clear()
        return Recognition(self.board.copy(), [m.uci() for m in moves], f'{len(moves)} hamle algılandı.')
