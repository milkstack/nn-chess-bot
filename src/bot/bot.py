import random
import chess
import numpy as np
# from game import ChessGame
from bot.neural_net import NeuralNet

PIECE_TYPES = {
    chess.KING: 6,
    chess.QUEEN: 5,
    chess.ROOK: 4,
    chess.BISHOP: 3,
    chess.KNIGHT: 2,
    chess.PAWN: 1,
}


def get_board_array(board): #type(board) == chess.Board()
    """
    Optimized board array conversion using direct board access instead of EPD parsing.
    Returns a flat 64-element numpy array representing the board state.
    This is faster than EPD parsing and avoids unnecessary string conversions.
    """
    matrix = np.zeros(64, dtype=np.float32)
    
    # Map each square directly using piece_map for better performance
    # piece_map is a dictionary of square -> Piece, which is faster than calling piece_at() repeatedly
    for square, piece in board.piece_map().items():
        # Get piece type and color
        piece_value = PIECE_TYPES[piece.piece_type]
        # Negative for black pieces
        if not piece.color:
            piece_value = -piece_value
        matrix[square] = piece_value
    
    return matrix

class Bot:
    def __init__(self, game, is_white: bool, quiet: bool = False):
        self.game = game
        self.neural_net = NeuralNet([64, 128, 128, 1])
        self.is_white = is_white
        self.quiet = quiet

    def get_move(self):
        """
        Optimized move selection using push/pop instead of board copying.
        This significantly reduces memory allocation overhead.
        """
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return None
            
        board = self.game.get_board()
        
        # Batch evaluate all positions for better performance
        # Use push/pop to avoid expensive board copying
        move_scores = []
        move_objects = []
        
        for move_uci in legal_moves:
            chess_move = chess.Move.from_uci(move_uci)
            # Push move temporarily
            board.push(chess_move)
            
            # Evaluate position
            board_array = get_board_array(board)
            score = self.neural_net.evaluate(board_array.tolist())[0] * (1 if self.is_white else -1)
            
            move_scores.append(score)
            move_objects.append(move_uci)
            
            # Pop move to restore board state
            board.pop()
        
        # Find best move using argmax for efficiency
        best_idx = np.argmax(move_scores) if len(move_scores) > 0 else 0
        best_move = move_objects[best_idx]

        if not self.quiet:
            print("best_move: ", move_scores[best_idx])
            print("worst_move: ", min(move_scores) if move_scores else 0)
        
        return best_move
        