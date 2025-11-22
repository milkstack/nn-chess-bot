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


def get_board_array(board, from_black_perspective=False): #type(board) == chess.Board()
    """
    Optimized board array conversion with additional chess features.
    Returns a flat 76-element numpy array representing the board state:
    - 64 elements for piece positions (one per square)
    - 4 elements for castling rights (white kingside, white queenside, black kingside, black queenside)
    - 8 elements for en passant availability (one per file a-h)
    
    This is faster than EPD parsing and avoids unnecessary string conversions.
    
    Args:
        board: chess.Board object
        from_black_perspective: If True, flip the board 180 degrees and negate pieces
                               so that black pieces appear as positive values
                               (evaluating from black's perspective)
    """
    # 64 squares + 4 castling rights + 8 en passant files = 76 total
    matrix = np.zeros(76, dtype=np.float32)
    
    # Map each square directly using piece_map for better performance
    # piece_map is a dictionary of square -> Piece, which is faster than calling piece_at() repeatedly
    for square, piece in board.piece_map().items():
        # Get piece type and color
        piece_value = PIECE_TYPES[piece.piece_type]
        # Negative for black pieces (from white's perspective)
        if not piece.color:
            piece_value = -piece_value
        
        # If evaluating from black's perspective, flip the board
        if from_black_perspective:
            # Flip square index: rotate 180 degrees (square 0 becomes 63, etc.)
            flipped_square = 63 - square
            # Negate piece value (what was white is now black and vice versa)
            piece_value = -piece_value
            matrix[flipped_square] = piece_value
        else:
            matrix[square] = piece_value
    
    # Add castling rights (indices 64-67)
    # From white's perspective: [white_kingside, white_queenside, black_kingside, black_queenside]
    # From black's perspective: [black_kingside, black_queenside, white_kingside, white_queenside]
    # Castling rights can be a SquareSet or integer bitmask depending on python-chess version
    castling = board.castling_rights
    
    # Helper function to check if a square is in castling rights
    def has_castling(square):
        try:
            # Try as SquareSet (newer versions)
            return square in castling
        except TypeError:
            # Fall back to bitwise check (integer bitmask)
            return bool(castling & (1 << square))
    
    if from_black_perspective:
        # Flip perspective: black's rights come first
        matrix[64] = 1.0 if has_castling(chess.H8) else 0.0  # Black kingside
        matrix[65] = 1.0 if has_castling(chess.A8) else 0.0  # Black queenside
        matrix[66] = 1.0 if has_castling(chess.H1) else 0.0  # White kingside
        matrix[67] = 1.0 if has_castling(chess.A1) else 0.0  # White queenside
    else:
        # White's perspective: white's rights come first
        matrix[64] = 1.0 if has_castling(chess.H1) else 0.0  # White kingside
        matrix[65] = 1.0 if has_castling(chess.A1) else 0.0  # White queenside
        matrix[66] = 1.0 if has_castling(chess.H8) else 0.0  # Black kingside
        matrix[67] = 1.0 if has_castling(chess.A8) else 0.0  # Black queenside
    
    # Add en passant availability (indices 68-75, one per file a-h)
    # Files a-h map to indices 68-75 (same regardless of perspective)
    # The file is the same from both perspectives, only the rank changes
    if board.ep_square is not None:
        ep_square = board.ep_square
        # Get the file (0-7 for a-h), which is perspective-independent
        ep_file = chess.square_file(ep_square)
        matrix[68 + ep_file] = 1.0
    
    return matrix

class Bot:
    def __init__(self, game, is_white: bool, quiet: bool = False):
        self.game = game
        
        self.neural_net = NeuralNet([76, 256, 256, 256, 256, 1])
        self.is_white = is_white
        self.quiet = quiet

    def get_move(self):
        """
        Optimized move selection using batch evaluation for all legal moves.
        This significantly improves performance by evaluating all positions in a single batch.
        
        When evaluating moves, the board representation is always from the perspective
        of the player making the move (white pieces positive when white plays,
        black pieces positive when black plays). This ensures symmetry between
        white and black play.
        """
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return None
            
        board = self.game.get_board()
        
        # Collect all board positions for batch evaluation
        board_arrays = []
        move_objects = []
        
        # When black plays, evaluate from black's perspective (flip board)
        from_black_perspective = not self.is_white
        
        for move_uci in legal_moves:
            chess_move = chess.Move.from_uci(move_uci)
            # Push move temporarily
            board.push(chess_move)
            
            # Collect board array from the appropriate perspective
            # This ensures the neural network always evaluates from the current player's perspective
            board_array = get_board_array(board, from_black_perspective=from_black_perspective)
            board_arrays.append(board_array)
            move_objects.append(move_uci)
            
            # Pop move to restore board state
            board.pop()
        
        # Batch evaluate all positions at once (much faster than individual evaluations)
        if board_arrays:
            # Stack into batch: shape (num_moves, 76) - 64 squares + 12 additional features
            batch_input = np.stack(board_arrays, axis=0)
            # Batch evaluate: returns shape (num_moves, 1)
            # Since we're evaluating from the current player's perspective,
            # we don't need to negate scores - higher is always better for the current player
            batch_scores = self.neural_net.evaluate_batch(batch_input)
            move_scores = batch_scores[:, 0]
        else:
            move_scores = np.array([])
        
        # Find best move using argmax for efficiency
        if len(move_scores) > 0:
            best_idx = np.argmax(move_scores)
            best_move = move_objects[best_idx]
            
            if not self.quiet:
                print("best_move: ", move_scores[best_idx])
                print("worst_move: ", np.min(move_scores))
        else:
            best_move = None
        
        return best_move
        