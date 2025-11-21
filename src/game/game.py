"""Chess game implementation using python-chess."""
import chess
import os
from typing import Optional, List, Tuple


class ChessGame:
    """A chess game wrapper around python-chess."""

    def __init__(self, fen: Optional[str] = None):
        """
        Initialize a new chess game.

        Args:
            fen: Optional FEN string to start from a specific position.
                 If None, starts from the standard starting position.
        """
        self.board = chess.Board(fen) if fen else chess.Board()

    def make_move(self, move: str) -> bool:
        """
        Make a move in the game.

        Args:
            move: Move in UCI format (e.g., "e2e4") or SAN format (e.g., "e4").

        Returns:
            True if the move was legal and made, False otherwise.
        """
        try:
            # Try UCI format first
            if len(move) == 4 or len(move) == 5:
                chess_move = chess.Move.from_uci(move)
            else:
                # Try SAN format
                chess_move = self.board.parse_san(move)

            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
                return True
            return False
        except (ValueError, chess.InvalidMoveError):
            return False

    def get_legal_moves(self) -> List[str]:
        """
        Get all legal moves in UCI format.

        Returns:
            List of legal moves as UCI strings.
        """
        return [move.uci() for move in self.board.legal_moves]

    def get_legal_moves_san(self) -> List[str]:
        """
        Get all legal moves in SAN format.

        Returns:
            List of legal moves as SAN strings.
        """
        return [self.board.san(move) for move in self.board.legal_moves]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()

    def get_result(self) -> str:
        """
        Get the game result.

        Returns:
            "1-0" if white wins, "0-1" if black wins, "1/2-1/2" if draw, "*" if ongoing.
        """
        return self.board.result()

    def is_check(self) -> bool:
        """Check if the current player is in check."""
        return self.board.is_check()

    def is_checkmate(self) -> bool:
        """Check if the current player is in checkmate."""
        return self.board.is_checkmate()

    def is_stalemate(self) -> bool:
        """Check if the game is in stalemate."""
        return self.board.is_stalemate()

    def is_draw(self) -> bool:
        """Check if the game is a draw."""
        return self.board.is_draw()

    def get_turn(self) -> bool:
        """
        Get whose turn it is.

        Returns:
            True if it's white's turn, False if it's black's turn.
        """
        return self.board.turn

    def get_turn_string(self) -> str:
        """
        Get whose turn it is as a string.

        Returns:
            "white" or "black".
        """
        return "white" if self.board.turn else "black"

    def get_fen(self) -> str:
        """Get the current position in FEN format."""
        return self.board.fen()

    def get_board_unicode(self) -> str:
        """Get a Unicode representation of the board."""
        return str(self.board)

    def get_board(self) -> chess.Board:
        """
        Get the underlying chess.Board object.

        Returns:
            The python-chess Board object representing the current game state.
        """
        return self.board

    def undo_move(self) -> bool:
        """
        Undo the last move.

        Returns:
            True if a move was undone, False if there are no moves to undo.
        """
        if len(self.board.move_stack) > 0:
            self.board.pop()
            return True
        return False

    def reset(self) -> None:
        """Reset the game to the starting position."""
        self.board.reset()

    def get_move_history(self) -> List[str]:
        """
        Get the move history in UCI format.

        Returns:
            List of moves played so far.
        """
        return [move.uci() for move in self.board.move_stack]

    def get_move_history_san(self) -> List[str]:
        """
        Get the move history in SAN format.

        Returns:
            List of moves played so far.
        """
        # Reconstruct SAN moves by replaying the game
        history = []
        temp_board = chess.Board()
        for move in self.board.move_stack:
            history.append(temp_board.san(move))
            temp_board.push(move)
        return history

    def get_pgn(self) -> str:
        """
        Get the game in PGN format.

        Returns:
            PGN string representation of the game.
        """
        return str(chess.pgn.Game.from_board(self.board))

    def copy(self) -> "ChessGame":
        """
        Create a copy of the current game state.

        Returns:
            A new ChessGame instance with the same position.
        """
        return ChessGame(self.board.fen())

   

    def get_display_header(self) -> str:
        """
        Get the header/instructions for the game display.
        
        Returns:
            Formatted header string with instructions.
        """
        return (
            "Chess Game - Enter moves in UCI format (e.g., e2e4) or SAN format (e.g., e4)\n"
            "Commands: 'undo' to undo last move, 'quit' to exit\n"
            + "-" * 60
        )

    def get_display_board_state(self, show_board_array: bool = True, history_count: int = 5) -> str:
        """
        Get the current board state display including board, turn, check status, and move history.
        
        Args:
            show_board_array: Whether to include the board array representation.
            history_count: Number of recent moves to show in history.
        
        Returns:
            Formatted string showing the current game state.
        """
        lines = []
        lines.append(f"\n{self.get_board_unicode()}")
        # lines.append(f"\nTurn: {self.get_turn_string()}")
        
        if self.is_check():
            lines.append("CHECK!")
        
        # if show_board_array:
        #     lines.append(f"\n{self.get_board_array()}")
        
        # Show move history if there are moves
        # move_history = self.get_move_history_san()
        # if move_history:
        #     recent_moves = move_history[-history_count:]
        #     lines.append(f"\nMove history: {' '.join(recent_moves)}")
        
        return "\n".join(lines)

    def get_display_invalid_move(self, move: str, max_moves: int = 10) -> str:
        """
        Get the invalid move error message.
        
        Args:
            move: The invalid move that was attempted.
            max_moves: Maximum number of legal moves to display.
        
        Returns:
            Formatted string with error message and legal moves.
        """
        lines = [f"\nInvalid move: {move}"]
        legal_moves = self.get_legal_moves()
        lines.append(f"Legal moves: {', '.join(legal_moves[:max_moves])}")
        if len(legal_moves) > max_moves:
            lines.append(f"... and {len(legal_moves) - max_moves} more")
        lines.append("\nPress Enter to continue...")
        return "\n".join(lines)

    def get_display_game_over(self) -> str:
        """
        Get the game over display message.
        
        Returns:
            Formatted string showing the final board and game result.
        """
        lines = [f"{self.get_board_unicode()}"]
        result = self.get_result()
        
        if result == "1-0":
            lines.append("\nWhite wins!")
        elif result == "0-1":
            lines.append("\nBlack wins!")
        elif result == "1/2-1/2":
            lines.append("\nDraw!")
        else:
            lines.append(f"\nGame over: {result}")
        
        lines.append(f"\nFull move history: {' '.join(self.get_move_history_san())}")
        return "\n".join(lines)

    def get_display_full_history(self) -> str:
        """
        Get the full move history display.
        
        Returns:
            Formatted string with the complete move history.
        """
        return f"Full move history: {' '.join(self.get_move_history_san())}"

    def print_board_state(self, show_board_array: bool = True, history_count: int = 5) -> None:
        """
        Print the current board state to the console.
        
        Args:
            show_board_array: Whether to include the board array representation.
            history_count: Number of recent moves to show in history.
        """
        # print(self.get_display_header())
        print(self.get_display_board_state(show_board_array=show_board_array, history_count=history_count))

    def print_game_over(self) -> None:
        """Print the game over message with final board state."""
        print(self.get_display_game_over())

    

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

