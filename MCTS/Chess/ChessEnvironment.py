from typing import List

'''
Storing all the information about the current state of chess game.
Determining valid moves at current state.
It will keep move log.
'''


class ChessEnvironment:
    def __init__(self):
        # NOTE: color might differ based on the theme
        self.white_pieces = {
            "R": "♜",
            "N": "♞",
            "B": "♝",
            "Q": "♛",
            "K": "♚",
            "p": "♟"
        }
        #
        self.white_piece_to_letter = {v: k for k, v in self.white_pieces.items()}
        self.black_pieces = {
            "R": "♖",
            "N": "♘",
            "B": "♗",
            "Q": "♕",
            "K": "♔",
            "p": "♙"
        }
        self.black_piece_to_letter = {v: k for k, v in self.black_pieces.items()}
        # default of black is south of the board and white on the north
        self.board = [
            # row 0 col 0~7
            [self.black_pieces["R"], self.black_pieces["N"], self.black_pieces["B"], self.black_pieces["Q"],
             self.black_pieces["K"],
             self.black_pieces["B"],
             self.black_pieces["N"], self.black_pieces["R"]],
            # row 1 col 0~7
            [self.black_pieces["p"], self.black_pieces["p"], self.black_pieces["p"], self.black_pieces["p"],
             self.black_pieces["p"], self.black_pieces["p"],
             self.black_pieces["p"], self.black_pieces["p"]],
            # row 2 col 0~7
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            # row 3 col 0~7
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            # row 4 col 0~7
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            # row 5 col 0~7
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            # row 6 col 0~7
            [self.white_pieces["p"], self.white_pieces["p"], self.white_pieces["p"], self.white_pieces["p"],
             self.white_pieces["p"], self.white_pieces["p"],
             self.white_pieces["p"], self.white_pieces["p"]],
            # row 1 col 0~7
            [self.white_pieces["R"], self.white_pieces["N"], self.white_pieces["B"], self.white_pieces["Q"],
             self.white_pieces["K"],
             self.white_pieces["B"],
             self.white_pieces["N"], self.white_pieces["R"]]]

        self.moveFunctions = {"p": self.getPawnMoves, "R": self.getRookMoves, "N": self.getKnightMoves,
                              "B": self.getBishopMoves, "Q": self.getQueenMoves, "K": self.getKingMoves}
        self.white_to_move = True
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.check_mate = False
        self.stale_mate = False
        self.in_check = False

    def makeMove(self, move):
        '''
        Takes a Move as a parameter and executes it.
        (this will not work for castling, pawn promotion and en-passant)
        '''
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.move_log.append(move)  # log the move so we can undo it later
        self.white_to_move = not self.white_to_move  # switch players
        # update king's location if moved
        if move.piece_moved == self.white_pieces["K"]:
            self.white_king_location = (move.end_row, move.end_col)
        elif move.piece_moved == self.black_pieces["K"]:
            self.black_king_location = (move.end_row, move.end_col)

        # pawn promotion
        if move.is_pawn_promotion:
            self.board[move.end_row][move.end_col] = self.white_pieces["Q"] if move.piece_moved == self.white_pieces[
                "p"] else self.black_pieces["Q"]

    def undoMove(self):
        '''
        Undo the last move
        '''
        if len(self.move_log) != 0:  # make sure that there is a move to undo
            move = self.move_log.pop()
            self.board[move.start_row][move.start_col] = move.piece_moved
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.white_to_move = not self.white_to_move  # swap players
            # update the king's position if needed
            if move.piece_moved == self.white_pieces["K"]:
                self.white_king_location = (move.start_row, move.start_col)
            elif move.piece_moved == self.black_pieces["K"]:
                self.black_king_location = (move.start_row, move.start_col)

    def getValidMoves(self):
        '''
        All moves considering checks.
        '''

        # naive algorithm
        # 1) generate all possible moves
        moves = self.getAllPossibleMoves()
        # 2) for each move, make the move
        for i in range(len(moves) - 1, -1, -1):  # when removing from a list go backwards through that list
            self.makeMove(moves[i])
            # 3) generate all opponent's moves
            # 4) for each of your opponent's moves, see if they attack your king
            self.white_to_move = not self.white_to_move
            if self.inCheck():
                # 5) if they do attack your king, not a valid move
                moves.remove(moves[i])
            self.white_to_move = not self.white_to_move
            self.undoMove()

        if len(moves) == 0:  # either checkmate or stalemate
            if self.inCheck():
                self.check_mate = True
            else:
                self.stale_mate = True
        else:
            self.check_mate = False
            self.stale_mate = False

        return moves

    def inCheck(self):
        if self.white_to_move:
            return self.squareUnderAttack(self.white_king_location[0], self.white_king_location[1])
        else:
            return self.squareUnderAttack(self.black_king_location[0], self.black_king_location[1])

    def squareUnderAttack(self, row, col):
        self.white_to_move = not self.white_to_move
        oppMoves: List[Move] = self.getAllPossibleMoves()
        self.white_to_move = not self.white_to_move
        for m in oppMoves:
            if m.end_row == row and m.end_col == col:
                return True
        return False

    def getAllPossibleMoves(self):
        '''
        All moves without considering checks.
        '''
        moves: List[Move] = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                piece = self.board[row][col]
                if piece in self.white_pieces.values() and self.white_to_move:
                    letter_piece = self.white_piece_to_letter.get(piece)
                    self.moveFunctions[letter_piece](row, col,
                                                     moves)  # calls appropriate move function based on piece type
                elif piece in self.black_pieces.values() and (not self.white_to_move):
                    letter_piece = self.black_piece_to_letter.get(piece)
                    self.moveFunctions[letter_piece](row, col,
                                                     moves)  # calls appropriate move function based on piece type
        return moves

    def getPawnMoves(self, row: int, col: int, moves: list):
        '''
        Get all the pawn moves for the pawn located at row, col and add the moves to the list.
        '''

        if self.white_to_move:
            if self.board[row - 1][col] == "--":
                moves.append(Move((row, col), (row - 1, col), self.board))
                if row == 6 and self.board[row - 2][col] == "--":
                    moves.append(Move((row, col), (row - 2, col), self.board))
            if col - 1 >= 0:
                if self.board[row - 1][col - 1] in self.black_pieces.values():
                    moves.append(Move((row, col), (row - 1, col - 1), self.board))
            if col + 1 <= 7:
                if self.board[row - 1][col + 1] in self.black_pieces.values():
                    moves.append(Move((row, col), (row - 1, col + 1), self.board))
        else:
            if self.board[row + 1][col] == "--":
                moves.append(Move((row, col), (row + 1, col), self.board))
                if row == 1 and self.board[row + 2][col] == "--":
                    moves.append(Move((row, col), (row + 2, col), self.board))
            if col - 1 >= 0:
                if self.board[row + 1][col - 1] in self.white_pieces.values():
                    moves.append(Move((row, col), (row + 1, col - 1), self.board))
            if col + 1 <= 7:
                if self.board[row + 1][col + 1] in self.white_pieces.values():
                    moves.append(Move((row, col), (row + 1, col + 1), self.board))

    def getRookMoves(self, row: int, col: int, moves: list):
        '''
        Get all the rook moves for the rook located at row, col and add the moves to the list.
        '''
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))  # up, left, down, right
        for direction in directions:
            for i in range(1, 8):
                end_row = row + direction[0] * i
                end_col = col + direction[1] * i
                if 0 <= end_row <= 7 and 0 <= end_col <= 7:  # check for possible moves only in boundaries of the board
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":  # empty space is valid
                        moves.append(Move((row, col), (end_row, end_col), self.board))
                    elif (end_piece in self.black_pieces.values() and self.white_to_move) or (
                            end_piece in self.white_pieces.values() and (not self.white_to_move)):  # capture enemy piece
                        moves.append(Move((row, col), (end_row, end_col), self.board))
                        break
                    else:  # friendly piece
                        break
                else:  # off board
                    break

    def getKnightMoves(self, row: int, col: int, moves: list):
        '''
        Get all the knight moves for the knight located at row col and add the moves to the list.
        '''
        knight_moves = ((-2, -1), (-2, 1), (-1, 2), (1, 2), (2, -1), (2, 1), (-1, -2),
                        (1, -2))  # up/left up/right right/up right/down down/left down/right left/up left/down
        for move in knight_moves:
            end_row = row + move[0]
            end_col = col + move[1]
            if 0 <= end_row <= 7 and 0 <= end_col <= 7:
                end_piece = self.board[end_row][end_col]
                if end_piece == "--":  # empty space is valid
                    moves.append(Move((row, col), (end_row, end_col), self.board))
                elif (end_piece in self.black_pieces.values() and self.white_to_move) or (
                        end_piece in self.white_pieces.values() and (not self.white_to_move)):  # capture enemy piece
                    moves.append(Move((row, col), (end_row, end_col), self.board))
                else:  # friendly piece
                    continue

    def getBishopMoves(self, row: int, col: int, moves: list):
        '''
        Get all the bishop moves for the bishop located at row col and add the moves to the list.
        '''
        directions = ((-1, -1), (-1, 1), (1, 1), (1, -1))  # diagonals: up/left up/right down/right down/left
        for direction in directions:
            for i in range(1, 8):
                end_row = row + direction[0] * i
                end_col = col + direction[1] * i
                if 0 <= end_row <= 7 and 0 <= end_col <= 7:  # check for possible moves only in boundaries of the board
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":  # empty space is valid
                        moves.append(Move((row, col), (end_row, end_col), self.board))
                    elif (end_piece in self.black_pieces.values() and self.white_to_move) or (
                            end_piece in self.white_pieces.values() and (not self.white_to_move)):  # capture enemy piece
                        moves.append(Move((row, col), (end_row, end_col), self.board))
                        break
                    else:  # friendly piece
                        break
                else:  # off board
                    break

    def getQueenMoves(self, row, col, moves):
        '''
        Get all the queen moves for the queen located at row col and add the moves to the list.
        '''
        self.getBishopMoves(row, col, moves)
        self.getRookMoves(row, col, moves)

    def getKingMoves(self, row, col, moves):
        king_moves = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
                      (1, 1))  # up/left up/right right/up right/down down/left down/right left/up left/down
        for move in king_moves:
            end_row = row + move[0]
            end_col = col + move[1]
            if 0 <= end_row <= 7 and 0 <= end_col <= 7:
                end_piece = self.board[end_row][end_col]
                if end_piece == "--":  # empty space is valid
                    moves.append(Move((row, col), (end_row, end_col), self.board))
                elif (end_piece in self.black_pieces.values() and self.white_to_move) or (
                        end_piece in self.white_pieces.values() and (not self.white_to_move)):  # capture enemy piece
                    moves.append(Move((row, col), (end_row, end_col), self.board))
                else:  # friendly piece
                    continue

    def print(self):
        for col in range(8):
            print(self.board[0:8][col])


class Move:
    '''
    in chess fields on the board are described by two symbols, one of them being number between 1-8 (which is corresponding to rows)
    and the second one being a letter between a-f (corresponding to columns), in order to use this notation we need to map our [row][col] coordinates
    to match the ones used in the original chess game
    '''
    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4,
                     "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {"a": 0, "b": 1, "c": 2, "d": 3,
                     "e": 4, "f": 5, "g": 6, "h": 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, pos1, pos2, board):
        self.start_row = pos1[0]
        self.start_col = pos1[1]
        self.end_row = pos2[0]
        self.end_col = pos2[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.is_pawn_promotion = (self.piece_moved == "♟" and self.end_row == 0) or (
                self.piece_moved == "♙" and self.end_row == 7)

        self.moveID = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col

    def __eq__(self, other):
        '''
        Overriding the equals method.
        '''
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        return self.piece_moved + " " + self.getRankFile(self.start_row, self.start_col) + "-->" + self.getRankFile(
            self.end_row, self.end_col) + " " + self.piece_captured

    def getRankFile(self, row, col):
        return self.cols_to_files[col] + self.rows_to_ranks[row]
