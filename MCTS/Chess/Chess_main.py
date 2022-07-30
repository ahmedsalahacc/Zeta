from ChessEnvironment import ChessEnvironment
from ChessEnvironment import Move


def main():
    game = ChessEnvironment()
    running = True

    game.print()
    while running:
        pos1 = []
        pos2 = []
        print("white turn:") if game.white_to_move else print("black turn:")
        print("Valid moves:")
        valid_moves = game.getValidMoves()
        if len(valid_moves) == 0 and game.white_to_move:
            print("**********Black Player Won!**********")
            break
        elif len(valid_moves) == 0 and not game.white_to_move:
            print("**********White Player Won!**********")
            break
        for move in valid_moves:
            print(move.getChessNotation())
        pos1.append(Move.files_to_cols[input("\tpos1 raw:")])
        pos1.append(Move.ranks_to_rows[input("\tpos1 col:")])
        pos2.append(Move.files_to_cols[input("\tpos2 raw:")])
        pos2.append(Move.ranks_to_rows[input("\tpos2 col:")])
        pos1 = pos1[::-1]
        pos2 = pos2[::-1]
        m = Move(pos1, pos2, game.board)
        move_flag = False
        for move in valid_moves:
            if m == move:
                move_flag = True
                game.makeMove(m)
                game.print()
                break
        if move_flag:
            while True:
                print("1.continue  2.undo")
                choice = int(input())
                if choice == 1:
                    break
                elif choice == 2:
                    game.undoMove()
                    break
                else:
                    print("Invalid choice")
        else:
            print("************** Invalid move **************")
        game.print()


if __name__ == "__main__":
    ask_play = input("want to play the game? (1 or 0): ")
    if int(ask_play):
        main()
    else:
        print("Too sad, I wanted to play with you sharp boy (¬_¬ )\_(ツ)_/¯")
