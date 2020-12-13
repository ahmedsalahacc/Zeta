from abc import abstractmethod

class Board:

    @abstractmethod
    def starting_state(self):
        # Returns representation of the starting state of the game
        pass

    @abstractmethod
    def current_player(self, state):
        # Takes the game state and returns the current player's number
        pass

    @abstractmethod
    def next_state(self, state, play):
        # Takes the game state and the move to be applied and returns the new game state
        pass

    @abstractmethod
    def legal_plays(self, state):
        """
        Takes the current state and returns the full list of moves that are legal plays for the current player
        """
        pass

    @abstractmethod
    def winner(self, state_hist):
        '''
        Takes the state and returns the player number if the game is now won. If the game is still on going, it returns zero. 
        If the game is tied, it return 0
        If the game is not ended yet it returns None
        '''
        pass
