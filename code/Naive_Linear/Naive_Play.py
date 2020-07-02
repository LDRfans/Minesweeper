from Naive_Linear import Naive_Linear
from MineSweeper import MineSweeper

game = MineSweeper()
naive_player = Naive_Linear(game.state, 8, 8, 10)

# pick a tile
game.selectCell([0,0])

print(naive_player.sweep())
print(naive_player.input)
print(naive_player.average_prob)
# print(naive_player.output)