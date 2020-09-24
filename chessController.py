#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import namedtuple
import queue
import serial
import RPi.GPIO as GPIO
import copy

###############################################################################
# From this point, and until stated otherwise, the code is taken from the SunFish chess engine implementation.
# link to sunfish : 
###############################################################################



# Piece-Square tables. Tune these to change sunfish's behaviour


piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' \
                            and j not in (self.ep, self.kp, self.kp-1, self.kp+1): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j == self.ep:
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1

        
        depth = max(depth, 0)

        
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        
        if DRAW_TEST:
            if not root and pos in self.history:
                return 0

        
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        
        def moves():
            
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)
            
            if depth == 0:
                yield None, pos.score
            
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
            
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Clear before setting, so we always have a value
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                break

       
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)

        return best

    def search(self, pos, history=()):
        
        self.nodes = 0
        if DRAW_TEST:
            self.history = set(history)
            
            self.tp_score.clear()

        
        for depth in range(1, 1000):
            
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower+upper+1)//2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                        self.bound(pos, lower, depth)
            
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower


###############################################################################
# User interface (sunfish)
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')

###############################################################################
# From this point, all code has been written by me for this project.
# In the main loop, calls are made to sunfish methods to generate chess engine moves and check for wrong moves
###############################################################################


# This function returns the corresponding column letter on the chessboard given a coordinate 
def posToLetter(pos):

    if pos == 0:
        return 'a'
    if pos == 1:
        return 'b'
    if pos == 2:
        return 'c'
    if pos == 3:
        return 'd'
    if pos == 4:
        return 'e'
    if pos == 5:
        return 'f'
    if pos == 6:
        return 'g'
    if pos == 7:
        return 'h'

# This function returns the corresping coordinate on the chessboard given its Letter column
def letterToPos(letter):

    if letter == 'a':
        return 0
    if letter == 'b':
        return 1
    if letter == 'c':
        return 2
    if letter == 'd':
        return 3
    if letter == 'e':
        return 4
    if letter == 'f':
        return 5
    if letter == 'g':
        return 6
    if letter == 'h':
        return 7

# creates an tuple of absolute coordinates given a chess coordinates
def fromChessToAbs(chessMove):

    # x is letter
    # y is number 
    PosCol = chessMove[0]
    PosRow = chessMove[1]

    return (letterToPos(PosCol),(int(PosRow) - 1))
    

# This function updates the position of a knight when it is moved, as knight movement has to be handled differently than other pieces
def updateKnighPos(knightPos, oldPosX,oldPosY, newPosX,newPosY):

    toReturn = knightPos
    i = 0
    for pos in knightPos:
        
        if (pos[0] == oldPosX) and (pos[1] == oldPosY) :
            toReturn[i] = (newPosX,newPosY)
        i+=1
    return toReturn

# Path planning
def grid(chessBoard, st, end, diag):
    
    lCol = 17
    lRow = 19
    end_pt = end
    start_pt = st

    flag = False

    if chessBoard[start_pt[0]][start_pt[1]] == "#":
        chessBoard[start_pt[0]][start_pt[1]] = " "
        flag = True
    
    
    visited = {end_pt: None}
    queue = [end_pt]
    while queue:
        current = queue.pop(0)
        if current == start_pt:
            shortest_path = []
            while current:
                shortest_path.append(current)
                current = visited[current]
            if flag:
                chessBoard[start_pt[0]][start_pt[1]] = "#"
            return shortest_path
        adj_points = []
        
        current_row, current_col = current
        
        
        # Add N,S,E,W to adjacent points    
        
        if current_row > 0:
            if chessBoard[current_row - 1][current_col] == " ":
                adj_points.append((current_row - 1, current_col))
        
        if current_col < (len(chessBoard[0])-1):
            if chessBoard[current_row][current_col + 1] == " ": ## There was an error here!
                adj_points.append((current_row, current_col + 1))
        
        if current_row < (len(chessBoard) - 1):
            if chessBoard[current_row + 1][current_col] == " ":
                adj_points.append((current_row + 1, current_col))
        
        if current_col > 0:
            if chessBoard[current_row][current_col - 1] == " ":
                adj_points.append((current_row, current_col - 1))

        # if the platform can use the diagonals to travel, also add to adjacent points
        if diag == 1:
            
            if (current_row > 0) and (current_col > 0):
                if chessBoard[current_row - 1][current_col - 1] == " ":
                    adj_points.append((current_row - 1, current_col - 1))

            if (current_row > 0) and (current_col < (lCol-1)):
                if chessBoard[current_row - 1][current_col + 1] == " ":
                    adj_points.append((current_row - 1, current_col + 1))


            if (current_row < (lRow - 1)) and (current_col > 0):
                if chessBoard[current_row + 1][current_col - 1] == " ":
                    adj_points.append((current_row +1, current_col - 1))


            if (current_row < (lRow - 1)) and (current_col < (lCol-1)):
                if chessBoard[current_row + 1][current_col + 1] == " ":
                    adj_points.append((current_row + 1, current_col + 1))


        
        for point in adj_points:
            if point not in visited:
                visited[point] = current
                queue.append(point)

# Count the number of pieces present on the board, takes a 8x8 matrix of 1s and 0s as input
def countPieces(board):
    i = 0
    for c in board:
        for r in c:
            if r == 1:
                i+=1

    return i

# Print the chessboard a a matrix of 1 and 0, for debugging and testing purposes
def printBoard(board):
    for r in board:
        for c in r:
            print(c,end = " ")
        print()
    print("-----------------------------------")

# Takes 2 8x8 matrix as input and returns the position where those matrix differs (to identify a move)
def getDiff(boardOld, boardNew):
    for r in range(8):
        for c in range(8):
            if boardOld[r][c] != boardNew[r][c]:
                return str(posToLetter(r))+str(c+1)

# Uses GPIO pins on the raspberry PI to scan the board and returns a 8x8 matrix one 1 and 0 corresponding to which square are occupied on the chessboard
def scanBoard(colChess, rowChess):
    board = [[0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0]]


    row = 0
    for c in colChess:
        #print(str(c) + " :")
        GPIO.output(c,GPIO.HIGH)
        col = 0
        for r in rowChess:
            #print(str(r) + " :")
            #print(GPIO.input(r))
            time.sleep(0.001)
            board[row][col] = GPIO.input(r)
            col += 1

        GPIO.output(c,GPIO.LOW)
        row+=1

    return board


def main():

	# Connect to the arduino
    ser = serial.Serial('/dev/ttyACM0',  57600, timeout=1)
    ser.flush()

    hist = [Position(initial, 0, (True,True), (True,True), 0, 0)]
    searcher = Searcher()

    # initial matrix of positions as 17x19 used for path planning The '#' sign represents an occupied square
    T = [[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ','#',' ','#',' ',' ',' ',' ',' ',' ',' ',' ',' ','#',' ','#',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']]

    # initial matrix of positions but without any occupied squares, this matrix never changes and is used in path planning when the travelling platform in not moving a piece
    D = [[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
        [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']]

    # set up raspberry pins
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(17,GPIO.OUT) #A
    GPIO.setup(27,GPIO.OUT) #B
    GPIO.setup(22,GPIO.OUT) #C
    GPIO.setup(5,GPIO.OUT) #D
    GPIO.setup(6,GPIO.OUT) #E
    GPIO.setup(13,GPIO.OUT) #F
    GPIO.setup(19,GPIO.OUT) #G
    GPIO.setup(26,GPIO.OUT) #H



    GPIO.setup(18,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #1
    GPIO.setup(23,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #2
    GPIO.setup(24,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #3
    GPIO.setup(25,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #4
    GPIO.setup(12,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #5
    GPIO.setup(16,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #6
    GPIO.setup(20,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #7
    GPIO.setup(21,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #8

    colChess = [17,27,22,5,6,13,19,26] #A to H
    rowChess = [18,23,24,25,12,16,20,21] #1 to 8

    # initialise magnet position
    mgPos = (1,1)

    # the moves queue will contain the sequences of move directions to be transmitted to the arduino
    moves = queue.Queue(maxsize=200)

    # the captured queue contains the list of positions that make up the graveyard where captured pieces go to
    # they are added to the queue starting from the middle of the board (for aesthetic purposes)
    captured = queue.Queue(maxsize=200)

    captured.put((18,8))
    captured.put((18,9))
    captured.put((18,7))
    captured.put((18,10))
    captured.put((18,6))
    captured.put((18,11))
    captured.put((18,5))
    captured.put((18,12))
    captured.put((18,4))
    captured.put((18,13))
    captured.put((18,3))
    captured.put((18,14))
    captured.put((18,2))
    captured.put((18,15))
    captured.put((18,1))


    # initial position of knights on the board
    KnighPositions = [(3,1),(13,1),(3,15),(13,15)]

    # initialise two 8x8 matrices that will be used in the detection
    board = [[0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0]]

    newBoard = [[0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0]]

    # game loop
    while True:
        print_pos(hist[-1])

        # check if previous engine move won the game for the system
        if hist[-1].score <= -MATE_LOWER:
            print("You lost")
            break

        
        # We query the user until she enters a (pseudo) legal move.
        move = None
        
        
        # board scanning to detect player move



        #initialise current board 
        board = scanBoard(colChess,rowChess)


        iniMove = ""		#starting coordinates
        targetMove = ""		#end coordinates


        while True:
            

            board = scanBoard(colChess,rowChess)

            # Loop until the number of pieces present on the board decreases (a piece has been lifted), the location where the piece is missing corresponds to the initialcoordinates
            while True:
                
                newBoard = scanBoard(colChess,rowChess)
                if countPieces(newBoard) < countPieces(board):
                    print("piece off")
                    iniMove = getDiff(board,newBoard)
                    board = copy.deepcopy(newBoard)
                    break
                board = copy.deepcopy(newBoard)

            # loop until the number of pieces on the board increases (the piece has been placed on another square) to get the final coordinates
            # OR
            # loop until the number of pieces decrease again (a piece is captured) then loop again until the number of pieces increases to get the final coordinates
            scanning = True
            while scanning:
                
                newBoard = scanBoard(colChess,rowChess)

                if countPieces(newBoard) < countPieces(board):
                    print("piece off again")
                    board = copy.deepcopy(newBoard)

                    while True:
                        
                        newBoard = scanBoard(colChess,rowChess)
                        if countPieces(newBoard) > countPieces(board):
                            print("piece on")
                            targetMove = getDiff(board,newBoard)
                            scanning = False
                            break


                elif countPieces(newBoard) > countPieces(board):
                    print("piece on")
                    targetMove = getDiff(board,newBoard)
                    scanning = False


    		
            # construct player move
            playerMove = iniMove + targetMove
            
            match = re.match('([a-h][1-8])'*2, playerMove)
            
            # check if the move is valid or not, if it isn't promp the user to replace its piece and loop again.
            # if move is valid, exit loop.
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")

            if move in hist[-1].gen_moves():
                
                break
            else :
                print("incorrect Move, please replace piece to original Position then press enter")
                z = input()
        hist.append(hist[-1].move(move))

        


        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        print_pos(hist[-1].rotate())

        # check if player move won the game.
        if hist[-1].score <= -MATE_LOWER:
            print("You won")
            break

        # Fire up the engine to look for a move.
        start = time.time()
        for _depth, move, score in searcher.search(hist[-1], hist):
            if time.time() - start > 1:
                break

        if score == MATE_UPPER:
            print("Checkmate!")

        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        print("My move:", render(119-move[0]) + render(119-move[1]))

        pmIni = iniMove
        pmEnd = targetMove

        # get chess engine move
        engineMoveIni = (render(119-move[0]))
        engineMoveEnd = (render(119-move[1]))
        
        playerMoveOGx = fromChessToAbs(pmIni)[0]*2 + 1
        playerMoveOGy = fromChessToAbs(pmIni)[1]*2 + 1
        playerMoveX = fromChessToAbs(pmEnd)[0]*2 + 1
        playerMoveY = fromChessToAbs(pmEnd)[1]*2 + 1

        # set the initial move coordinate of the player move as free and final coordinates of the player move as occupied (for plath planning)
        T[playerMoveOGx][playerMoveOGy] = " "
        T[playerMoveX][playerMoveY] = "#"

        engineMoveOGx = fromChessToAbs(engineMoveIni)[0]*2 + 1
        engineMoveOGy = fromChessToAbs(engineMoveIni)[1]*2 + 1
        engineMoveX = fromChessToAbs(engineMoveEnd)[0]*2 + 1
        engineMoveY = fromChessToAbs(engineMoveEnd)[1]*2 + 1
        
        # set the starting point as free
        T[engineMoveOGx][engineMoveOGy] = " "

        # use path planning to create the sequence of direction moves to be executed by the platform to execute chess engive chess moves, everytime add to the moves queue
        
        # if piece at target position is occupied
        if (T[engineMoveX][engineMoveY] == "#"):

            prison = captured.get()
            # bring magnet under taken piece -- magnet off
            path = grid(D, (mgPos[0],mgPos[1]), (engineMoveX,engineMoveY), 1)
            for i in path:
                moves.put((i,0))

            # bring magnet to side -- magnet on
            path = grid(T, (engineMoveX,engineMoveY), prison, 0)
            for i in path:
                moves.put((i,1))

            # bring magnet to original player move point -- magnet off
            path = grid(D, prison, (engineMoveOGx,engineMoveOGy), 1)
            for i in path:
                moves.put((i,0))
        else:

            # bring magnet to original player move point -- magnet off
            path = grid(D, (mgPos[0],mgPos[1]), (engineMoveOGx,engineMoveOGy), 1)
            for i in path:
                moves.put((i,0))

        # if a knight is moving
        if ( (engineMoveOGx, engineMoveOGy) ) in KnighPositions:
            path = grid(T, (engineMoveOGx,engineMoveOGy), (engineMoveX,engineMoveY), 0)
            for i in path:
                moves.put((i,1))
            KnighPositions = updateKnighPos(KnighPositions,engineMoveOGx,engineMoveOGy,engineMoveX,engineMoveY)

        else:
            path = grid(T, (engineMoveOGx,engineMoveOGy), (engineMoveX,engineMoveY), 1)
            for i in path:
                moves.put((i,1))

        #update magnet Pos
        mgPos = (engineMoveX,engineMoveY)
        T[engineMoveX][engineMoveY] = "#"


        

        

        hist.append(hist[-1].move(move))
        wait = moves.qsize()
        ini = moves.get()
        
        # Convert the path that is represented by the moves queue, to a sequence of directions to be excuted in order for the platform to travel along that path.
        # transmit every move to the arduino board

        while (moves.empty() == False):
            temp = moves.get()
            

            if ( (ini[0][0]) == (temp[0][0]) ) and ( (ini[0][1]) == (temp[0][1]) ):
                s = str(5) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] -1) == (temp[0][0]) ) and ( (ini[0][1]+1) == (temp[0][1]) ):
                s = str(1) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] ) == (temp[0][0]) ) and ( (ini[0][1]+1) == (temp[0][1]) ):
                s = str(2) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] +1) == (temp[0][0]) ) and ( (ini[0][1]+1) == (temp[0][1]) ):
                s = str(3) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] -1) == (temp[0][0]) ) and ( (ini[0][1]) == (temp[0][1]) ):
                s = str(4) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] + 1) == (temp[0][0]) ) and ( (ini[0][1]) == (temp[0][1]) ):
                s = str(6) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] -1) == (temp[0][0]) ) and ( (ini[0][1]-1) == (temp[0][1]) ):
                s = str(7) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] ) == (temp[0][0]) ) and ( (ini[0][1]-1) == (temp[0][1]) ):
                s = str(8) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())
            elif ( (ini[0][0] +1) == (temp[0][0]) ) and ( (ini[0][1]-1) == (temp[0][1]) ):
                s = str(9) + "," + str(temp[1]) +  "\n"
                #print(s)
                ser.write(s.encode())

            ini = temp

        # once all moves direction that form the path have been transmitted, transmit (10) to signal end of transmission, arduino will then start execution
        s = str(10) + "," + str(temp[0]) +  "\n"
        ser.write(s.encode())  

        # wait for the arduino to finish execution.
        # since Raspeberry to arduino communication is only one way, we set the wait time as a function of the number of directions commands in the moves queue.
        # approx 0.6 seconds for one direction move (half a board square)
        time.sleep(0.6*wait)
        



if __name__ == '__main__':
    main()