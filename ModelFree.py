from Type1ModelFree import Type1ModelFree
from Type2ModelFree import Type2ModelFree
from Type3ModelFree import Type3ModelFree

from CloseToPin import CloseToPin
from InTheHole import InTheHole
from LeftOfThePin import LeftOfThePin
from SameLevelAsPin import SameLevelAsPin
from OverTheGreen import OverTheGreen
from Fairway import Fairway1
from Ravine import Ravine1
from PastPin import *
from LeftPin import *
from AtPin import *
from Chip import *
from Pitch import *
from Putt import *
import numpy as np
import sys
import pylab as pl
import networkx as nx

#Type 1 = Ravine and Fairway
#Type 2 = Close, Same, Left, In ("on the green")
#Type 3 = Over the Green

# Fairway = Type1ModelFree("Fairway")
# Ravine = Type1ModelFree("Ravine")
# Close = Type2ModelFree("Close")
# Same = Type2ModelFree("Same")
# Left = Type2ModelFree("Left")
# In = Type2ModelFree("In")
# Over = Type3ModelFree("Over")
#
# #Find local location of input
# #data = open(input('Please give file path:'), 'r')
#
# data = open('C:\\Users\\ppsmith\\Desktop\\learning.txt')
#
# #read each line of data
# for entry in data:
#     #make sure that data hasn't ended
#     if entry != '\n':
#         #Each line goes start location, action, end location, and probability of ending in end state
#         #Each aspect is separated by a /, so find positions of /
#         sub = entry
#         startlocation = "invalid"
#         action = "invalid"
#         endlocation = "invalid"
#         prob = "0"
#         current_index = 0
#         num_slash = 0
#         #break up each line to find the start location, the action, the end location, and the probability
#         while (current_index != -1):
#             current_index = sub.find("/")
#             print(current_index)
#             num_slash = num_slash + 1
#             if (num_slash == 1): #start location
#                 startlocation = sub[0:current_index]
#                 sub = sub[current_index + 1:len(sub)] #create substring starting from action
#             elif (num_slash == 2): #action
#                 action = sub[0:current_index] #create substring starting from end location
#                 sub = sub[current_index + 1:len(sub)]
#             elif (num_slash == 3): #end location and probability
#                 endlocation = sub[0:current_index]
#                 prob = sub[current_index + 2:len(sub)]
#                 sub = sub[current_index + 1:len(sub)]
#             else:
#                 break
#         prob = float(prob) #convert probability from string to float
#
#         #fill out each class depending on information in line read
#         if (startlocation == "Fairway"):
#             Fairway.setProb(prob, action, endlocation)
#         elif (startlocation == "Ravine"):
#             Ravine.setProb(prob, action, endlocation)
#         elif (startlocation == "Close"):
#             Close.setProb(prob, endlocation)
#         elif (startlocation == "Same"):
#             Same.setProb(prob, endlocation)
#         elif (startlocation == "Left"):
#             Left.setProb(prob, endlocation)
#         elif (startlocation == "In"):
#             In.setProb(prob, endlocation)
#         elif (startlocation == "Over"):
#             Over.setProb(prob, action, endlocation)
#     #if entry is '\n', data has ended
#     if entry == '\n':
#         break

#print(f'Fairway: {Fairway.printState()}')
#print(f'Ravine: {Ravine.printState()}')
#print(f'Close: {Close.printState()}')
#print(f'State: {Same.printState()}')
#print(f'Left: {Left.printState()}')
#print(f'In: {In.printState()}')
#print(f'Over: {Over.printState()}')


fairway = Fairway1()
ravine = Ravine1()
leftOfPin = LeftOfThePin()
closeToPin = CloseToPin()
inTheHole = InTheHole()
sameLevel = SameLevelAsPin()
overTheGreen = OverTheGreen()
pastPin = PastPin(1)
leftPin = LeftPin(2)
atPin = AtPin(3)
putt = Putt(4)
chip = Chip(5)
pitch = Pitch(6)

fairway.addToActions(pastPin)
fairway.addToActions(leftPin)
fairway.addToActions(atPin)

ravine.addToActions(pastPin)
ravine.addToActions(leftPin)
ravine.addToActions(atPin)

closeToPin.addToActions(putt)
leftOfPin.addToActions(putt)
sameLevel.addToActions(putt)

overTheGreen.addToActions(chip)
overTheGreen.addToActions(pitch)

edges = [(fairway, leftOfPin), (fairway, inTheHole), (fairway, sameLevel) ,(fairway, closeToPin), (fairway, overTheGreen),
         (ravine, leftOfPin), (ravine, inTheHole), (ravine, sameLevel) ,(ravine, closeToPin), (ravine, overTheGreen), (overTheGreen, leftOfPin),
         (overTheGreen, sameLevel), (overTheGreen, closeToPin), (overTheGreen, inTheHole), (leftOfPin, sameLevel), (leftOfPin, overTheGreen),
         (leftOfPin,inTheHole), (leftOfPin, closeToPin), (sameLevel, leftOfPin), (sameLevel, closeToPin), (sameLevel, overTheGreen), (sameLevel, inTheHole),
         (closeToPin, leftOfPin), (closeToPin, overTheGreen), (closeToPin, sameLevel), (closeToPin, inTheHole)]

goal = inTheHole

# G = nx.Graph()
# G.add_edges_from(edges)
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_labels(G, pos)
# pl.show()

MATRIX_SIZE = 7
M = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1

for point in edges:
    print(point)
    if point[1] is goal:
        M[point[0].getNumber(), point[1].getNumber()] = 100
    elif (point[0] is fairway or point[0] is ravine) and (point[1] is sameLevel or point[1] is leftOfPin or point[1] is overTheGreen):
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif (point[0] is fairway or point[0] is ravine) and point[1] is closeToPin:
        M[point[0].getNumber(), point[1].getNumber()] = 1
    elif point[0] is closeToPin and (point[1] is leftOfPin or point[1] is overTheGreen or point[1] is sameLevel):
        M[point[0].getNumber(), point[1].getNumber()] = -.5
    elif point[0] is overTheGreen and point[1] is leftOfPin or point[1] is sameLevel:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif point[0] is sameLevel and point[1] is overTheGreen or point[1] is leftOfPin:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif point[0] is leftOfPin and point[1] is overTheGreen:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    else:
        M[point[0].getNumber(), point[1].getNumber()] = 1

M[goal.getNumber(), goal.getNumber()] = 100
print('\n')
print(M)

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

gamma = 0.9
# learning parameter
initial_state = 1

#
#
# def takeAction(state, action):
#     prob = action.getProb()
#     if state == 1:
#         nextPostions = [inTheHole, closeToPin, leftOfPin, overTheGreen, sameLevel, ravine]
#         nextState = np.random.choice(nextPostions, None, False, [0, .25, .1, .15, .35, .15])
#     elif state == 2:
#         nextPostions = [inTheHole, closeToPin, leftOfPin, overTheGreen, sameLevel, fairway]
#         nextAction = np.random.choice(nextPostions, None, False, [0, .25, .1, .15, .35, .15])
#     elif state == 3:
#         nextPostions = [inTheHole, closeToPin, leftOfPin, sameLevel]
#         nextState = np.random.choice(nextPostions, None, False, [.1, .25, .3, .35])
#     elif state == 2:
#         nextPostions = [inTheHole, closeToPin, sameLevel, overTheGreen]
#         nextState = np.random.choice(nextPostions, None, False, [.1, .25, .3, .35])
#     elif state == 4:
#         nextPositions = [inTheHole, closeToPin, leftOfPin, overTheGreen]
#         nextState = np.random.choice(nextPostions, None, False, [.1, .25, .3, .35])
#     elif state == 5:
#         nextPositions = [inTheHole, sameLevel, leftOfPin, overTheGreen]
#         nextState = np.random.choice(nextPostions, None, False, [.5, .25, .15, .1])
#     elif state == 6:
#         nextState = None
#
#     return nextState

# Determines the available actions for a given state
def available_actions(state):
    if state == 1 or state == 2:
        available_action = fairway.getActions()
    elif state == 3:
        available_action = overTheGreen.getActions()
    else:
        available_action = sameLevel.getActions()
    #current_state_row = M[state,]
    #available_action = np.where(current_state_row >= 0)[1]
    return available_action

available_action = available_actions(initial_state)
# nextState = takeAction(initial_state, available_action)
# utility = utility(initial_state, )

#Chooses one of the available actions at random
def sample_next_action(available_actions_range):
    next_action = np.random.choice(available_action, 1)
    next_action[0].getProb()
    return next_action

action = sample_next_action(available_action)  #say next action is chip with value 1, action = 1

def update(current_state, action, gamma):
    max_index = np.where(Q[action[0].getProb(),] == np.max(Q[action[0].getProb(),]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action[0].getProb(), max_index]
    Q[current_state, action[0].getProb()] = M[current_state, action[0].getProb()] + gamma * max_value
    if (np.max(Q) > 0):
        return (np.sum(Q / np.max(Q) * 100))
    else:
        return (0)
    # Updates the Q-Matrix according to the path chosen

update(initial_state, action, gamma)

scores = []

for i in range(2000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_action = available_actions(current_state)
    action = sample_next_action(available_action)
    score = update(current_state, action, gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q / np.max(Q)*100)
# You can uncomment the above two lines to view the trained Q matrix

# Testing
current_state = 0
steps = [current_state]

while current_state != 6:
    print(current_state)
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

pl.plot(scores)
pl.xlabel('No of iterations')
pl.ylabel('Reward gained')
pl.show()

