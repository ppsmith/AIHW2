class Ravine1:
    #State of being close to the pin

    def __init__(self):
        self.Name = "closeToPin"
        self.Number = 1
        self.Actions = []

    def getName(self):
        return self.Name

    def getNumber(self):
        return self.Number

    def addToActions(self, action):
        self.Actions.append(action)

    def getActions(self):
        return self.Actions