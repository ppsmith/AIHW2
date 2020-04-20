class OverTheGreen:
    # State of being over the green

    def __init__(self):
        self.Name = "overTheGreen"
        self.Number = 3
        self.Actions = []

    def getName(self):
        return self.Name
    def getNumber(self):
        return self.Number

    def addToActions(self, action):
        self.Actions.append(action)

    def getActions(self):
        return self.Actions