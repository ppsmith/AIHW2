class Fairway1:
    #State of being close to the pin

    def __init__(self):
        self.Name = "Fairway"
        self.Number = 0
        self.Actions = []


    def getName(self):
        return self.Name

    def getNumber(self):
        return self.Number

    def addToActions(self, action):
        self.Actions.append(action)

    def getActions(self):
        return self.Actions