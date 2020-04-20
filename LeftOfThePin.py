class LeftOfThePin:
    # State of being Left of the pin

    def __init__(self):
        self.Name = "Left of Pin"
        self.Number = 2
        self.Actions = []

    def getName(self):
        return self.Name
    def getNumber(self):
        return self.Number

    def addToActions(self, action):
        self.Actions.append(action)

    def getActions(self):
        return self.Actions