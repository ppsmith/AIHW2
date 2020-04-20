class SameLevelAsPin:
    # State of being same level as pin

    def __init__(self):
        self.Name = "sameLevel"
        self.Number = 4
        self.Actions = []

    def getName(self):
        return self.Name
    def getNumber(self):
        return self.Number

    def addToActions(self, action):
        self.Actions.append(action)

    def getActions(self):
        return self.Actions