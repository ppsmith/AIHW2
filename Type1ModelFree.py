class Type1ModelFree:
    #Type 1 includes Fairway and Ravine
    #constructor
    def __init__(self, type):
        self.type = type
        self.AtToCloseProb = -1000
        self.AtToSameProb = -1000
        self.AtToRavineProb = -1000
        self.AtToFairwayProb = -1000
        self.AtToLeftProb = -1000
        self.AtToOverProb = -1000
        self.AtToInProb = -1000
        self.PastToCloseProb = -1000
        self.PastToSameProb = -1000
        self.PastToRavineProb = -1000
        self.PastToFairwayProb = -1000
        self.PastToLeftProb = -1000
        self.PastToOverProb = -1000
        self.PastToInProb = -1000
        self.LeftToCloseProb = -1000
        self.LeftToSameProb = -1000
        self.LeftToRavineProb = -1000
        self.LeftToFairwayProb = -1000
        self.LeftToLeftProb = -1000
        self.LeftToOverProb = -1000
        self.LeftToInProb = -1000

    def setType(self, name):
        self.Type = name

    def setProb(self, prob, action, endlocation):
        if (action == "At"):
            if (endlocation == "Close"):
                self.AtToCloseProb = prob
            elif (endlocation == "Same"):
                self.AtToSameProb = prob
            elif (endlocation == "Ravine"):
                self.AtToRavineProb = prob
            elif (endlocation == "Fairway"):
                self.AtToFairwayProb = prob
            elif (endlocation == "Left"):
                self.AtToLeftProb = prob
            elif (endlocation == "Over"):
                self.AtToOverProb = prob
            elif (endlocation == "In"):
                self.AtToInProb = prob
        if (action == "Past"):
            if (endlocation == "Close"):
                self.PastToCloseProb = prob
            elif (endlocation == "Same"):
                self.PastToSameProb = prob
            elif (endlocation == "Ravine"):
                self.PastToRavineProb = prob
            elif (endlocation == "Fairway"):
                self.PastToFairwayProb = prob
            elif (endlocation == "Left"):
                self.PastToLeftProb = prob
            elif (endlocation == "Over"):
                self.PastToOverProb = prob
            elif (endlocation == "In"):
                self.PastToInProb = prob
        if (action == "Left"):
           if (endlocation == "Close"):
                self.LeftToCloseProb = prob
           elif (endlocation == "Same"):
                self.LeftToSameProb = prob
           elif (endlocation == "Ravine"):
                self.LeftToRavineProb = prob
           elif (endlocation == "Fairway"):
                self.LeftToFairwayProb = prob
           elif (endlocation == "Left"):
                self.LeftToLeftProb = prob
           elif (endlocation == "Over"):
                self.LeftToOverProb = prob
           elif (endlocation == "In"):
                self.LeftToInProb = prob

    def printState(self):
        print(f'Type: {self.type }');
        print(f'At To Close Proabilityb: {self.AtToCloseProb}\n');
        print(f'At to Same Probability: {self.AtToSameProb}\n');
        print(f'At to Ravine Probability: {self.AtToRavineProb}\n');
        print(f'At To Fairway Probabilit: { self.AtToFairwayProb}\n');
        print(f'At To Left Probaility: {self.AtToLeftProb}\n');
        print(f'At To Over Probability{self.AtToOverProb}\n');
        print(f'At To in Probability: {self.AtToInProb}\n');
        print(f'Past To Close Probability: {self.PastToCloseProb}\n');
        print(f'Past to Same Probability: {self.PastToSameProb}\n');
        print(f'Past to Ravine Probability: {self.PastToRavineProb}\n');
        print(f'Past to Fairway Probability: {self.PastToFairwayProb}\n');
        print(f'Past TO Left Probability: {self.PastToLeftProb}\n');
        print(f'Past To Over Probability: {self.PastToOverProb}\n');
        print(f'Past to in Probaility: {self.PastToInProb}\n');
        print(f'Lest To CLose Probability: {self.LeftToCloseProb}\n');
        print(f'Left to Same Probability: {self.LeftToSameProb}\n');
        print(f'Left to Ravine Probability: {self.LeftToRavineProb}\n');
        print(f'Left to Fairway Probability: {self.LeftToFairwayProb}\n');
        print(f'Left to Left Probability: {self.LeftToLeftProb}\n');
        print(f'Left to Over Probability: {self.LeftToOverProb}\n');
        print(f'Left to in Probability: {self.LeftToInProb}\n');