class Type2ModelFree:
    #Type 2 includes Close, Same, Left, In ("on the green")
    #constructor
    def __init__(self, type):
        self.PuttToFairwayProb = -1000
        self.PuttToRavineProb = -1000
        self.PuttToCloseProb = -1000
        self.PuttToSameProb = -1000
        self.PuttToLeftProb = -1000
        self.PuttToOverProb = -1000
        self.PuttToInProb = -1000
        self.type = type

    def setType(self, name):
        self.Type = name

    def setProb(self, prob, endlocation):
        if (endlocation == "Fairway"):
            self.PuttToFairwayProb = prob
        elif (endlocation == "Ravine"):
            self.PuttToRavineProb = prob
        elif (endlocation == "Close"):
            self.PuttToCloseProb = prob
        elif (endlocation == "Same"):
            self.PuttToSameProb = prob
        elif (endlocation == "Left"):
            self.PuttToLeftProb = prob
        elif (endlocation == "Over"):
            self.PuttToOverProb = prob
        elif (endlocation == "In"):
            self.PuttToInProb = prob

    def printState(self):
        print(f'Type: {self.type }');
        #print(f'At To Close Proabilityb: {self.AtToCloseProb}\n');
        print(f'At to Same Probability: {self.AtToSameProb}\n');
        print(f'At to Ravine Probability: {self.AtToRavineProb}\n');
        print(f'At To Fairway Probabilit: { self.AtToFairwayProb}\n');
        print(f'At To Left Probaility: {self.AtToLeftProb}\n');
        #print(f'At To Over Probability{self.AtToOverProb}\n');
        print(f'At To in Probability: {self.AtToInProb}\n');
        print(f'Past To Close Probability: {self.PastToCloseProb}\n');
        print(f'Past to Same Probability: {self.PastToSameProb}\n');
        print(f'Past to Ravine Probability: {self.PastToRavineProb}\n');
        print(f'Past to Fairway Probability: {self.PastToFairwayProb}\n');
        print(f'Past TO Left Probability: {self.PastToLeftProb}\n');
        print(f'Past To Over Probability: {self.PastToOverProb}\n');
        print(f'Past to in Probaility: {self.PastToInProb}\n');
        print(f'Lert To Close Probability: {self.LeftToCloseProb}\n');
        print(f'Left to Same Probability: {self.LeftToSameProb}\n');
        print(f'Left to Ravine Probability: {self.LeftToRavineProb}\n');
        print(f'Left to Fairway Probability: {self.LeftToFairwayProb}\n');
        print(f'Left to Left Probability: {self.LeftToLeftProb}\n');
        print(f'Left to Over Probability: {self.LeftToOverProb}\n');
        print(f'Left to in Probability: {self.LeftToInProb}\n');