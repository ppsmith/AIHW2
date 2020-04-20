class Type3ModelFree:
    #Type 3 includes Over the Green
    #constructor
    def __init__(self, type):
        self.type = type
        self.ChipToFairwayProb = -1000
        self.ChipToRavineProb = -1000
        self.ChipToCloseProb = -1000
        self.ChipToSameProb = -1000
        self.ChipToLeftProb = -1000
        self.ChipToOverProb = -1000
        self.ChipToInProb = -1000
        self.PitchToFairwayProb = -1000
        self.PitchToRavineProb = -1000
        self.PitchToCloseProb = -1000
        self.PitchToSameProb = -1000
        self.PitchToLeftProb = -1000
        self.PitchToOverProb = -1000
        self.PitchToInProb = -1000

    def setType(self, name):
        self.type = type

    def setProb(self, prob, action, endlocation):
        if (action == "Chip"):
            if (endlocation == "Fairway"):
                self.ChipToFairwayProb = prob
            elif (endlocation == "Ravine"):
                self.ChipToRavineProb = prob
            elif (endlocation == "Close"):
                self.ChipToCloseProb = prob
            elif (endlocation == "Same"):
                self.ChipToSameProb = prob
            elif (endlocation == "Left"):
                self.ChipToLeftProb = prob
            elif (endlocation == "Over"):
                self.ChipToOverProb = prob
            elif (endlocation == "In"):
                self.ChipToInProb = prob
        elif (action == "Pitch"):
            if (endlocation == "Fairway"):
                self.PitchToFairwayProb = prob
            elif (endlocation == "Ravine"):
                self.PitchToRavineProb = prob
            elif (endlocation == "Close"):
                self.PitchToCloseProb = prob
            elif (endlocation == "Same"):
                self.PitchToSameProb = prob
            elif (endlocation == "Left"):
                self.PitchToLeftProb = prob
            elif (endlocation == "Over"):
                self.PitchToOverProb = prob
            elif (endlocation == "In"):
                self.PitchToInProb = prob

    def printState(self):
        print(f'Type: {self.type}');
        print(f'At To Close Proabilityb: {self.AtToCloseProb}\n');
        print(f'At to Same Probability: {self.AtToSameProb}\n');
        print(f'At to Ravine Probability: {self.AtToRavineProb}\n');
        print(f'At To Fairway Probabilit: {self.AtToFairwayProb}\n');
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