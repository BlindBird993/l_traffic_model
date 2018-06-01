from mesa import Agent, Model
import numpy as np
import math

class RoadInterface(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.buyers = []
        self.sellers = []
        self.waitCars = []

        self.demands = []
        self.productions = []
        self.demandPrice = []
        self.supplyPrice = []

        self.historyDemands = []
        self.historyProductions = []

        self.testListOfWaitingCars = []
        self.listOfTotalDistribution = []

        self.gateDistribution = []
        self.bar1Distribution = []
        self.bar2Distribution = []

        self.gate_travel_time = []
        self.bar1_travel_time = []
        self.bar2_travel_time = []

        self.distributedDemands = []
        self.summedDemands = []

        self.buyerPriceList = []
        self.clearPriceList = []
        self.satisfiedDemands = []

        self.demandCount = 0
        self.dealCount = 0
        self.noDealCount = 0

        self.dealsList = []
        self.noDealsList = []

        self.amountOfCars = []
        self.amountOfFerries = []
        self.amountOfEmergensies = []

        self.buyerPrices = []
        self.sellerPrices = []

        self.pricesListCar = []
        self.pricesListlorry = []

        self.speedValue = []

        self.commonPrice = 0
        self.clearPrice = 0

        self.currentSeller = 0
        self.currentBuyer = 0

        self.numberOfBuyers = 0
        self.numberOfSellers = 0
        self.numberOfWaitingCars = 0

        self.waitList = []

        self.hour = 0
        self.day = 0
        self.week = 0
        self.price = 0

    def getRoadTravelTime(self):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.readyToSell is True:
                    if agent.unique_id == 'Gate 0':
                        self.gate_travel_time.append(agent.timeOfTravel)
                    elif agent.unique_id == 'Bar 0':
                        self.bar1_travel_time.append(agent.timeOfTravel)
                    elif agent.unique_id == 'Bar 1':
                        self.bar2_travel_time.append(agent.timeOfTravel)
        return self.gate_travel_time,self.bar1_travel_time,self.bar2_travel_time

    def rewardFunctionCar(self, densityVal, price, priceList, maxCapacity):
        test_price = priceList
        max_capacity = maxCapacity
        max_elem = max(test_price)
        price_elem = price/max_elem
        price_elem = round(price_elem, 2)

        density_val = densityVal / max_capacity
        capacity = max_capacity - densityVal

        y = (math.exp(price_elem) - 2)
        y_val = 1 - round(y, 1) * 10

        if density_val * 100 > 60:
            check_var = 6 - abs(y_val)
        else:
            check_var = abs(y_val)
        reward = check_var
        if reward < 0:
            reward = 0.0
        print("Reward {}".format(reward))
        return reward

    def rewardFunctionlorry(self, densityVal, price, priceList, maxCapacity):
        max_capacity = maxCapacity
        max_elem = max(priceList)
        price_elem = price / max_elem
        price_elem = round(price_elem,2)

        density_val = densityVal / max_capacity
        capacity = max_capacity - densityVal

        x = (2 - math.exp(price_elem)) * 10

        if density_val * 100 > 60:
            check_var = (7+round(x,0))
        else:
            check_var = abs(2+round(x,0))
        reward = check_var
        if reward < 0:
            reward = 0.0
        print("Reward {}".format(reward))
        return reward

    def getWaitingCars(self):
        self.numberOfWaitingCars = 0
        self.waitCars = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent) and agent.isWaiting):
                print("Agent {} status {}".format(agent.unique_id,agent.isWaiting))
                self.numberOfWaitingCars += 1
                self.waitCars.append(agent.unique_id)
        print("List of Cars Waited {}".format(self.waitCars))
        print("Number of Cars Waited {}".format(self.numberOfWaitingCars))

    def getSellers(self):
        self.numberOfSellers = 0
        self.sellers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                print("Agent {} ready to Sell {}".format(agent.unique_id,agent.readyToSell))
                if agent.readyToSell is True:
                    self.numberOfSellers += 1
                    self.sellers.append(agent.unique_id)
        print("List of Sellers {}".format(self.sellers))
        print("Number of sellers {}".format(self.numberOfSellers))

    def getAvailableCapacity(self):
        capacityValue = 0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent, BarAgent)):
                if agent.readyToSell is True:
                    capacityValue += agent.maxCapacity
        self.historyProductions.append(capacityValue)
        return capacityValue

    def getBuyres(self):
        self.numberOfBuyers = 0
        self.buyers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent)):
                if agent.readyToBuy is True:
                    self.numberOfBuyers += 1
                    self.buyers.append(agent.unique_id)
        self.historyDemands.append(self.numberOfBuyers)
        print("List of Buyers {}".format(self.buyers))
        print("Number of buyers {}".format(self.numberOfBuyers))

    def updatePrice(self, new_price,buyer_type):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, BarAgent) or isinstance(agent,GateAgent)):
                if agent.open is True:
                    if buyer_type == 'car':
                        agent.price = new_price
                    elif buyer_type == 'lorry':
                        agent.pricelorry = new_price

    def chooseSeller(self, buyer, seller_id,price, amount=None):
        seller = seller_id
        for agent in self.model.schedule.agents:
            if (isinstance(agent, BarAgent) or isinstance(agent,GateAgent)):
                if agent.readyToSell is True and agent.unique_id == seller:
                    seller_price = 0
                    if buyer.type == 'car':
                        seller_price = agent.price
                    elif buyer.type == 'lorry':
                        seller_price = agent.pricelorry
                    elif buyer.type == 'emergency':
                        seller_price = 0

                    print("Seller {}".format(agent.unique_id))
                    print("Seller price {}".format(seller_price))

                    if buyer.price >= seller_price:
                        print("Deal !")

                        self.dealCount += 1

                        agent.queue.append(buyer.unique_id)
                        agent.density += 1
                        agent.maxCapacity -= 1
                        print("Agent density {}".format(agent.density))
                        print("Agent capacity {}".format(agent.maxCapacity))
                        agent.calculateSpeed()

                        buyer.readyToBuy = False
                        buyer.isWaiting = False
                        self.buyers.remove(buyer.unique_id)

                        if agent.maxCapacity <= 0:
                            agent.readyToSell = False
                            self.numberOfSellers -= 1

                        if buyer.type == 'car':
                            reward_coef = round(np.mean(agent.pricePropensitiesCar),1)
                            reward = self.rewardFunctionCar(agent.density, agent.price, agent.setOfPricesCar,
                                                                agent.max)
                            if reward > 0:
                                reward += reward_coef
                            agent.updateStates(reward)
                            agent.updateStateProbabilities()
                        elif buyer.type == 'lorry':
                            reward_coef = round(np.mean(agent.pricePropensitieslorry), 1)
                            reward = self.rewardFunctionlorry(agent.density, agent.pricelorry, agent.setOfPriceslorry,
                                                                  agent.max)
                            if reward > 0:
                                reward += reward_coef
                            agent.updateStateslorry(reward)
                            agent.updateStateProbabilitieslorry()

                        agent.calculateSpeed()
                        agent.getState()
                        agent.makeChoice()
                        agent.makeChoicelorry()
                        print("Car price choice {}".format(buyer.price))
                        print("Fery price choice {}".format(agent.pricelorry))

                        self.numberOfBuyers -= 1
                        if self.numberOfWaitingCars > 0:
                            self.numberOfWaitingCars -= 1
                        self.demandCount += 1
                        print("Number of sellers {}".format(self.numberOfSellers))
                        print("Number of buyers {}".format(self.numberOfBuyers))
                        print("Number of waiting cars {}".format(self.numberOfWaitingCars))

                        self.pricesListCar.append(agent.price)
                        self.pricesListlorry.append(agent.pricelorry)

                        if (isinstance(agent, BarAgent) and agent.open is False):
                            agent.checkMainGate()
                            agent.checkIfOpen(agent.mainGatedensity, agent.mainGateCapacity)


                    else:
                        print('No deal')
                        seller_price = 0
                        self.noDealCount += 1

                        if buyer.type == 'car':
                            agent.updateStates(0.0) #penalty
                            agent.updateValues()
                            agent.updateStateProbabilities()
                            agent.getState()
                            agent.makeChoice()


                        elif buyer.type == 'lorry':
                            agent.updateStateslorry(0.0)
                            agent.updateValues()
                            agent.updateStateProbabilitieslorry()
                            agent.getState()
                            agent.makeChoicelorry()


                        elif buyer.type == 'emergency':
                            pass

                        if (isinstance(agent, BarAgent) and agent.open is False):
                            agent.checkMainGate()
                            agent.checkIfOpen(agent.mainGatedensity, agent.mainGateCapacity)

    def distributeCars(self):

        self.getAvailableCapacity()

        self.sellPrice = 0
        self.buyPrice = 0
        self.demandCount = 0
        self.dealCount = 0
        self.noDealCount = 0
        if self.numberOfWaitingCars > 0:
            while (not (self.numberOfSellers <= 0 or self.numberOfWaitingCars <= 0)):
                for agent in self.model.schedule.agents:
                    if (isinstance(agent, CarAgent) and agent.isWaiting is True):
                        self.buyPrice = agent.price
                        print("Buyer {} Type {}".format(agent.unique_id, agent.type))
                        print("Buy price {}".format(agent.price))
                        if self.numberOfSellers > 0:
                            seller_choice = agent.makeChoice()
                            self.chooseSeller(agent, seller_choice, self.buyPrice)
                        else:
                            print("No sellers!")
                            break

        while (not (self.numberOfSellers <= 0 or self.numberOfBuyers <= 0)):
            buyer_id = np.random.choice(self.buyers)
            print("Car Random ID {}".format(buyer_id))
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                    if agent.unique_id == buyer_id:
                        self.buyPrice = agent.price
                        seller_choice = agent.makeChoice()
                        print("Buyer {} Type {}".format(agent.unique_id,agent.type))
                        print("Buy price {}".format(agent.price))
                        if self.numberOfSellers > 0:
                            seller_choice = agent.makeChoice()
                            self.chooseSeller(agent, seller_choice, self.buyPrice)
                        else:
                            print("No sellers!")
                            break

        self.satisfiedDemands.append(self.demandCount)

        if (self.numberOfBuyers > 0 and self.numberOfSellers == 0) or (self.numberOfWaitingCars > 0 and self.numberOfSellers == 0):
            print("Not enough place")
            self.waitList = []
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                        agent.isWaiting = True

                        self.waitList.append(agent.unique_id)
            print("Cars left {}".format(self.waitList))

            for agent in self.model.schedule.agents:
                if (isinstance(agent, GateAgent) and agent.readyToSell is False):
                    self.openBars(agent.density,agent.max)

        elif self.numberOfBuyers == 0 and self.numberOfSellers > 0:
            print("Place left")
            for agent in self.model.schedule.agents:
                if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                    if agent.open is True:
                        print("Agent {} Queue {}\nNumber of cars {}".format(agent.unique_id,agent.queue,len(agent.queue)))
        else:
            print("No sellers and No buyers")

        self.dealsList.append(self.dealCount)
        self.noDealsList.append(self.noDealCount)

    def openBars(self,density,max_capacity):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, BarAgent) and agent.readyToSell is False and agent.open is False):
                agent.open = np.random.choice([True,False],p=[density/max_capacity,1-(density/max_capacity)])
                agent.emergencyOpen = agent.open
                agent.readyToSell = agent.open
                print("Agent {} is open: {}".format(agent.unique_id,agent.readyToSell))

    def getRoadCarsDistribution(self):
        sellerDistributionList = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.unique_id == 'Gate 0':
                    self.gateDistribution.append(len(agent.queue))
                elif agent.unique_id == 'Bar 0':
                    self.bar1Distribution.append(len(agent.queue))
                elif agent.unique_id == 'Bar 1':
                    self.bar2Distribution.append(len(agent.queue))
        return self.gateDistribution,self.bar1Distribution,self.bar2Distribution


    def step(self):
        print("Trade!")
        self.getWaitingCars()
        self.getSellers()
        self.getBuyres()
        self.distributeCars()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class CarAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)
        self.hour = 0
        self.day = 0
        self.week = 0
        self.statusPriority = None

        self.price = 0
        self.priceListCar = []
        self.priceListlorry = []

        self.traided = None

        self.goingToPass = None

        self.isWaiting = False
        self.waitingTime = 0

        self.priceHistory = []
        self.priorityHistorySell = []
        self.priorityHistoryBuy = []

        self.type = None

        self.readyToSell = False
        self.readyToBuy = True

        self.initialStep = True

    def makeChoice(self):
        choiceList = []
        sellers = []
        valueList = []
        max_val = 0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, BarAgent) or isinstance(agent,GateAgent)):
                if agent.open is True and agent.readyToSell is True:
                    sellers.append(agent.unique_id)
                    if self.type == 'car':
                        choiceVal = agent.price + agent.timeOfTravel
                        valueList.append(choiceVal)
                        choiceList.append((agent.unique_id,choiceVal))
                    elif self.type == 'lorry':
                        choiceVal = agent.pricelorry + agent.timeOfTravel
                        valueList.append(choiceVal)
                        choiceList.append((agent.unique_id, choiceVal))
                    else:
                        choiceVal = agent.timeOfTravel
                        valueList.append(choiceVal)
                        choiceList.append((agent.unique_id, choiceVal))
        choice = min(valueList)
        choiceIndex = valueList.index(choice)
        for index, elem in enumerate(sellers):
            if index == choiceIndex:
                choice = elem
        print("Available choices {}".format(sellers))
        print("Priorities {}".format(valueList))
        print("Choice list {}".format(choiceList))
        print("Suggested choice {}".format(choice))
        return choice


    def name_func(self):
        print("Agent {}".format(self.unique_id))

    def getType(self):
        if not self.isWaiting:
            self.type = np.random.choice(['car','lorry','emergency'],p=[0.6,0.3,0.1])
        print("Type {}".format(self.type))

    def getPassStatus(self): #get probability based on rush hours
        if self.hour >= 7 and self.hour <= 9:
            self.goingToPass = np.random.choice([True,False],p=[0.9,0.1])
        elif self.hour >= 15 and self.hour <= 17:
            self.goingToPass = np.random.choice([True, False], p=[0.9, 0.1])
        else:
            self.goingToPass = np.random.choice([True, False])
        print("Status {}".format(self.goingToPass))

    def checkIfWaiting(self):
        if self.isWaiting:
            self.goingToPass = True
        print("Waiting {}".format(self.isWaiting))
        print("Going to Pass {}".format(self.goingToPass))

    def getTradeStatus(self):
        if self.goingToPass:
            self.readyToBuy = True
        else:
            self.readyToBuy = False

    def setPriceStrategies(self):
        if self.initialStep is True:
            for i in range(46, 57, 1):
                self.priceListCar.append(i)
            for j in range(132, 163, 1):
                self.priceListlorry.append(j)
        self.initialStep = False

    def calculatePrice(self):
        if self.type == 'car':
            self.price = np.random.choice(self.priceListCar)
        elif self.type == 'lorry':
            self.price = np.random.choice(self.priceListlorry)
        else:
            self.price = 0
        print("Price {}".format(self.price))

    def step(self):
        self.name_func()
        self.setPriceStrategies()
        self.getType()
        self.getPassStatus()
        self.checkIfWaiting()
        self.getTradeStatus()
        self.calculatePrice()
        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class GateAgent(Agent):
    def __init__(self,unique_id, model,max_capacity=10,road_lenght = 7):
        super().__init__(unique_id, model)
        self.queue = []
        self.open = True
        self.readyToSell = True

        self.initialSpeed = 70
        self.currentSpeed = 0

        self.timeOfTravel = 0

        self.initial_step = True

        self.hour = 0
        self.day = 0
        self.week = 0

        self.price = 0
        self.pricelorry = 0

        self.roadLength = road_lenght
        self.speedLimit = 0

        self.initialPropensityValuesCar = []
        self.initialPropensityValueslorry = []

        self.pricesEmptyCar = []
        self.pricesLowCar = []
        self.pricesIntermediateCar = []
        self.pricesPackedCar = []
        self.pricesFullCar = []

        self.pricesEmptyLorry = []
        self.pricesLowLorry = []
        self.pricesIntermediateLorry = []
        self.pricesPackedLorry = []
        self.pricesFullLorry = []

        self.densityLevelLow = []
        self.densityLevelPacked = []

        self.rushHour = False
        self.number_of_cars = 0
        self.density = 0
        self.max = max_capacity
        self.maxCapacity = max_capacity

        self.setOfPricesCar = []
        self.setOfPriceslorry = []

        self.pricePropensitiesCar = []
        self.priceProbsCar = []

        self.pricePropensitieslorry = []
        self.priceProbslorry = []

        self.memory_param = 0.1
        self.experimental_param = 0.1

        self.priceChoiceCar = 0
        self.priceChoicelorry = 0

        self.choiceCar = 0
        self.stateChoiceCar = None

        self.choicelorry = 0
        self.stateChoicelorry = None

        self.currentState = None

        self.statesCar = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.choicesDict = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.statesProbCar = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.stateslorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.choicesDictlorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.statesProblorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }

    def calculateSpeed(self):
        n = 2
        m = 2
        speed_val = self.initialSpeed*((1-self.density/self.max)**m)**n
        self.currentSpeed = round(speed_val,3)
        if self.currentSpeed <= 0:
            self.timeOfTravel = self.roadLength*10
        else:
            self.timeOfTravel = self.roadLength/self.currentSpeed
            self.timeOfTravel = round(self.timeOfTravel*10,3)
        print("Speed value {}".format(self.currentSpeed))
        print("Time value {}".format(self.timeOfTravel))

    def getState(self):
        density_var = (self.density/self.max)*100
        if density_var >= 0 and density_var <= 20:
            self.currentState = 'Empty'
        elif density_var > 20 and density_var <= 40:
            self.currentState = 'Low'
        elif density_var > 40 and density_var <=60:
            self.currentState = 'Intermediate'
        elif density_var > 60 and density_var <=80:
            self.currentState = 'Packed'
        else:
            self.currentState = 'Full'
        print("density {}, State {}".format(self.density,self.currentState))

    def updateStates(self,reward):
        updated_states = []
        i = 0
        for index, elem in enumerate(self.statesCar[self.currentState]):
            if index == self.choiceCar:
                print("elem {}".format(elem))
                elem = (1 - self.memory_param) * elem + (reward * (1 - self.experimental_param))
            else:
                elem = (1 - self.memory_param) * elem + (elem * (self.experimental_param / (len(self.statesCar[self.currentState]) - 1)))
            updated_states.insert(i, elem)
            i += 1
        print("Old states {}".format(self.statesCar))
        self.statesCar[self.currentState] = updated_states
        print("Updated states {}".format(self.statesCar))

    def updateStateslorry(self,reward):
        updated_states = []
        i = 0
        for index, elem in enumerate(self.stateslorry[self.currentState]):
            if index == self.choicelorry:
                print("elem {}".format(elem))
                elem = (1 - self.memory_param) * elem + (reward * (1 - self.experimental_param))
            else:
                elem = (1 - self.memory_param) * elem + (elem * (self.experimental_param / (len(self.statesCar[self.currentState]) - 1)))

            updated_states.insert(i, elem)
            i += 1
        print("Old states {}".format(self.stateslorry))
        self.stateslorry[self.currentState] = updated_states
        print("Updated states {}".format(self.stateslorry))

    def updateStateProbabilities(self):
        propensitiesList = self.statesCar[self.currentState]
        probs = []
        propSum = sum(propensitiesList)
        for index, elem in enumerate(propensitiesList):
            elem = elem / propSum
            probs.insert(index, elem)
        self.statesProbCar[self.currentState] = probs
        print("Probabilities {}".format(self.statesProbCar))

    def updateStateProbabilitieslorry(self):
        propensitiesList = self.stateslorry[self.currentState]
        probs = []
        propSum = sum(propensitiesList)
        for index, elem in enumerate(propensitiesList):
            elem = elem / propSum
            probs.insert(index, elem)
        self.statesProblorry[self.currentState] = probs
        print("Probabilities {}".format(self.statesProblorry))

    def makeChoice(self):
        choicesList = self.choicesDict[self.currentState]
        choiceProbList = self.statesProbCar[self.currentState]
        choiceVar = np.random.choice(choicesList, p=choiceProbList)
        self.stateChoiceCar = choiceVar
        self.choiceCar = choicesList.index(choiceVar)
        self.price = self.stateChoiceCar

        self.addPriceBasedOnStateCar()
        print("Choice {}, Index {}".format(self.stateChoiceCar, choicesList.index(choiceVar)))

    def makeChoicelorry(self):
        choicesList = self.choicesDictlorry[self.currentState]
        choiceProbList = self.statesProblorry[self.currentState]
        choiceVar = np.random.choice(choicesList, p=choiceProbList)
        self.stateChoicelorry = choiceVar
        self.choicelorry = choicesList.index(choiceVar)
        self.pricelorry = self.stateChoicelorry

        self.addPriceBasedOnStateLorry()
        print("Choice lorry {}, Index {}".format(self.stateChoicelorry, choicesList.index(choiceVar)))

    def addPriceBasedOnStateCar(self):
        if self.currentState == 'Empty':
            self.pricesEmptyCar.append(self.price)

        elif self.currentState == 'Low':
            self.pricesLowCar.append(self.price)
            self.densityLevelLow.append(self.density)

        elif self.currentState == 'Intermediate':
            self.pricesIntermediateCar.append(self.price)

        elif self.currentState == 'Packed':
            self.pricesPackedCar.append(self.price)
            self.densityLevelPacked.append(self.density)
        else:
            self.pricesFullCar.append(self.price)

    def addPriceBasedOnStateLorry(self):
        if self.currentState == 'Empty':
            self.pricesEmptyLorry.append(self.pricelorry)

        elif self.currentState == 'Low':
            self.pricesLowLorry.append(self.pricelorry)
            self.densityLevelLow.append(self.density)

        elif self.currentState == 'Intermediate':
            self.pricesIntermediateLorry.append(self.pricelorry)

        elif self.currentState == 'Packed':
            self.pricesPackedLorry.append(self.pricelorry)
            self.densityLevelPacked.append(self.density)

        elif self.currentState == 'Full':
            self.pricesFullLorry.append(self.pricelorry)

    def updateValues(self):
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for state in stateElements:
            for index, elem in enumerate(self.statesCar[state]):
                self.statesCar[state][index] = elem * 0.9
                self.statesCar[state][index] += 0.1
            for index, elem in enumerate(self.stateslorry[state]):
                self.stateslorry[state][index] = elem * 0.9
                self.stateslorry[state][index] += 0.1

    def setPriceStrategies(self):
        self.setOfPricesCar = []
        self.setOfPriceslorry = []
        for i in range(44, 57, 1):
            self.setOfPricesCar.append(i)
        for i in range(132, 163, 1):
            self.setOfPriceslorry.append(i)

        stateElements = ['Empty','Low','Intermediate','Packed','Full']
        for elem in stateElements:
            self.choicesDict[elem] = self.setOfPricesCar
            self.choicesDictlorry[elem] = self.setOfPriceslorry

    def setPricePropensities(self):
        self.pricePropensitiesCar = []
        self.pricePropensitieslorry= []
        for i in range(len(self.setOfPricesCar)):
            self.pricePropensitiesCar.append(1)
        for i in range(len(self.setOfPriceslorry)):
            self.pricePropensitieslorry.append(1)
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for elem in stateElements:
            self.statesCar[elem] = self.pricePropensitiesCar
            self.stateslorry[elem] = self.pricePropensitieslorry
        self.initialPropensityValuesCar = self.pricePropensitiesCar
        self.initialPropensityValueslorry = self.pricePropensitieslorry

    def setPriceProbabilities(self):
        self.priceProbsCar = []
        self.priceProbslorry = []
        for i in range(len(self.setOfPricesCar)):
            self.priceProbsCar.append(1 / len(self.setOfPricesCar))
        for i in range(len(self.setOfPriceslorry)):
            self.priceProbslorry.append(1 / len(self.setOfPriceslorry))
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for elem in stateElements:
            self.statesProbCar[elem] = self.priceProbsCar
            self.statesProblorry[elem] = self.priceProbslorry

    def prepareData(self):
        if self.readyToSell:
            if self.initial_step:
                self.setPriceStrategies()
                self.setPricePropensities()
                self.setPriceProbabilities()
                self.initial_step = False
            print("Set of prices {}".format(self.choicesDict))
            print("Set of propensities {}".format(self.statesCar))
            print("Set of probabilities {}".format(self.statesProbCar))

            print("Set of prices lorry{}".format(self.choicesDictlorry))
            print("Set of propensities lorry{}".format(self.stateslorry))
            print("Set of probabilities  lorry{}".format(self.statesProblorry))

    def checkIfReadyToSell(self):
        if self.open:
            self.readyToSell = True
        print("Ready to Sell {}".format(self.readyToSell))
        print("Capacity {}".format(self.maxCapacity))

    def updateQueue(self):
        self.queue = []
        self.density = 0
        self.maxCapacity = self.max

    def checkIfRushHour(self):
        if self.hour >= 7 and self.hour <= 9:
            self.rushHour = True
        elif self.hour >= 15 and self.hour <= 17:
            self.rushHour = True
        else:
            self.rushHour = False
        print("Rush Hour {}".format(self.rushHour))


    def name_func(self):
        print("Agent {}, length {}".format(self.unique_id,self.roadLength))

    def step(self):
        self.name_func()
        self.updateQueue()
        self.checkIfReadyToSell()
        self.checkIfRushHour()

        self.getState()
        self.calculateSpeed()

        self.prepareData()
        self.makeChoice()
        self.makeChoicelorry()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class BarAgent(Agent):
    def __init__(self,unique_id, model, max_capacity = 10,road_length = 10):
        super().__init__(unique_id, model)
        self.open = True

        self.initialSpeed = 60
        self.currentSpeed = 0
        self.timeOfTravel = 0

        self.queue = []
        self.readyToSell = True

        self.initial_step = True

        self.emergencyOpen = False

        self.hour = 0
        self.day = 0
        self.week = 0

        self.price = 0
        self.pricelorry = 0

        self.roadLength = road_length
        self.speedLimit = 0

        self.number_of_cars = 0
        self.density = 0
        self.max = max_capacity
        self.maxCapacity = max_capacity

        self.setOfPricesCar = []
        self.setOfPriceslorry = []

        self.pricesEmptyCar = []
        self.pricesLowCar = []
        self.pricesIntermediateCar = []
        self.pricesPackedCar = []
        self.pricesFullCar = []

        self.pricesEmptyLorry = []
        self.pricesLowLorry = []
        self.pricesIntermediateLorry = []
        self.pricesPackedLorry = []
        self.pricesFullLorry = []

        self.pricePropensitiesCar = []
        self.priceProbsCar = []

        self.pricePropensitieslorry = []
        self.priceProbslorry = []

        self.memory_param = 0.1
        self.experimental_param = 0.1

        self.priceChoiceCar = 0
        self.priceChoicelorry = 0

        self.choiceCar = 0
        self.stateChoiceCar = None

        self.choicelorry = 0
        self.stateChoicelorry = None

        self.currentState = None

        self.statesCar = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.choicesDict = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.statesProbCar = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.stateslorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.choicesDictlorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }
        self.statesProblorry = {
            'Empty': [],
            'Low': [],
            'Intermediate': [],
            'Packed': [],
            'Full': []
        }

    def calculateSpeed(self):
        n = 2
        m = 2
        speed_val = self.initialSpeed*((1-self.density/self.max)**m)**n
        self.currentSpeed = round(speed_val,3)
        if self.currentSpeed <= 0:
            self.timeOfTravel = self.roadLength*10
        else:
            self.timeOfTravel = self.roadLength/self.currentSpeed
            self.timeOfTravel = round(self.timeOfTravel*10,3)
        print("Speed value {}".format(self.currentSpeed))
        print("Time value {}".format(self.timeOfTravel))

    def getState(self):
        density_var = (self.density/self.max)*100
        if density_var >= 0 and density_var <= 20:
            self.currentState = 'Empty'
        elif density_var > 20 and density_var <= 40:
            self.currentState = 'Low'
        elif density_var > 40 and density_var <=60:
            self.currentState = 'Intermediate'
        elif density_var > 60 and density_var <=80:
            self.currentState = 'Packed'
        else:
            self.currentState = 'Full'
        print("density {}, State {}".format(self.density,self.currentState))

    def updateStates(self,reward):
        updated_states = []
        i = 0
        for index, elem in enumerate(self.statesCar[self.currentState]):
            if index == self.choiceCar:
                print("elem {}".format(elem))
                elem = (1 - self.memory_param) * elem + (reward * (1 - self.experimental_param))
            else:
                elem = (1 - self.memory_param) * elem + (elem * (self.experimental_param / (len(self.statesCar[self.currentState]) - 1)))
            updated_states.insert(i, elem)
            i += 1
        print("Old states {}".format(self.statesCar))
        self.statesCar[self.currentState] = updated_states
        print("Updated states {}".format(self.statesCar))

    def updateStateslorry(self,reward):
        updated_states = []
        i = 0
        for index, elem in enumerate(self.stateslorry[self.currentState]):
            if index == self.choicelorry:
                print("elem {}".format(elem))
                elem = (1 - self.memory_param) * elem + (reward * (1 - self.experimental_param))
            else:
                elem = (1 - self.memory_param) * elem + (elem * (self.experimental_param / (len(self.statesCar[self.currentState]) - 1)))
            updated_states.insert(i, elem)
            i += 1
        print("Old states {}".format(self.stateslorry))
        self.stateslorry[self.currentState] = updated_states
        print("Updated states {}".format(self.stateslorry))

    def updateStateProbabilities(self):
        propensitiesList = self.statesCar[self.currentState]
        probs = []
        propSum = sum(propensitiesList)
        for index, elem in enumerate(propensitiesList):
            elem = elem / propSum
            probs.insert(index, elem)
        self.statesProbCar[self.currentState] = probs
        print("Probabilities {}".format(self.statesProbCar))

    def updateStateProbabilitieslorry(self):
        propensitiesList = self.stateslorry[self.currentState]
        probs = []
        propSum = sum(propensitiesList)
        for index, elem in enumerate(propensitiesList):
            elem = elem / propSum
            probs.insert(index, elem)
        self.statesProblorry[self.currentState] = probs
        print("Probabilities {}".format(self.statesProblorry))


    def addPriceBasedOnStateCar(self):
        if self.currentState == 'Empty':
            self.pricesEmptyCar.append(self.price)
        elif self.currentState == 'Low':
            self.pricesLowCar.append(self.price)
        elif self.currentState == 'Intermediate':
            self.pricesIntermediateCar.append(self.price)
        elif self.currentState == 'Packed':
            self.pricesPackedCar.append(self.price)
        else:
            self.pricesFullCar.append(self.price)

    def addPriceBasedOnStateLorry(self):
        if self.currentState == 'Empty':
            self.pricesEmptyLorry.append(self.pricelorry)
        elif self.currentState == 'Low':
            self.pricesLowLorry.append(self.pricelorry)
        elif self.currentState == 'Intermediate':
            self.pricesIntermediateLorry.append(self.pricelorry)
        elif self.currentState == 'Packed':
            self.pricesPackedLorry.append(self.pricelorry)
        else:
            self.pricesFullLorry.append(self.pricelorry)

    def makeChoice(self):
        choicesList = self.choicesDict[self.currentState]
        choiceProbList = self.statesProbCar[self.currentState]
        choiceVar = np.random.choice(choicesList, p=choiceProbList)
        self.stateChoiceCar = choiceVar
        self.choiceCar = choicesList.index(choiceVar)
        self.price = self.stateChoiceCar

        self.addPriceBasedOnStateCar()
        print("Choice {}, Index {}".format(self.stateChoiceCar, choicesList.index(choiceVar)))

    def makeChoicelorry(self):
        choicesList = self.choicesDictlorry[self.currentState]
        choiceProbList = self.statesProblorry[self.currentState]
        choiceVar = np.random.choice(choicesList, p=choiceProbList)
        self.stateChoicelorry = choiceVar
        self.choicelorry = choicesList.index(choiceVar)
        self.pricelorry = self.stateChoicelorry

        self.addPriceBasedOnStateLorry()
        print("Choice lorry {}, Index {}".format(self.stateChoicelorry, choicesList.index(choiceVar)))

    def updateValues(self):
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for state in stateElements:
            for index, elem in enumerate(self.statesCar[state]):
                self.statesCar[state][index] = elem * 0.9
                self.statesCar[state][index] += 0.1
            for index, elem in enumerate(self.stateslorry[state]):
                self.stateslorry[state][index] = elem * 0.9
                self.stateslorry[state][index] += 0.1


    def setPriceStrategies(self):
        self.setOfPricesCar = []
        self.setOfPriceslorry = []
        for i in range(44, 60, 1):
            self.setOfPricesCar.append(i)
        for i in range(132, 164, 1):
            self.setOfPriceslorry.append(i)

        stateElements = ['Empty','Low','Intermediate','Packed','Full']
        for elem in stateElements:
            self.choicesDict[elem] = self.setOfPricesCar
            self.choicesDictlorry[elem] = self.setOfPriceslorry

    def setPricePropensities(self):
        self.pricePropensitiesCar = []
        self.pricePropensitieslorry= []
        for i in range(len(self.setOfPricesCar)):
            self.pricePropensitiesCar.append(1)
        for i in range(len(self.setOfPriceslorry)):
            self.pricePropensitieslorry.append(1)
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for elem in stateElements:
            self.statesCar[elem] = self.pricePropensitiesCar
            self.stateslorry[elem] = self.pricePropensitieslorry
        self.initialPropensityValuesCar = self.pricePropensitiesCar
        self.initialPropensityValueslorry = self.pricePropensitieslorry

    def setPriceProbabilities(self):
        self.priceProbsCar = []
        self.priceProbslorry = []
        for i in range(len(self.setOfPricesCar)):
            self.priceProbsCar.append(1 / len(self.setOfPricesCar))
        for i in range(len(self.setOfPriceslorry)):
            self.priceProbslorry.append(1 / len(self.setOfPriceslorry))
        stateElements = ['Empty', 'Low', 'Intermediate', 'Packed', 'Full']
        for elem in stateElements:
            self.statesProbCar[elem] = self.priceProbsCar
            self.statesProblorry[elem] = self.priceProbslorry

    def prepareData(self):
        if self.readyToSell:
            if self.initial_step:
                self.setPriceStrategies()
                self.setPricePropensities()
                self.setPriceProbabilities()
                self.initial_step = False

    def checkIfReadyToSell(self):
        if self.open:
            self.readyToSell = True
        print("Ready to Sell {}".format(self.readyToSell))
        print("Capacity {}".format(self.maxCapacity))

    def updateQueue(self):
        self.queue = []
        self.density = 0
        self.maxCapacity = self.max

    def setStatus(self):
        if self.hour >= 7 and self.hour <= 9 and self.emergencyOpen is False:
            self.open = False
            self.readyToSell = False
        else:
            self.open = True
            self.readyToSell = True
            self.emergencyOpen = False
        print("Status {}".format(self.open))

    def checkIfRushHour(self):
        if self.hour >= 7 and self.hour <= 9:
            self.rushHour = True
        elif self.hour >= 15 and self.hour <= 17:
            self.rushHour = True
        else:
            self.rushHour = False
        print("Rush Hour {}".format(self.rushHour))

    def name_func(self):
        print("Agent {}, length {}".format(self.unique_id,self.roadLength))


    def checkMainGate(self):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent)):
                self.mainGatedensity = agent.density
                self.mainGateCapacity = agent.max

    def checkIfOpen(self,density,max_capacity):
        self.open = np.random.choice([True,False],p=[density/max_capacity,1-(density/max_capacity)])
        self.emergencyOpen = self.open
        self.readyToSell = self.open
        print("Agent {} is open: {}".format(self.unique_id,self.readyToSell))

    def step(self):
        self.name_func()
        self.updateQueue()
        self.setStatus()
        self.checkIfReadyToSell()
        self.checkIfRushHour()

        self.getState()
        self.calculateSpeed()

        self.prepareData()
        self.makeChoice()
        self.makeChoicelorry()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

