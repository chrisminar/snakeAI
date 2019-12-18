#write a thing that takes the best thing every 
#change the values to not be negative numbers for certain square types -1,0,1 -> 1,2,3
#change get random neurons to get recursive neurons
  #could just throw away any attempted new connection that causes a recurse
  #computationally hard to generate the depth chart

#todo write load function
#todo make an gui for the AI snake
#write funciton doesnt work

#fitness should be run multiple times per genome

#todo is score of 10000 an appropriate exit condition?
  #consider changing the snake scoring
#todo consider giving snake body length information
#todo parallelize the main evaluation loop
#rotate y axis text

from typing import List,Tuple
import typing
from snake import snake
from copy import deepcopy
import pickle
import operator
import numpy as np
import time
import pygame as pg
from random import seed
from random import random
from random import randint

gridX = 8
gridY = 8
seed(np.floor(time.time()))
populationSize = 300

DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 20

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

innovation = 0

buttons = {0: "up",
           1: "right",
           2: "down",
           3: "left",
           4: "none"}

class node():
  def __init__(self):
    self.value = 0 #feed forward value
    self.bias = 0 #node bias

class neuron():
  def __init__(self):
    self.inputs = [] #list of gene input indexes
    self.ID = (0,0) #(layer,index)
    self.bias = random()*4-2 #random bias
    return

  def setValues(self, ID:Tuple[int]):
    self.ID=ID

class gene():
  def __init__(self, in_Idx_ = -1, out_Idx_ = -1, ID_ = -1):
    self.in_Idx  = in_Idx_ #input neuron id (layer,index)
    self.out_Idx = out_Idx_ #output neuron id (layer,index)
    self.id      = ID_ #unique genome identifier
    self.weight  = 0.0 #weight
    self.enabled = True #is enabled
    self.addedFrom = ""
    return

  ## return hard copy of this gene
  def returnCopy(self):
    return deepcopy(self)

class network():
  def __init__(self, genes_:List[gene], neurons_:List[neuron], sizeX:int=gridX, sizeY:int=gridY):
    self.nodes = [[node() for _ in range(len(neurons_[0]))],[node() for _ in range(len(neurons_[1]))],[node() for _ in range(len(neurons_[2]))]] #[[inputs][hidden][outputs]]
    self.geneLevels = [0]*len(genes_) #depth of each gene
    self.nodeLevels = [0]*len(self.nodes[1]) #depth of each neuron
    self.genes = genes_ #list of genes
    self.neurons = neurons_

    # set node biases from incoming neurons
    for i in range(len(self.nodes)): #for each layertype
      for j in range(len(self.nodes[i])): #for each neuron id
        self.nodes[i][j].bias = neurons_[i][j].bias #set bias

    # establish depth of each gene
    for i in range(len(self.genes)): #for each gene
      self.geneLevels[i] = self.recurseToInput(i) #follow input chain until you get to an input layer

    # establish depth of each hidden neuron
    for j in range(len(self.nodes[1])):
      #for each input gene, get max level
      maxLvl = 0 #deepest gene
      for geneIdx in neurons_[1][j].inputs: #step through genes that come into this neuron
        if self.geneLevels[geneIdx] > maxLvl: #if this gene has the highest level seen so far
          maxLvl = self.geneLevels[geneIdx] # set new highest level
      self.nodeLevels[j] = maxLvl #set max level

    # sort by index
    self.orderIndex = np.argsort(np.array(self.nodeLevels)) #get sorted index from lowest to highest (index0 -> lowest nodelevel, indexn -> highest nodelevel)
    return

  ## for any given geneid, trace the longest gene path to an input layer
  def recurseToInput(self, geneIdx:int):
    inNeuronLayer = self.genes[geneIdx].in_Idx[0] #input neuron layer
    inNeuronId    = self.genes[geneIdx].in_Idx[1] #input neruon id
    if inNeuronLayer == 0: #you have reached the input layer, return
      return 0
    numGenesToInNeuron = len(self.neurons[inNeuronLayer][inNeuronId].inputs) #number of genes coming into the input neuron
    maxLevel = 0
    #if numGenesToInNeuron > 1:
    #  print ("geneidx:", geneIdx, "inneuronID:",(inNeuronLayer, inNeuronId), "in neuron inputs:", self.neurons[inNeuronLayer][inNeuronId].inputs, "geneid" ,id(self.genes[geneIdx]), "inneuron addr:", id(self.neurons[inNeuronLayer][inNeuronId]))
    #  for i in range(len(self.neurons[inNeuronLayer][inNeuronId].inputs)):
    #    print("\t gene{}: {}".format(i, self.neurons[inNeuronLayer][inNeuronId].inputs[i]), id(self.genes[self.neurons[inNeuronLayer][inNeuronId].inputs[i]]),
    #          "inputs: {}".format(self.genes[self.neurons[inNeuronLayer][inNeuronId].inputs[i]].in_Idx) )

    for i in range(numGenesToInNeuron): #step through input genes to input neuron
      lenToInput = self.recurseToInput(self.neurons[inNeuronLayer][inNeuronId].inputs[i]) + 1 #length of lower level gene +1
      if lenToInput > maxLevel: #if this is the new maximum
        maxLevel = lenToInput #set it
    return maxLevel

  ## sigmoid function
  def sigmoid(self, x:float):
    return 2/(1+np.exp(-4.9*x))-1

  ## feedforward network
  def evaluate(self):
    global buttons
    #hidden layers
    for i in range(len(self.orderIndex)):
      sum = self.nodes[1][self.orderIndex[i]].bias #add bias
      for j in self.neurons[1][self.orderIndex[i]].inputs: #step through genes that go into this neuron
        if self.genes[j].enabled:
          inLayer = self.genes[j].in_Idx[0] #genes input neuron layer
          inId    = self.genes[j].in_Idx[1] #genes input neuron id
          sum += self.genes[j].weight * self.nodes[inLayer][inId].value #gene wieght * gene input neuron value

      if j>0:
        self.nodes[1][self.orderIndex[i]].value = self.sigmoid(sum) #sigmoid +relu?

    #output layers
    for i in range(len(buttons)):
      sum = self.nodes[2][i].bias #add bias
      for j in self.neurons[2][i].inputs:
        if self.genes[j].enabled:
          inLayer = self.genes[j].in_Idx[0] #genes input neuron layer
          inId    = self.genes[j].in_Idx[1] #genes input neuron id
          sum += self.genes[j].weight * self.nodes[inLayer][inId].value #gene wieght * gene input neuron value
      self.nodes[2][i].value = sum
    maxID = 0

    #get max output id
    for i in range(1,len(buttons)):
      if self.nodes[2][maxID].value < self.nodes[2][i].value:
        maxID = i
    return buttons[maxID] #return highest

  def setInputs(self, snakeInstance:snake):
    for i in range(gridX): #for each input
      for j in range(gridY):
        idx = i*gridX+j
        if snakeInstance.grid[i][j] == -1: #empty
          self.nodes[0][idx].value = 0
        elif snakeInstance.grid[i][j] > 0: #snake body
          self.nodes[0][idx].value = 2 #food is -1, empty is 0, body is 1 to n
        elif snakeInstance.grid[i][j] == 0: #head
          self.nodes[0][idx].value = 1
        else: #food
          self.nodes[0][idx].value =-1

class genome():
  def __init__(self, gx:int, gy:int):
    self.gridX = gx
    self.gridY = gy
    self.neurons = [[neuron() for _ in range(gx*gy)],[],[neuron() for _ in range(len(buttons))]] #[[inputs][hidden][outputs]]
    self.genes = [] #genes (node connections)
    self.fitness = 0 #how well does this genome solve the cost function
    self.network = [] #neural network
    self.globalRank = 0 #global ranking (higher is better)
    global MutateMutationChance
    global LinkMutationChance
    global BiasMutationChance
    global NodeMutationChance
    global EnableMutationChance
    global DisableMutationChance
    global StepSize
    self.mutationRates = {"connections": MutateConnectionsChance, #chance to mutate varios things
                          "link": LinkMutationChance,
                          "bias": BiasMutationChance,
                          "node": NodeMutationChance,
                          "enable": EnableMutationChance,
                          "disable": DisableMutationChance,
                          "step": StepSize}
    self.initNeurons()
    self.mutate() #mutate new genome
    return

  ## initialize input and output neurons
  def initNeurons(self):
    global buttons
    #traverse input snake grid in reading order
    for j in range(gridY): # for input grid y
      for i in range(gridX): # for input grid x
        self.neurons[0][j*gridX+i].setValues(ID = (0,j*gridX+i)) # set neruon to input value, set ID
    #traverse output buttons
    for i in range(len(buttons)):
      self.neurons[2][i].ID = (2,i) #set neuron to output, set id
    return

  ## mutate gene weight
  def weightMutate(self):
    global PerturbChance
    step = self.mutationRates["step"] #get step mutation
    for gene_ in self.genes: #for each gene
      if random()<PerturbChance: #roll the die
        gene_.weight += step*(random()*2 -1) #slightly change
      else:
        gene_.weight = random()*4-2 #reset to new random value
    return

  #mutate node biases
  def biasMutate(self):
    global PerturbChance
    step = self.mutationRates["step"] #get step mutation
    for neuron_ in self.neurons[1]: #mutate hidden neurons
      if random()<PerturbChance:
        neuron_.bias += step*(random()*2-1) #slightly change
      else:
        neuron_.bias = random()*4-2 #reset to new random value
    for neuron_ in self.neurons[2]: #mutate output neurons
      if random()<PerturbChance:
        neuron_.bias += step*(random()*2-1) #slightly change
      else:
        neuron_.bias = random()*4-2 #reset to new random value
    return

  ## return a random input and output neruon from this genome
  def randomNeurons(self):
    #get an input index, can not include outputs
    maxID = len(self.neurons[0]) + len(self.neurons[1]) #inputs + hiddens
    tempId = randint(0,maxID-1) #get random id
    if tempId < len(self.neurons[0]):#id is is input neuron
      input = (0,tempId) #output type, id 
    else: #id is hidden neuron
      input = (1,tempId-len(self.neurons[0]))
    maxID = len(self.neurons[1]) + len(self.neurons[2])# hidden+output
    doneFlag = False
    while not doneFlag: #get a random input until we don't recurse ##???
      tempId = randint(0,maxID-1)#get random id
      if tempId < len(self.neurons[1]): #id is hidden neuron
        output = (1,tempId)
      else: #id is output neuron
        output = (2,tempId-len(self.neurons[1]))
      if output != input:
        doneFlag = True
    return input,output

  ## check if link exists in genome
  def containsLink(self, link:gene):
    for geneID in self.neurons[link.out_Idx[0]][link.out_Idx[1]].inputs: #step through gene ids in the input list
      if self.genes[geneID].in_Idx == link.in_Idx: # one of the inputs matches the links input
        return True
    return False

  ## add a new gene to the genome
  def addGeneMutate(self):
    global innovation
    neuron1,neuron2 = self.randomNeurons() #get two random neruons

    newLink = gene() #init new gene (link)
    newLink.in_Idx  = neuron1 #set input to new gene as neruon 1
    newLink.out_Idx = neuron2 #set output to new gene as neuron 2
    newLink.addedFrom = "addGeneMutate"

    if self.containsLink(newLink): #if this link already exists, give up
      return
    innovation += 1 #increment innovation number
    newLink.id = innovation #set innovation number
    newLink.weight = random()*4-2 #init weight
    self.genes.append(newLink) #add gene to genome
    outPutNeuronLayer = neuron2[0] #layer of output neuron
    outPutNeuronId = neuron2[1] #id of output neuron
    self.neurons[outPutNeuronLayer][outPutNeuronId].inputs.append(len(self.genes)-1) #connect output neuron to link
    return

  ## add a mutated neuron
  def addNeuronMutate(self):
    global innovation
    if not self.genes: #list is empty
      return #give up

    gene1_ = self.genes[randint(0,len(self.genes)-1)].returnCopy() #get copy of random gene, this will be the input
    if not gene1_.enabled: #if gene is disabled give up
      return
    gene2_ = gene1_.returnCopy() #make a second copy, this will be the output
    gene1_.addedFrom = "addNeuronMutate1"
    gene2_.addedFrom = "addNeuronMutate2"

    self.genes.append(gene1_) #add input gene
    self.genes.append(gene2_) #add output gene

    #change neruon to accound for new gene coming in
    self.neurons[1].append(neuron()) #add new neuron to hidden layer list
    self.neurons[1][-1].inputs = [len(self.genes)-2] #init neuron input list with input synapse
    self.neurons[1][-1].ID = (1,len(self.neurons[1])-1) #set neuron id #??? needed?

    #change input gene to account for different output neuron
    gene1_.out_Idx = self.neurons[1][-1].ID #set output neuron id
    gene1_.enabled = True #enable
    innovation += 1 #increment innovation
    gene1_.id = innovation #set innovation

    #change output gene to account for different input neuron
    gene2_.in_Idx = self.neurons[1][-1].ID #set input neuron id
    gene2_.enabled = True #enable
    innovation += 1 #increment innovation
    gene2_.id = innovation #set id

    #change output genes out neuron to include it
    self.neurons[gene2_.out_Idx[0]][gene2_.out_Idx[1]].inputs.append(len(self.genes)-1)

    return

  ## check if gene/neuron sturcture is ok
  def checkIntegrity(self):
    geneIndex = 0
    for gene_ in self.genes:
      neuronOutLayer = gene_.out_Idx[0]
      neuronOutID = gene_.out_Idx[1]
      if len(self.neurons[neuronOutLayer]) > neuronOutID:
        neuron_ = self.neurons[neuronOutLayer][neuronOutID]
        nFound = False
        for input_ in neuron_.inputs:
          if input_ == geneIndex:
            nFound = True
            break
        if not nFound:
          return False
      else:
        return False
      geneIndex +=1
    return True

  ## mutate(toggle) a synapse's (genes) existance
  def enableDisableMutate(self, isEnable:bool):
    candidates = []
    for i in range(len(self.genes)): #find all candidates that are disabled(enabled)
      if not self.genes[i].enabled == isEnable:
        candidates.append(i) #add to list

    if not candidates: #if there are no candidates, give up
      return

    idx = randint(0,len(candidates)-1) #gt random index
    self.genes[candidates[idx]].enabled = not self.genes[candidates[idx]].enabled #swap index
    return

  ## call all the genome mutate functions
  def mutate(self):
    #update mutation rates
    for key in self.mutationRates: #for all mutation rates
      if random()>0.5: #roll the die
        self.mutationRates[key] *= 0.95 #decrease rate
      else:
        self.mutationRates[key] *= 1.05263 #increase rate

    #consider mutating gene weight
    if random() <self.mutationRates["connections"]:
      self.weightMutate()

    #add new genes
    p = self.mutationRates["link"]
    while p>0:
      if random()<p:
        self.addGeneMutate()
      p-=1

    #mutate neuron bias
    p = self.mutationRates["bias"]
    while p>0:
      if random()<p:
        self.biasMutate()
      p-=1

    #add new neurons
    p = self.mutationRates["node"]
    while p>0:
      if random()<p:
        self.addNeuronMutate()
      p-=1

    #mutate enabledness
    p = self.mutationRates["enable"]
    while p>0:
      if random()<p:
        self.enableDisableMutate(True)
      p-=1

    #mutate disabledness
    p = self.mutationRates["disable"]
    while p>0:
      if random()<p:
        self.enableDisableMutate(False)
      p-=1
    return

  ## return a hard copy of this genome
  def returnCopy(self):
    return deepcopy(self)

  ## generate network
  def generateNetwork(self):
    self.network = network(self.genes, self.neurons, self.gridX, self.gridY)

class species():
  def __init__(self):
    self.topFitness = 0 #best genome fitness of species
    self.staleness = 0 #how many generations since species improvement
    self.genomes = [] #genome list
    self.averageFitness = 0 #average fitness of genome

  ## calculate average fitness of species
  def calculateAverageFitness(self): #todo this doesn't really calculate the average fitness
    total = 0
    for genome_ in self.genomes: #for all genomes
      total += genome_.globalRank #sum global rank
    self.averageFitness = total/len(self.genomes) #average global rank (higher is better)
    return

  ## mate two genomes, return child
  def crossOver(self, g1: genome, g2: genome):
    if g2.fitness > g1.fitness:
      g2,g1 = g1,g2 #set highest fitness to g1

    global gridX
    global gridY

    child = g1.returnCopy() #init new genome

    innovations = []
    for gene_ in g1.genes: #for all genes in genome1
      innovations.append(gene_.id) #add id to list

    for i in range(len(g2.genes)):
      geneIndex = np.where(innovations==g2.genes[i].id) #find first matching value
      if geneIndex[0].size == 0: #if there is no matching value
        gene2Exists=False
      else:
        gene2Exists=True
      if gene2Exists and random()>0.5 and g2.genes[i].enabled: #if genes match, random die roll, and gene2 is enabled
        child.genes[geneIndex].addedFrom += "crossover"
        child.genes[geneIndex].weight = g2.genes[i].weight #copy gene weight from g2

    return child

  ## def addGeneToChild(self, child, gene_):
  #  return
    #gene inputid
    #geneID
    #gene outputid
    #does input neuron exist already?
      #add neuron
      #add gene to neuron input list
    #set gene input neuron
    #set gene output neuron
    

  ## breed two random genomes to get a child genome, appends child to children list
  def breedChild(self, children):
    global CrossoverChance
    if random() < CrossoverChance: #roll die
      g1 = self.genomes[randint(0, len(self.genomes)-1)] #get random genome1
      g2 = self.genomes[randint(0, len(self.genomes)-1)] #get random genome2
      child = self.crossOver(g1, g2) #mate genomes
    else:
      g = self.genomes[randint(0, len(self.genomes)-1)] #get random genome
      child = g.returnCopy() #copy
    
    child.mutate()
    children.append(child)
    return

class pool():
  def __init__(self, gx, gy):
    self.gridX = gx
    self.gridY = gy
    self.species = []
    self.generation = 0            # current generation iteration
    self.maxFitness = 0            #maximum genome fitness
    self.fitnessHistory = [0]      #maximume gneome fitnes of each generation
    self.currentGenome = 0         # current genome being evaluated
    self.currentSpecies = 0        # current species being evaluated
    self.generationTime = 0        # time to evaluate last generation
    self.genomeTime = 0            # time to evaluate last genome
    self.speciesTime = 0           # time to evaluate last species
    global populationSize
    for i in range(populationSize): #for population size
      self.addToSpecies(genome(self.gridX, self.gridY))  # add new genome
    pg.init() #init pygame
    self.DISPLAY = pg.display.set_mode((1600,400),0,32, 1)#init display
    pg.display.set_caption('SnakeTrainer') #set pygame screen caption
    self.font = pg.font.Font('freesansbold.ttf',12) #make font
    self.updateGenerationGUI()
    return

  ## write pygame text
  def drawText(self, txt:str, pos:Tuple[int], color=(0,0,0)):
    text = self.font.render(txt, True, color) #set text
    textRect = text.get_rect()
    textRect.center = (pos) #change position of text
    self.DISPLAY.blit(text,textRect) #display text
    return

  ## update genome part of gui
  def updateGenomeGUI(self):
    pg.draw.rect(self.DISPLAY, (200,200,200), (20,  70, 190,  20)) #draw genome num square
    pg.draw.rect(self.DISPLAY, (200,200,200), (210, 70, 190,  20)) #draw genome time square
    pg.draw.rect(self.DISPLAY, (200,200,200), (410, 70, 380, 20)) #draw genome progress square
    self.drawText('Genome #{}'.format(self.currentGenome), (105,80)) #draw genome #
    self.drawText('Genome time {:0.2f}s'.format(self.genomeTime),(305,80)) #draw genome time
    ratioG = self.currentGenome/len(self.species[self.currentSpecies].genomes) #ration of genomes completed of species
    pg.draw.rect(self.DISPLAY, (100,100,100), (415, 70, 370*ratioG, 20)) #draw genome progress square
    self.drawText('{:0.2f}%'.format(ratioG*100), (600,80)) #draw genome progress %
    pg.display.update() #update what is shown
    return

  ## update species part of gui
  def updateSpeciesGUI(self):
    pg.draw.rect(self.DISPLAY, (200,200,200), (20,  40, 190,  20)) #draw species num square
    pg.draw.rect(self.DISPLAY, (200,200,200), (210, 40, 190,  20)) #draw species time square
    pg.draw.rect(self.DISPLAY, (200,200,200), (410, 40, 380, 20)) #draw species progress square
    self.drawText('Species #{}'.format(self.currentSpecies), (105,50)) #draw species number
    self.drawText('Species time {:0.2f}s'.format(self.speciesTime),(305,50)) #write time to complete previous species
    ratioS = self.currentSpecies/len(self.species) #ration of species completed out of genome
    pg.draw.rect(self.DISPLAY, (100,100,100), (410, 40, 380*ratioS, 20)) #draw species progress square
    self.drawText('{:0.2f}%'.format(ratioS*100), (600,50)) #draw species progress %
    pg.display.update() #update display
    return

  ## update fitness plot
  def updatePlot(self):
    pg.draw.rect(self.DISPLAY, (200,200,200), (50, 100, 700, 250)) #draw fitness plot bg
    #figure out x and y scales
    maxY = np.max(self.fitnessHistory)
    minY = np.min(self.fitnessHistory)
    maxX = len(self.fitnessHistory)
    color = (0,0,0) #black
    for i in range(9): #divide axises by 8
      pg.draw.line(self.DISPLAY, color, (i*700/8+50,340), (i*700/8+50, 350)) #x ticks
      pg.draw.line(self.DISPLAY, color, (50,350-i*250/8), (60,350-i*250/8)) #y ticks
      self.drawText('{:0.0f}'.format(i*maxX/8), (i*700/8+50, 360), color) #draw x tick labels
      self.drawText('{:0.0f}'.format(i*(maxY-minY)/8), (25, 350-i*250/8), color) #draw y tick labels
    self.drawText('Fitness', (775, 225), color)#draw x axis label
    self.drawText('Generation #', (400,370), color)#draw y axis label
    lpix = 50 #left
    rpix = 750 #right
    bpix = 350 #bot
    tpix = 100 #top
    for i in range(len(self.fitnessHistory)-1):#draw fitness
      x1 = i/maxX
      x2 = (i+1)/maxX
      y1 = (self.fitnessHistory[i]-minY)/(maxY-minY)
      y2 = (self.fitnessHistory[i+1]-minY)/(maxY-minY)
      X1,X2 = self.convertXToPix(x1,x2, lpix,rpix)
      Y1,Y2 = self.convertXToPix(y1,y2, bpix,tpix)
      pg.draw.line(self.DISPLAY, color, (X1,Y1), (X2, Y2)) #lines
    return

  def convertXToPix(self, x1,x2, left,right):
    X1 = left + x1*(right-left)
    X2 = left + x2*(right-left)
    return X1,X2

  def convertYToPix(self, y1,y2, bot,top):
    Y1 = bot - (bot-top)*y1 #convert y to pixels
    Y2 = bot - (bot-top)*y2
    return Y1,Y2

  ## update generation part of gui
  def updateGenerationGUI(self):
    self.DISPLAY.fill((255,255,255)) #fill with white
    pg.draw.rect(self.DISPLAY, (200,200,200), (20,  10, 190,  20)) #draw generation text background square
    pg.draw.rect(self.DISPLAY, (200,200,200), (210, 10, 190,  20)) #draw generation time background square
    self.drawText('Generation #{}'.format(self.generation), (105,20)) #draw current generation #
    self.drawText('Generation time {:0.2f}s'.format(self.generationTime), (305,20)) #draw genration time
    self.updatePlot() #update fitness plot
    self.updateSpeciesGUI()
    #self.updateGenomeGUI()
    return

  def drawGame(self, grid):
    for i in range(self.gridX):
      for j in range(self.gridY):
        pg.draw.rect(self.DISPLAY, self.gridNum2Color(grid[i][j]), (i*21+800, j*21, 20, 20))

  def gridNum2Color(self, num):
    if num == -2:#food
      return (255,0,0)
    elif num == 0:#head
      return (0,0,0)
    elif num >0:#tail
      return (100,100,100)
    else:#background
      return (200, 200, 200)

  #draw neuron netowrk
  def drawNetwork(self, grid, network_:network):
    levelCounter = []
    xlev = [[],[],[]]
    ylev = [[],[],[]]
    #add input layer x+y
    for i in range(gridX): #for each input
      for j in range(gridY):
        xlev[0].append(i*21+800)
        ylev[0].append(j*21)

    #draw hidden layer neuron squares
    for i in range(len(network_.neurons[1])):
      if len(levelCounter) <= network_.nodeLevels[i]:
        levelCounter.append(0)
      else:
        levelCounter[network_.nodeLevels[i]] += 1
      xl=network_.nodeLevels[i]
      yl=levelCounter[network_.nodeLevels[i]]
      xlev[1].append(30*xl+1300)
      ylev[1].append(21*yl)
      pg.draw.rect(self.DISPLAY, (100,100,100), (xlev[1][-1], ylev[1][-1], 20,20))

    #draw output layer squares
    if len(network_.nodeLevels) == 0:
      xl = 1
    else:
      xl = max(network_.nodeLevels) + 1
    maxval = 0
    for i in range(len(network_.neurons[2])):
      if  network_.nodes[2][maxval].value < network_.nodes[2][i].value:
        maxval = i
    for i in range(len(network_.neurons[2])):
      xlev[2].append(30*xl+1300)
      ylev[2].append(21*i)
      if i== maxval:
        col = (230,230,230)
      else:
        col = (100,100,100)
      pg.draw.rect(self.DISPLAY, col, (xlev[2][-1], ylev[2][-1], 20,20))

    #draw genes
    for i in range(len(network_.genes)):
      inLayer = network_.genes[i].in_Idx[0]
      inID = network_.genes[i].in_Idx[1]
      outLayer = network_.genes[i].out_Idx[0]
      outID = network_.genes[i].out_Idx[1]
      x1 = xlev[inLayer][inID]
      y1 = ylev[inLayer][inID]
      x2 = xlev[outLayer][outID]
      y2 = ylev[outLayer][outID]
      pg.draw.line(self.DISPLAY, (100,200,100), (x1+10,y1+10),(x2+10,y2+10))


  ## draw genome
  def drawBestGenome(self):
    self.species.sort(key=operator.attrgetter('topFitness'), reverse=True) #sort species by fitness (highest first)
    species_ = self.species[0]
    snakeInstance = snake(False, self.gridX, self.gridY)
    self.drawGame(snakeInstance.grid) #draw grid
    self.drawNetwork(snakeInstance.gridNum2Color,species_.genomes[0].network) #draw network
    while not snakeInstance.gameover:
      species_.genomes[0].network.setInputs(snakeInstance)
      newGeneration = species_.genomes[0].network.evaluate()
      snakeInstance.aiRunStep(newGeneration)
      self.drawGame(snakeInstance.grid) #draw grid
      self.drawNetwork(snakeInstance.gridNum2Color,species_.genomes[0].network) #draw network
      pg.display.update()
      pg.time.wait(500)
    self.updateGenerationGUI


  ## evaluate fitness of a genome
  def evaluateFitnessOfGenome(self, genome_: genome):
    genome_.generateNetwork() #generate neural network in feedforward format
    values = np.zeros(10)
    for i in range(10):
      snakeInstance = snake(False, gridX, gridY) #make new snake ai game instance
      while not snakeInstance.gameover: #while the game has not ended
        genome_.network.setInputs(snakeInstance) #convert snake to neural net inputs
        newDirection = genome_.network.evaluate()
        snakeInstance.aiRunStep(newDirection)
      values[i] = snakeInstance.score
    genome_.fitness = values.mean()
    return

  ## evaluate fitness of all genomes
  def evaluateFitnessOfAll(self):
    self.currentSpecies = 0 # reset current species
    for species_ in self.species: #loop though species
      self.currentGenome = 0 # reset current genome
      speciesStartTime = time.time() #start species time
      for genome_ in species_.genomes: #loop through genomes
        genomeStartTime = time.time() #start timing genome
        genome_.network = network(genome_.genes, genome_.neurons, self.gridX, self.gridY) #init network
        self.evaluateFitnessOfGenome(genome_) #evalaute genome
        self.genomeTime = time.time()-genomeStartTime #end genome time
        #self.updateGenomeGUI #update gui
        self.currentGenome += 1 #increment genome for gui
      self.speciesTime = time.time()-speciesStartTime #end species time
      self.updateSpeciesGUI() #update gui
      self.currentSpecies += 1 #increment species for gui
    return

  ## remove all lower half of genomes from all species
  def cullSpecies(self, cutToOne: bool):
    for species_ in self.species: #sort each species genomes by fitness
      species_.genomes.sort(key=operator.attrgetter('fitness'), reverse=True) #sort genomes by attribute fitness (highest to lowest)
    
      if cutToOne: #if we want to remove everything but one genome
        startCull = 1 #start position to remove from
      else: #otherwise, remove half
        startCull = max(np.floor(len(species_.genomes)/2).astype(int),1) #if there was only 1 genome to start with, don't delete it
      endCull = len(species_.genomes) #end position to remove from

      for i in reversed(range(startCull,endCull)): #for all genomes in the cull range
        del species_.genomes[i] #delete genome
    return

  ## rank all genomes fitness, set maxfitness
  def rankGlobally(self):
    globalRank = []
    for species_ in self.species: #for all species
      for genome_ in species_.genomes: #for all genomes
        globalRank.append(genome_.fitness) #get fitness

    gr = np.argsort(np.array(globalRank)) #get index of sorted global rank (0 lowest, len(globalrank)-1 highest)
    counter = 0
    for species_ in self.species:
      for genome_ in species_.genomes:
        genome_.globalRank = gr[counter] #set rank in genome ... globalRank (0 is worst rank, the higher the rank, the better)
        counter += 1

    self.maxFitness = np.max(globalRank)#set max pool fitness
    return

  ## remove stales species from population
  def removeStaleSpecies(self):
    global StaleSpecies
    for species_ in reversed(self.species): # for all species
      species_.genomes.sort(key=operator.attrgetter('fitness'), reverse=True) #sort genomes by fitness

      if species_.genomes[0].fitness>species_.topFitness: #if this is the top fitness of the species
        species_.topFitness = species_.genomes[0].fitness #set top fitness
        species_.staleness = 0 #reset staleness
      else: #otherwise
        species_.staleness += 1 #increment staleness

      if species_.staleness < StaleSpecies or species_.topFitness >= self.maxFitness: # if not stale, or the best species in the pool, keep it
        pass
      else: #otherwise, cull it
        del species_
    return

  ## sum fitness of all species
  def sumAverageFitness(self):
    total=0
    for species_ in self.species: #for all species
      total += species_.averageFitness #add average fitness
    return total

  ## remove weaker half of species from populaiton
  def removeWeakSpecies(self):
    global populationSize
    sumFit = self.sumAverageFitness() #sum the fitness of all species
    for species_ in reversed(self.species): #for all species
      avgFit = sumFit/populationSize #average species fitness
      breed = np.floor( species_.averageFitness /avgFit ) #ratio of current species fitness to average species fitness
      if breed < 1: #if belove average
        del species_ #cull
    return

  ## number of genes that don't match between genomes / number of genes in genome
  def disjoint(self, genes1: List[gene], genes2: List[gene]):
    i1 = np.zeros(len(genes1)) #make an array that is #genes long
    for i in range(len(genes1)): #for each gene
      i1[i] = genes1[i].id #get gene innovation

    i2 = np.zeros(len(genes2)) #make an array that is #genes long
    for i in range(len(genes2)): #for each gene
      i2[i] = genes2[i].id #get gene innovation

    disjointNum = 0 #number of disjoints
    n = max(len(genes1),len(genes2)) #total number of genes

    #look for matches in genome 1
    for gene_ in genes1: #for each gene in genes1
      doesMatch = np.any(i2==gene_.id) #try to find gene2 id in gene1 list
      if not doesMatch: #no match?
        disjointNum += 1 #increment distjoint

    #look for mismatch in geneome2
    for gene_ in genes2: #for each gene in genes2
      doesMatch = np.any(i1==gene_.id) #try to find gene2 id in gene1 list
      if not doesMatch: #no match?
        disjointNum += 1 #increment distjoint

    return disjointNum/n #return ratio of disjoints

  ## compute a number for how different two genes weights are from each other
  def weights(self, genes1: List[gene], genes2: List[gene]):
    i1 = np.zeros(len(genes1)).astype(int) #make array that is #genes long
    for i in range(len(genes1)): #for each gene
      i1[i] = genes1[i].id #get gene innovation

    weightSum = 0 #sum of differnt weights
    n = 0 #number of genes

    for gene_ in genes2:
      matchId = np.where(i1==gene_.id) #get id of matching gene
      if matchId[0].size == 0: #there is no matching gene
        pass #do nothing
      else:
        n+=1 #increment matches
        weightSum += np.abs(gene_.weight - genes1[matchId[0][0]].weight) #increment weight differences
        
    if n==0:
      return 0
    else:
      return weightSum/n #return ratio

  ## do genome1 and genome2 belong to the same species?
  def sameSpecies(self, genome1: genome, genome2:genome):
    global DeltaDisjoint
    global DeltaWeights
    global DeltaThreshold
    deltaDisjointSum = DeltaDisjoint*self.disjoint(genome1.genes, genome2.genes) #disjoint ratio
    deltaWeightSum = DeltaWeights*self.weights(genome1.genes, genome2.genes) #weight ratio
    return deltaDisjointSum+deltaWeightSum < DeltaThreshold #below threshold?

  ## add child genome to the correct species
  def addToSpecies(self, child: genome):
    foundSpecies = False #has this species been found in existing species?
    for species_ in self.species: #loop through all species
      if not foundSpecies and self.sameSpecies(child, species_.genomes[0]): #there is only one genome in each species at this point, compare to it
        species_.genomes.append(child) #add child to this genome list
        foundSpecies = True # say we've found the species
        break #exit loop

    if not foundSpecies: #if the child doesn't match anything
      childSpecies = species() #make it a new species
      childSpecies.genomes.append(child) #add genome to new species
      self.species.append(childSpecies) #add species to species list
    return

  ## dump generation data to file
  def dumpToFile(self):
    name = "generation_" + str(self.generation) + ".txt"
    file = open(name,'w')
    pickle.dump(self, file)
    file.close()

  ## process changes for new generation
  def newGeneration(self):
    global populationSize
    self.cullSpecies(False) #remove the weakest half of genomes
    self.rankGlobally() #rank all genomes by fitness
    self.removeStaleSpecies() #remove species that haven't improved in a long time
    self.rankGlobally() #rank all genomes by fitness
    for species_ in self.species: #calculate average species fitness
      species_.calculateAverageFitness() # calculate average species fitness
    self.removeWeakSpecies() #remove weakest half of species (not nessicarily half)
    sumFit = self.sumAverageFitness() #sum fitness of all species
    children = []
    for species_ in self.species: #if the species is fit enough, make some babies
      avgFit = sumFit/populationSize #average species fitness
      breed = np.floor( species_.averageFitness /avgFit ).astype(int) #ratio of current species fitness to average species fitness
      for i in range(breed): #number of babies dependend on how much more fit than average it is
        species_.breedChild(children) #breed child
    self.cullSpecies(True) #remove all parent genomes except the strongest one
    while len(children) + len(self.species) < populationSize: #create new species from a random parent until we are at population cap
      species_ = self.species[randint(0,len(self.species)-1)] #get random species
      children.append(species_.breedChild())
    for child in children: #add children to pool
      self.addToSpecies(child)
    self.generation += 1 #increment generation
    #self.dumpToFile()
    self.fitnessHistory.append(self.maxFitness) #add fitness to history
    return

  ## training loop
  def run(self):
    while self.maxFitness < 10000: #end condition
      generationStartTime = time.time() #start timing this generation
      self.evaluateFitnessOfAll() #evaluate fitness of all genomes
      self.drawBestGenome()
      self.newGeneration() #breed new generation
      self.generationTime = time.time()-generationStartTime #end generation time
      self.updateGenerationGUI() #update gui
    return


if __name__ == "__main__":
  trainingPool = pool(gridX, gridY)
  trainingPool.run()