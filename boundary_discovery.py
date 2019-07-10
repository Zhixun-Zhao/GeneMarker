from math import log
import numpy as np


def calc_ent(Feature, splitpoint, Label):
    CLabel_count = [0, 0, 0, 0]
    probs = [0, 0, 0, 0]
    Fea = Feature.tolist()
    for feature in Feature:
        if feature <= splitpoint:
            if Label[Fea.index(feature)] == 0:
                CLabel_count[0] += 1
            else:
                CLabel_count[1] += 1
        else:
            if Label[Fea.index(feature)] == 0:
                CLabel_count[2] += 1
            else:
                CLabel_count[3] += 1

    probs[0] = CLabel_count[0] / (CLabel_count[0] + CLabel_count[1])
    probs[1] = CLabel_count[1] / (CLabel_count[0] + CLabel_count[1])
    probs[2] = CLabel_count[2] / (CLabel_count[2] + CLabel_count[3])
    probs[3] = CLabel_count[3] / (CLabel_count[2] + CLabel_count[3])

    below_ratio = (CLabel_count[0] + CLabel_count[1]) / len(Feature)
    above_ratio = (CLabel_count[2] + CLabel_count[3]) / len(Feature)

    if probs[0] != 0:
        shannon0 = - below_ratio * probs[0] * log(probs[0], 2)
    else:
        shannon0 = 0

    if probs[1] != 0:
        shannon1 = - below_ratio * probs[1] * log(probs[1], 2)
    else:
        shannon1 = 0

    if probs[2] != 0:
        shannon2 = - above_ratio * probs[2] * log(probs[2], 2)
    else:
        shannon2 = 0

    if probs[3] != 0:
        shannon3 = - above_ratio * probs[3] * log(probs[3], 2)
    else:
        shannon3 = 0

    shannon_ent = shannon0 + shannon1 + shannon2 + shannon3
    splitinfo = -below_ratio * log(below_ratio, 2) - above_ratio * log(below_ratio, 2)
    return shannon_ent, splitinfo


def Boundary_discovery(Feature, Label, baseEntropy):
    bestinfoGain=0
    bestInfoGainRatio = 0
    bestpoint = -1
    vector_orignal = sorted(Feature)
    splitpoint = []
    if vector_orignal[-1] == 0:
        bestInfoGainRatio = 0
        bestpoint = 0
    else:
        vector = list(set(vector_orignal))
        for i in range(len(vector)):
            if i < len(vector) - 1:
                splitpoint.append((vector[i] + vector[i + 1]) / 2)
        for j in range(len(splitpoint)):
            newEntropy, splitInfo = calc_ent(Feature, splitpoint[j], Label)
            infoGain = baseEntropy - newEntropy
            infoGainRatio = infoGain / splitInfo
            if (infoGainRatio > bestInfoGainRatio):
                bestInfoGainRatio = infoGainRatio
                bestpoint = splitpoint[j]
                bestinfoGain = infoGain
    return bestinfoGain, bestInfoGainRatio, bestpoint


hea = np.loadtxt('../healthy.txt', dtype=float)
H = hea.T
tum = np.loadtxt('../tumor.txt', dtype=float)
T = tum.T

Data = np.row_stack((H, T))
label = []
for i in range(len(Data)):
    if i < len(H):
        label.append(0)
    else:
        label.append(1)
label = np.array(label)

baseEntropy = -(len(H)/len(Data)) * log(len(H)/len(Data), 2) -(len(T)/len(Data)) * log(len(T)/len(Data), 2)

fw = open(r'Entropy_HT.txt', 'w')
for i in range(len(hea)):
    bestInfoGain, bestInfoGainRatio, bestpoint = Boundary_discovery(Data[:, i], label, baseEntropy)
    fw.write(str(i+1) + '  ' + str(bestInfoGain) + '  ' + str(bestInfoGainRatio) + '  ' + str(bestpoint) + '\n')
fw.close()
