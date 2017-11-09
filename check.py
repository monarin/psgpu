import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

pedFile = '/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_pedCorrected_95.txt'
calibFile = '/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_calib_95.txt'

with open(pedFile, 'r') as f: ped = map(float, f.read().splitlines())
with open(calibFile, 'r') as f: calib = map(float, f.read().splitlines())

sectorSize = 185 * 388
nSectors = 4 * 8
cPar = (1,10,10,100)
cmmThr = 100

ped = np.array(ped)

calib = np.array(calib)
dif = ped - calib

sectorMed = np.zeros(nSectors)
trueSectorMed = np.zeros(nSectors)

for i in range(nSectors):
  offset = sectorSize * i
  sector = ped[offset:offset+sectorSize]
  filterSector = sector[sector < cmmThr]
  print i, len(filterSector), np.median(filterSector), np.mean(filterSector), np.max(filterSector), np.min(filterSector), dif[offset]
  sectorMed[i] = np.mean(filterSector)
  trueSectorMed[i] = dif[offset]
  sortedFilterSector = np.sort(filterSector)
  #print sortedFilterSector[0], sortedFilterSector[1], sortedFilterSector[2], sortedFilterSector[-3], sortedFilterSector[-2], sortedFilterSector[-1]
"""
plt.scatter(sectorMed, trueSectorMed)
plt.xlabel("Calculated")
plt.ylabel("True value")
plt.title("Comparision between methods")
plt.show()
"""

"""
calcCalib = np.zeros(len(ped))
for i in range(nSectors):
  offset = sectorSize * i
  for j in range(sectorSize):
    calcCalib[offset + j] = ped[offset +j] - sectorAvg[i]
    
print ped[0], ped[1], ped[2], ped[sectorSize], ped[sectorSize+1], ped[sectorSize+2]
print calib[0], calib[1], calib[2], calib[sectorSize], calib[sectorSize+1], calib[sectorSize+2]
print calcCalib[0], calcCalib[1], calcCalib[2], calcCalib[sectorSize], calcCalib[sectorSize+1], calcCalib[sectorSize+2]
"""  
