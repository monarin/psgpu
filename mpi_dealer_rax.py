from psana import *
import pickle

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'Dealer mode requires at least 2 ranks.'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exprun", help="psana experiment/run string (e.g. exp=xppd7114:run=43)")
parser.add_argument("-n","--noe",help="number of events, all events=1e10",default=-1, type=int)
args = parser.parse_args()

class ConvertToPyObj():
    def __init__(self,psanaOffset):
        self.filenames = psanaOffset.filenames()
        self.offsets = psanaOffset.offsets()
        self.lastBeginCalibCycleDgram = psanaOffset.lastBeginCalibCycleDgram()

def master():
    ds = DataSource(args.exprun+':smd:dir=/reg/neh/home/monarin/psnumba/xtc')
    for nevt, evt in enumerate(ds.events()):
        if nevt==args.noe: break
        offset = evt.get(EventOffset)
        rankreq = comm.recv(source=MPI.ANY_SOURCE)
        comm.send(ConvertToPyObj(offset),dest=rankreq)
    for rankreq in range(size-1):
        rankreq = comm.recv(source=MPI.ANY_SOURCE)
        comm.send('endrun',dest=rankreq)

def client():
    ds = DataSource(args.exprun+':rax')
    det = Detector('CxiDs2.0:Cspad.0')
    while True:
        comm.send(rank,dest=0)
        offset = comm.recv(source=0)
        if offset == 'endrun': break
        evt = ds.jump(offset.filenames, offset.offsets, offset.lastBeginCalibCycleDgram)
        print rank,evt.get(EventId)
        peds = det.pedestrals(evt)
        print type(peds)

if rank==0:
    master()
else:
    client()

print '*** Rank',rank,'completed ***'
