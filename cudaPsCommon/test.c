#include <stdio.h>

#define MaxSectors 8
#define SectorSize 71780


void fill( short *p, int n, float val ) {
  int i = 0;
  for(i = 0; i < n; i++){
    p[i] = val;
  }
}

void processQuad(unsigned qNum, const short* data, short* corrdata, float* common_mode)
{
  int seg = 0;
  int m_ampThr = 30;
  int sect = 0;
  int i = 0;
  for (sect = 0; sect < MaxSectors; ++ sect) {
    
    double sum = 0;
    int npix = 0;
    const short* segData = data + seg*SectorSize;

    for (i =0; i < SectorSize; ++ i) {
      if (segData[i] > m_ampThr ) continue;
        sum += (double) segData[i];
        ++ npix;
    }
    common_mode[sect] = (float) sum/npix;
    short average = (short) common_mode[seg];
    
    printf("sect=%d sum=%6.2f npix=%d common_mode[sect]=%6.2f average=%d\n", sect, sum, npix, common_mode[sect], average);

    short* corrData = corrdata + seg*SectorSize;

    for (i = 0; i < SectorSize; ++ i) {
      corrData[i] = segData[i] - average;
    }
    ++seg;
  }
}

int main(int argc, char **argv)
{
  short *data, *corrdata;
  data = malloc(SectorSize * MaxSectors * sizeof(short));
  corrdata = malloc(SectorSize * MaxSectors * sizeof(short));
  fill(data, SectorSize * MaxSectors, 1);
  fill(corrdata, SectorSize * MaxSectors, 0);
  printf("%d %d %d\n", data[0], data[1], data[2]);
  printf("%d %d %d\n", corrdata[0], corrdata[1], corrdata[2]);
  float *common_mode;
  common_mode = malloc(MaxSectors * sizeof(float));
  int i = 0;
  for (i = 0; i < MaxSectors; i++) common_mode[i] = 0.0;
  for (i = 0; i < MaxSectors; i++) printf("%6.2f ", common_mode[i]);
  printf("\n");
  processQuad(0, data, corrdata, common_mode);
}
