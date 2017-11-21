#include <stdio.h>
#include <stdlib.h>


/* left is the index of the leftmost element of the subarray; right is one
 * past the index of the rightmost element */
void merge_helper(float *input, int left, int right, float *scratch)
{
  /* base case: one element */
  if (right == left + 1) {
    
    return;
  } else {
    
    int i = 0;
    int length = right - left;
    int midpoint_distance = length / 2;
    /* l and r are to the positions in the left and right subarrays */
    int l = left, r = left + midpoint_distance;

    /* sort each subarray */
    merge_helper(input, left, left + midpoint_distance, scratch);
    merge_helper(input, left + midpoint_distance, right, scratch);

    /* merge the arrays together using scratch for temporary storage */
    for (i = 0; i < length; i++) {
      /* Check to see if any elements remain in the left array; if so,
       * we check if there are any elements left in the right array; if
       * so, we compare them. Otherwise, we know that the merge must
       * take the element from the left array */
      if (l < left + midpoint_distance &&
          (r == right || max(input[l], input[r]) == input[l])) {
        
        scratch[i] = input[l];
        l++;
      } else {

        scratch[i] = input[r];
        r++;
      }
    }   
    
    /* Copy the sorted subarray back to the input */
    for(i = left; i < right; i++) {
      input[i] = scratch[i - left];
    }
  }
}

void host_calc(float *a, float *dark, int nPixels, int cmmThr) {
  // host calculation
  struct timeval start, end;

  long seconds, useconds;
  double mtime;

  gettimeofday(&start, NULL);
  
  // dark
  for(int i = 0; i < nPixels; i++)
    a[i] -= dark[i];

  // common mode 
  float *sectorMedian = (float *)malloc(MAX_QUADS * MAX_SECTORS * sizeof(float));
  for (int i = 0; i < MAX_QUADS * MAX_SECTORS; i++) {
    
    int offset = i * SECTOR_SIZE;
    
    // select only this sector and sort this sector
    float *sector = (float *)malloc(SECTOR_SIZE * sizeof(float));
    for (int j = 0; j < SECTOR_SIZE; j++) {
      sector[j] = a[offset + j]; 
    }

    //printf("\n");
    //printf("s[0]=%6.2f, s[1]=%6.2f, s[2]=%6.2f\n", sector[0], sector[1], sector[2]);
    
    float *scratch = (float *)malloc(SECTOR_SIZE * sizeof(float));
    if (scratch != NULL) {

      merge_helper(sector, 0, SECTOR_SIZE, scratch);
      free(scratch);
      printf("%6.2f, %6.2f, %6.2f ... %6.2f, %6.2f, %6.2f\n", sector[0], sector[1], sector[2], sector[SECTOR_SIZE-3], sector[SECTOR_SIZE-2], sector[SECTOR_SIZE-1]); 
    } 

    // apply the threshold
    int foundPos = 0;
    for (int j = SECTOR_SIZE - 1; j >= 0; j--) {
      if (sector[j] <= cmmThr) {
        foundPos = j;
        break;
      }
      if (j == 0) foundPos = SECTOR_SIZE - 1;
    }   
    
    // calculate median
    if(foundPos%2 == 0) {
      sectorMedian[i] = (sector[foundPos/2] + sector[foundPos/2 - 1]) / 2.0;
    } else {
      sectorMedian[i] = sector[foundPos/2];
    } 
    free(sector);
    printf("sector: %d foundPos: %d med: %6.4f \n", i, foundPos, sectorMedian[i]); 
    
  }

  // apply common mode
  for(int i=0; i < nPixels; i++) {
    int iSector = floor(i / SECTOR_SIZE);
    a[i] -= sectorMedian[iSector];
  }
  
  gettimeofday(&end, NULL);

  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  mtime = ((seconds) * 1000000 + useconds)/1000.0;// + 0.5;

  cout << "Host dark-subtraction and common mode took "<< mtime <<" ms for 1 event."<< endl;
}
