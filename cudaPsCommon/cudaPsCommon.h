
// ---------------------- peak finder expose -------------------------

const long EVENTS = 100;
// const long EVENTS = 5;
const int MAX_PEAKS = 150;
const long SHOTS = 32;
const long WIDTH = 388;
const long HEIGHT = 185;
const long LSIZE = EVENTS * SHOTS * WIDTH * HEIGHT;

const int rank = 4;
const float thr_high = 150;
const float thr_low = 10;
const float r0 = 5;
const float dr = 0.05;
const int HALF_WIDTH = (int)(r0 + dr);
const float peak_npix_min = 2;
const float peak_npix_max = 50;
const float peak_amax_thr = 10;
const float peak_atot_thr = 20;
const float peak_son_min = 5;

struct Peak{
  bool valid;
  float evt;
  float seg;
  float row;
  float col;
  float npix;
  float npos;
  float amp_max;
  float amp_tot;
  float row_cgrav;
  float col_cgrav;
  float row_sigma;
  float col_sigma;
  float row_min;
  float row_max;
  float col_min;
  float col_max;
  float bkgd;
  float noise;
  float son;
};

struct Win{
  int top;
  int bot;
  int left;
  int right;
  Win():top(0),bot(HEIGHT),left(0),right(WIDTH){}
};

// parameters for filterByThrHigh 
const int FILTER_PATCH_WIDTH = 32;
const int FILTER_PATCH_HEIGHT = 4;
const int FILTER_THREADS_PER_PATCH = FILTER_PATCH_WIDTH * FILTER_PATCH_HEIGHT;
const int FILTER_PATCH_ON_WIDTH = (WIDTH) / FILTER_PATCH_WIDTH;
const int FILTER_PATCH_ON_HEIGHT = (HEIGHT + FILTER_PATCH_HEIGHT - 1) / FILTER_PATCH_HEIGHT;
const int FILTER_PATCH_PER_SECTOR = FILTER_PATCH_ON_WIDTH * FILTER_PATCH_ON_HEIGHT;

// exposed functions
__global__ void floodFill_v2(const float *d_data, const uint *d_centers, Peak *d_peaks, uint *d_conmap, int offset, int nEvents);
__global__ void filterByThrHigh_v2(const float *d_data, uint *d_centers, int offset);
