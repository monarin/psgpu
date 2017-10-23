# !!! NOTE: None is returned whenever requested information is missing.

# import
import psana

# retreive parameters from psana etc.
dsname = 'exp=cxid9114:run=95'
src = 'CxiDs2.0:Cspad.0' # or its alias 'cspad'

ds  = psana.DataSource(dsname)
env = ds.env()
evt = ds.events().next()
runnum = evt.run()

# parameter par can be either runnum or evt    
par = runnum # or = evt
cmpars=(1,25,10,91) # custom tuple of common mode parameters

det = psana.Detector(src, env, pbits=0)

# or directly
from Detector.AreaDetector import AreaDetector    
det = AreaDetector(src, env, pbits=0)

# set parameters, if changed
det.set_env(env)
det.set_source(source)
det.set_print_bits(pbits)

# for Camera type of detector only
det.set_do_offset(do_offset=False) # NOTE: should be called right after det object is created, before getting data

# print info
det.print_attributes()    
det.print_config(evt)

# get pixel array shape, size, and nomber of dimensions
shape = det.shape(par=0)
size  = det.size(par=0)
ndim  = det.ndim(par=0)
instrument = det.instrument()

# access intensity calibration parameters
peds   = det.pedestals(par) # returns array of pixel pedestals from calib store type pedestals
rms    = det.rms(par)       # returns array of pixel dark noise rms from calib store type pixel_rms
gain   = det.gain(par)      # returns array of pixel gain from calib store type pixel_gain
bkgd   = det.bkgd(par)      # returns array of pixel background from calib store type pixel_bkgd
status = det.status(par)    # returns array of pixel status from calib store type pixel_status
stmask = det.status_as_mask(par, mode=0) # returns array of masked bad pixels in det.status 
                                         # mode=0/1/2 masks zero/four/eight neighbors around each bad pixel
mask   = det.mask_calib(par)  # returns array of pixel mask from calib store type pixel_mask
cmod   = det.common_mode(par) # returns 1-d array of common mode parameters from calib store type common_mode

# per-pixel (int16) gain mask from configuration data; 1/0 for low/high gain pixels,
# or (float) per-pixel gain factors if gain is not None
gmap = det.gain_mask(par, gain=None) # returns array of pixel gains using configuration data
gmnz = det.gain_mask_non_zero(par, gain=None) # returns None if ALL pixels have high gain and mask should not be applied

# set gfactor=high/low gain factor for CSPAD(2X2) in det.calib and det.image methods
det.set_gain_mask_factor(gfactor=6.85)

# set flag (for Chuck)
det.do_reshape_2d_to_3d(flag=False) 

# get raw data
nda_raw = det.raw(evt)

# get calibrated data (applied corrections: pedestals, common mode, gain mask, gain, pixel status mask)
nda_cdata = det.calib(evt)
# and with custom common mode parameter sequence
nda_cdata = det.calib(evt, cmpars=(1,25,10,91)) # see description of common mode algorithms in confluence,
# and with combined mask.
nda_cdata = det.calib(evt, mbits=1) # see description of det.mask_comb method.
