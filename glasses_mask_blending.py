from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

def taper_mask(fr,
               rl,
               ll,
               outer_dilation_factor,
               outer_blur_factor,
               inner_blur_factor,
               plot=False):
  
  ### Dilate frames mask
  flip_fr = fr==0
  dilated = gaussian_filter(input = flip_fr, 
                          sigma=outer_dilation_factor, 
                          order=0, 
                          output=int, 
                            mode='reflect', 
                          cval=0.0, 
                            truncate=4.0)

  ### Blur Frame mask
  flip_dilated =  dilated==0
  blured_mask = gaussian_filter(input = flip_dilated, 
                                sigma=outer_blur_factor, 
                                order=0, 
                                output=float, 
                                  mode='reflect', 
                                  cval=0.0, 
                                  truncate=4.0)

  ### Remove extra blur from lense area
  blurred_mask_lens_corrected = blured_mask.copy()
  blurred_mask_lens_corrected[rl] = 0
  blurred_mask_lens_corrected[ll] = 0

  ### Add blur to inside of lenses
  final_mask = gaussian_filter(input = blurred_mask_lens_corrected, 
                                sigma=inner_blur_factor, 
                                order=0, 
                                output=float, 
                                  mode='reflect', 
                                  cval=0.0, 
                                  truncate=4.0)

  ### Add in lenses to dilated frames
  mask_lens_added = flip_dilated.copy()
  mask_lens_added[rl] = 1
  mask_lens_added[ll] = 1

  ### Blur the full glasses mask
  tapered_full_glasses_mask = gaussian_filter(input = mask_lens_added, 
                                sigma=outer_blur_factor, 
                                order=0, 
                                output=float, 
                                  mode='reflect', 
                                  cval=0.0, 
                                  truncate=4.0)


  if plot:
    plt.figure(dpi=100)
    plt.imshow(fr)

    plt.figure(dpi=100)
    plt.imshow(dilated)

    plt.figure(dpi=100)
    plt.imshow(blured_mask)

    plt.figure(dpi=100)
    plt.imshow(blurred_mask_lens_corrected)

    plt.figure(dpi=100)
    plt.imshow(final_mask)

    plt.figure(dpi=100)
    plt.imshow(tapered_full_glasses_mask)

  tapered_frames_mask = final_mask

  return tapered_frames_mask, tapered_full_glasses_mask 
  
  
  
  
def blend_images_with_tapered_mask(orig, mod, final_mask):
  '''
  orig: 3d int in range 0-255
  mod: 3d int in range 0-255
  final_mask: 2d float in range 0-1

  returns blend: 3d int in range 0-255
  '''
  if orig.shape[0:2] != final_mask.shape[0:2]:
    final_mask = resize(final_mask, output_shape=orig.shape[0:2])
    
  final_mask = np.expand_dims(final_mask, axis=-1)
  blend =  (1-final_mask) * orig + final_mask * mod 
  blend = blend.astype(orig.dtype)

  return blend
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  