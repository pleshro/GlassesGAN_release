import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from tqdm import tqdm
import imageio
import joblib
import os
import shutil
#import torch
#import torchvision.transforms as transforms
#from shutil import copy
#import pandas as pd
#from PIL import Image, ImageDraw, ImageFont
#import PIL
#from IPython.display import Image as display_image
#from argparse import Namespace
#import time
#from cycler import cycler
#from scipy.ndimage import gaussian_filter
#from sklearn.decomposition import PCA, FastICA
#import random
#import matplotlib as mpl
#from matplotlib import cm
#from glob import glob

# sys.path.insert(0, '../DatasetGAN')
from glasses_mask_blending import taper_mask, blend_images_with_tapered_mask

cwd = os.getcwd()
os.chdir('./encoder4editing')
from e4e_encoder import e4e_encoder_inference
os.chdir(cwd)

sys.path.insert(0, './datasetGAN_release/datasetGAN')
from custom_deeplab_class import deeplab_segmenter_inference


### Glasses blending and segmentation functions
def replace_pixels_using_mask(orig, #original face image
                              mod, #modified face_image
                              mod_mask, #mask with locations of glasses
                              frame_color_bias=(1,1,1), #values to bias colors
                              left_lens_color_bias=(1,1,1), #values to bias colors
                              right_lens_color_bias=(1,1,1), #values to bias colors
                              gaussian_bias=True, #apply gaussian bias to make reflection
                              gaussian_bias_intensity=10,
                              use_tapered_mask=True,
                              use_full_glasses_or_frames_mask='frames',
                              outer_dilation_factor=5,
                              outer_blur_factor=25,
                              inner_blur_factor=5):
  
  assert orig.dtype == float
  assert mod.dtype == float
  assert mod_mask.dtype == int 
  
  if use_tapered_mask:
    # print(outer_dilation_factor)
    # print(outer_blur_factor)
    # print(inner_blur_factor)
    tapered_frames_mask, tapered_full_glasses_mask  = taper_mask(
                                  fr=mod_mask==1,
                                  rl=mod_mask==2,
                                  ll=mod_mask==3,
                                  outer_dilation_factor=outer_dilation_factor,
                                  outer_blur_factor=outer_blur_factor,
                                  inner_blur_factor=inner_blur_factor)
    if use_full_glasses_or_frames_mask=='frames':
      final_mask = tapered_frames_mask
    elif use_full_glasses_or_frames_mask=='full_glasses':
      final_mask = tapered_full_glasses_mask
    else:
      raise Exception('Options are "frames" or "full_glasses"')

    orig = blend_images_with_tapered_mask(orig=orig, 
                                          mod=mod, 
                                          final_mask=final_mask)

  else:
    ### Transfer frame pixels from modified face to original face
    orig[mod_mask == 1] = mod[mod_mask == 1]

  ### Bias the colors in the mask location
  ## 0:background
  ## 1:frames
  ## 2:right lens
  ## 3:left lens
  for mask_value, color_bias  in zip((1, 2, 3),(frame_color_bias, left_lens_color_bias, right_lens_color_bias)):
    rr,cc = np.where(mod_mask == mask_value)
    for chan, bias in zip((0, 1, 2), color_bias):
      orig[rr,cc,chan] = orig[rr,cc,chan] * bias


  if gaussian_bias:
    for lens_mask_value in 2,3:
      rr,cc = np.where(mod_mask == lens_mask_value)
      avg_lens_dim = int(np.sqrt(len(rr)))
      Z = generate_2d_gaussian(Xlen=avg_lens_dim*2, Ylen=avg_lens_dim*2) * gaussian_bias_intensity
      rr_norm = rr - np.min(rr)
      cc_norm = cc - np.min(cc)
      for chan in 0,1,2:
        orig[rr,cc,chan] = orig[rr,cc,chan] * Z[rr_norm,cc_norm]

  orig = np.clip(orig, 0, 1)

  return orig, final_mask



class glasses_vton_inference:
  def __init__(self, 
               fitted_pca_fp, 
               ave_add_glasses_diff_fp,
               base_working_dir,
               temp_save_folder,
               resume_model_ckpt,
               deeplab_script_location,
               chosen_deeplab_epoch,
               load_loc,
               use_full_glasses_or_frames_mask,
               run_tests=False,
               outer_dilation_factor=5,
               outer_blur_factor=25,
               inner_blur_factor=5,
               ideal_avg_glasses_frame_area = 0.020,
			   auto_clean=False):


    self.outer_dilation_factor=outer_dilation_factor
    self.outer_blur_factor=outer_blur_factor
    self.inner_blur_factor=inner_blur_factor
    self.ideal_avg_glasses_frame_area = ideal_avg_glasses_frame_area

    # self.norm_pc = np.load(norm_pc_fp) 

    self.fitted_pca =  joblib.load(fitted_pca_fp)

    self.ave_add_glasses_diff = np.load(ave_add_glasses_diff_fp)
    self.use_full_glasses_or_frames_mask = use_full_glasses_or_frames_mask

    self.e4e = e4e_encoder_inference(base_working_dir=f'{base_working_dir}/GlassesGAN_release', 
                                     load_loc=load_loc)

    self.DLseg = deeplab_segmenter_inference(
      temp_save_folder=temp_save_folder, #full path to temporary save folder
      resume_model_ckpt=resume_model_ckpt,
      deeplab_script_location=deeplab_script_location,
      chosen_deeplab_epoch=chosen_deeplab_epoch,
	  auto_clean=auto_clean)
    
    if run_tests:
      self.e4e.run_test_1()
      image_path = f'{load_loc}/cannon_test_images/IMG_3229.JPG'
      _, _, _, _, _, _, _ = self.e4e.run_test_2(image_path)
      self.DLseg.test_segmenter()

  def embed_input_image(self, image_path, show_plots=True):
    input_img = imageio.imread(image_path)
    input_image, transformed_image = self.e4e.pre_process_image(image_path, 
                                                                run_allign=True)
    result_image, base_latent= self.e4e.run_inference(transformed_image)

    if show_plots:
      plt.figure(dpi=150)
      plt.title('Original Image')
      plt.imshow(input_img)

      coupled_res, final_rec = self.e4e.visualize_results(result_image, transformed_image)
      plt.figure()
      plt.title('Cropped Original and Embedded Image')
      plt.imshow(coupled_res)

      plt.figure(dpi=100)
      plt.imshow(final_rec)

    return input_image, result_image, base_latent

  @staticmethod
  def get_glasses_area(seg_map):
    area = np.sum(seg_map==1) / seg_map.size
    return area

  def ret_mask_areas(self, latents):
    embedded_images = [np.array(self.e4e.run_generator(latent)[1]) 
                       for latent in latents]
    seg_maps = self.DLseg.run_segmentation(faces=embedded_images)
    #Calculate the areas of the image that are covered by glasses frames
    mask_areas = [self.get_glasses_area(seg_map) for seg_map in seg_maps]
    mask_areas = np.array(mask_areas)
    return mask_areas, embedded_images

  def add_avg_glasses(self, 
                      input_image, 
                      base_latent,
                      base_bias=1, 
                      show_plots=True,
                      auto_pick_bias=True,
                      return_blended_image=False):

    if auto_pick_bias:
      # ideal_avg_glasses_frame_area=0.02
      biases = np.linspace(base_bias/2,base_bias*1.5,20)
      start_latents = [base_latent + (self.ave_add_glasses_diff * bias) 
                      for bias in biases]
      mask_areas, embedded_images = self.ret_mask_areas(start_latents)
      min_index = np.argmin(np.abs(mask_areas - self.ideal_avg_glasses_frame_area))
      start_latent = start_latents[min_index]
      chosen_mask_area = mask_areas[min_index]
    else:
      start_latent = base_latent + (self.ave_add_glasses_diff * base_bias)
      chosen_mask_areas, _ = self.ret_mask_areas([start_latent])
      chosen_mask_area = chosen_mask_areas[0]

    _, embedded_image_pil = self.e4e.run_generator(start_latent)
    embedded_image = np.array(embedded_image_pil)

    if show_plots:
      if auto_pick_bias:
        print(f'bias linspace: {biases}')
        print(f'mask_areas {mask_areas.shape}')
        print(mask_areas)
        print(f'min index: {min_index}')
      input_image = np.array(input_image)
      seg_map = self.DLseg.run_segmentation(faces=[embedded_image])[0]
      blends, seg_maps, blur_masks = self.blend_in_edits(edits=[embedded_image], 
                          input_image=input_image)
      blend=blends[0]
      plt.figure(dpi=150)
      plt.title('Embedded image with normal glasses')
      plt.imshow(embedded_image_pil)
      plt.figure(dpi=150)
      plt.title('Blended image with normal glasses')
      plt.imshow(blend)
      plt.figure()
      plt.imshow(seg_map==1, cmap='gray')
      plt.figure()
      plt.imshow(1-(seg_map==1), cmap='gray')
      plt.figure()
      plt.imshow(np.squeeze(blur_masks),cmap='gray')
      plt.figure()
      plt.imshow(1-np.squeeze(blur_masks),cmap='gray')

      for bias, mask_area, embedded_image in zip(biases, mask_areas, embedded_images):
        plt.figure(dpi=150)
        plt.title(f'Vec Strength: {bias*100:.0f}% - Frames area: {mask_area:.3f} - Target: {self.ideal_avg_glasses_frame_area}', 
                  fontsize=9)
        plt.imshow(embedded_image)

    if return_blended_image:
      # seg_map = self.DLseg.run_segmentation(faces=[embedded_image])[0]
      edit_image = self.blend_in_edits(edits=[embedded_image], 
                                  input_image=input_image)[0][0]
    else:
      edit_image = embedded_image

    return start_latent, edit_image, chosen_mask_area

  def embedding_image_PC_direction_edits(self,
                         start_latent, 
                         PC_num, 
                         pca_vector_multipliers):
    ### Pre-compute range of outputs for plus or minus a certain latent vector magnitute 

    PCX_edit_images=[]
    PCX_edit_codes=[]
    for bias in pca_vector_multipliers:
      img, code = self.e4e.run_gen_add_pc_direction_bias(start_latent = start_latent, 
                                                         fitted_pca = self.fitted_pca, 
                                                         bias = bias, 
                                                         PC_num = PC_num) 
      PCX_edit_images.append(img)
      PCX_edit_codes.append(code)


    # PCX_edits = [ self.e4e.run_gen_add_pc_direction_bias(start_latent = start_latent, 
    #                                                      fitted_pca = self.fitted_pca, 
    #                                                      bias = bias, 
    #                                                      PC_num = PC_num) 
    #               for bias in pca_vector_multipliers]

    edit_desc = [f'PC: {PC_num} - Mag: {pca_vector_multiplier:.0f}' for pca_vector_multiplier in pca_vector_multipliers ]

    return PCX_edit_images, PCX_edit_codes, edit_desc

  def blend_in_edits(self, 
                     edits, 
                     input_image,
                     final_output_shape = (1024,1024,3)):
    
    input_image = np.array(input_image)

    assert input_image.dtype == np.uint8
    for edit in edits:
      assert edit.dtype == np.uint8

    #Enforce size of input image
    input_image = input_image.astype(float) / 255
    if input_image.shape != final_output_shape:
      # print('\n\n\nResizing input image')
      input_image = transform.resize(image=input_image, 
                                     output_shape=final_output_shape)

    #Run segmentation
    seg_maps = self.DLseg.run_segmentation(faces=edits)

    blends=[]
    blur_masks=[]
    for edit, seg_map in zip(edits, seg_maps):
      #Enforce size of segmentation map
      if seg_map.shape != (final_output_shape[0], final_output_shape[1]):
        # print('\n\n\nResizing seg map')
        seg_map = seg_map.astype(float) / 255
        seg_map = transform.resize(image=seg_map, 
                                   output_shape=(final_output_shape[0], 
                                                 final_output_shape[1]),
                                   preserve_range=False)
        seg_map = (seg_map*255)
        seg_map = seg_map.round()
        seg_map = seg_map.astype(int)

      #Enforce size of embedded image
      edit = edit.astype(float) / 255
      if edit.shape != final_output_shape:
        # print('\n\n\nResizing embedded image')
        edit = transform.resize(image=edit, 
                                output_shape=final_output_shape)

      #Perform blending
      blend, final_mask = replace_pixels_using_mask(
          input_image, #original face image
          mod=edit, #modified face_image
          mod_mask=seg_map, #mask with locations of glasses
          frame_color_bias=(1,1,1), #values to bias colors
          left_lens_color_bias=(1,1,1), #values to bias colors
          right_lens_color_bias=(1,1,1), #values to bias colors
          gaussian_bias=False, #apply gaussian bias to make reflection
          gaussian_bias_intensity=None,
          use_tapered_mask=True,
          use_full_glasses_or_frames_mask=self.use_full_glasses_or_frames_mask,
          outer_dilation_factor=self.outer_dilation_factor,
          outer_blur_factor=self.outer_blur_factor,
          inner_blur_factor=self.inner_blur_factor)

      blends.append(blend)
      blur_masks.append(final_mask)
    
    return blends, seg_maps, blur_masks

  def create_before_after_avg_glasses_dataset(self,
                                              image_paths, 
                                              image_ids, 
                                              save_dir,
                                              max_num_images,
                                              run_independent_chunks=False,
                                              chunk_number=0,
                                              num_splits=5,
                                              remove_subspace_tuning=False,
                                              remove_mask_blending=False):

    if run_independent_chunks:
      image_paths_chunks = list(self.split(image_paths, num_splits))
      image_ids_chunks = list(self.split(image_ids, num_splits))
      image_paths = image_paths_chunks[chunk_number]
      image_ids = image_ids_chunks[chunk_number]

    if os.path.exists(save_dir):
      if input('A clean folder is required to proceed. Remove old temporary files? (y/n): ') == 'y':
        print('\nRemoving temporary folder')
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=False)

    failures = []
    num_imgs=0
    for image_path, image_id in tqdm( zip(image_paths, image_ids), total=len(image_paths) ):

      print(f'Processing image: {image_id}')

      if num_imgs >= max_num_images:
        break

      try:
        input_image, result_image, base_latent = self.embed_input_image(
                                                              image_path, 
                                                              show_plots=False)
      except:
        print('Face preprocessing failed')
        failures.append(image_path)
        continue

      start_latent, blend, chosen_mask_area = self.add_avg_glasses(
                                input_image=input_image, 
                                base_latent=base_latent,
                                show_plots=False,
                                auto_pick_bias = not remove_subspace_tuning,
                                return_blended_image = not remove_mask_blending)

      orig_image_fp = f'{save_dir}/{image_id}_orig-none_0_glassesarea-none.png'
      blend_image_fp = f'{save_dir}/{image_id}_avg-clear_0_glassesarea-{chosen_mask_area:.3f}.png'

      imageio.imwrite(orig_image_fp, np.array(input_image))
      imageio.imwrite(blend_image_fp, blend)

      for pc_num in range(6):
        lower_bound, upper_bound = self.edit_bounds(pc_num)
        pca_vector_multiplier_set = [lower_bound*(2/3), upper_bound*(2/3)]
        for bias in pca_vector_multiplier_set:
          img, _ = self.e4e.run_gen_add_pc_direction_bias(start_latent = start_latent, 
                                                          fitted_pca = self.fitted_pca, 
                                                          bias = bias, 
                                                          PC_num = pc_num) 
          
          if remove_mask_blending:
            blend=img
            glasses_area = 0
          else:
            blends, seg_maps, _ = self.blend_in_edits(edits=[img], 
                                          input_image=input_image)
            blend=blends[0]
            seg_map = seg_maps[0]
            glasses_area = self.get_glasses_area(seg_map)
  
          blend_image_fp = f'{save_dir}/{image_id}_PC{pc_num}-clear_{bias:.1f}_glassesarea-{glasses_area:.3f}.png'
          imageio.imwrite(blend_image_fp, blend)

      num_imgs+=1

    return failures

  @staticmethod
  def split(a, n):
    '''
    Parameters
    ----------
    a : list
        List to split.
    n : int
        Number of parts.

    Returns
    -------
    list
        Split list into a list-of-lists with equal size chunks.

    '''
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))  

  @staticmethod
  def edit_bounds(pc_num,
                  return_edit_desc=False):
    # PC_nums = [0] #Shrink/Enlarge
    # lower_bound = -5
    # upper_bound = 10

    # PC_nums = [1] #Lower/Raise frames
    # lower_bound = -10
    # upper_bound = 4

    # PC_nums = [2] #Square/Round lenses
    # lower_bound = -4
    # upper_bound = 20

    # PC_nums = [3] #Shrink+Round/Enlarge+Square lenses
    # lower_bound = -5
    # upper_bound = 15

    # PC_nums = [4] #uncateye/cateye glasses
    # lower_bound = -15
    # upper_bound = 20

    # PC_nums = [5] #Thicken+Darken/Thin+Lighten frames
    # lower_bound = -10
    # upper_bound = 10

    bounds = {0: (-5,10),
              1: (-10,4),
              2: (-4,20),
              3: (-5,15),
              4: (-15,20),
              5: (-10,10)}

    edit_descs = {0: ('Shrink','Enlarge'),
                  1: ('Lower','Raise'),
                  2: ('Square','Round'),
                  3: ('Shrink/Round','Enlarge/Square'),
                  4: ('Reverse Cateye','Cateye'),
                  5: ('Thicken','Thin')}

    lower_bound, upper_bound = bounds[pc_num]

    if return_edit_desc:
      upper_edit_desc, lower_edit_desc  = edit_descs[pc_num]
      return lower_bound, upper_bound, lower_edit_desc, upper_edit_desc
    else:
      return lower_bound, upper_bound


  def apply_edits_in_stages(self, 
                            PC_nums, 
                            pca_vector_multiplier_sets, 
                            start_latent):
    PCX_edit_images=[]
    edit_desc=[]
    for stage_num, (PC_num, pca_vector_multipliers) in enumerate(zip(PC_nums, pca_vector_multiplier_sets)):

      if stage_num==0: #Use beginning latent 
        SL = start_latent
      else: #Use final latent code from last edit sequence
        SL = PCXEC[-1]

      PCEI, PCXEC, ED = self.embedding_image_PC_direction_edits(
                                      start_latent=SL, 
                                      PC_num=PC_num,
                                      pca_vector_multipliers=pca_vector_multipliers)
      
      ## Accumulate edit images and descriptions
      PCX_edit_images = PCX_edit_images + PCEI
      edit_desc = edit_desc + ED
    return PCX_edit_images, edit_desc






  def generate_images_of_specific_samples(self, image_ids):
    images=[]
    # image_ids = []

    for dataset, cnum_target, pcnum, mag in image_ids:
      fps = cpfs_all[dataset] 
      # print(cnums_all[dataset])
      # print(cnum_target)
      cnum_bool_index = [cnum_target == str(cnum) for cnum in cnums_all[dataset]]
      # print(cnum_bool_index)
      # print(np.sum(cnum_bool_index))
      assert np.sum(cnum_bool_index) == 1
      cnum_index = np.nonzero(cnum_bool_index)
      # print(cnum_index)
      assert len(cnum_index) == 1
      cnum_index = cnum_index[0][0]     
      image_path = fps[cnum_index]

      input_image, result_image, base_latent = self.embed_input_image(
                                                                  image_path, 
                                                                  show_plots=False)

      start_latent, _, _ = self.add_avg_glasses(input_image=input_image, 
                                                base_latent=base_latent,
                                                base_bias=1,
                                                show_plots=False,
                                                auto_pick_bias=True)

      img, _ = self.e4e.run_gen_add_pc_direction_bias(start_latent = start_latent, 
                                                      fitted_pca = self.fitted_pca, 
                                                      bias = mag, 
                                                      PC_num = pcnum) 
      
      blends, _, _ = self.blend_in_edits(edits=[img], 
                                        input_image=input_image)
      
      assert blends[0].shape == (1024,1024,3)

      images.append(blends[0])

      # plot_images_subplot(images=images, #list of images or image paths
      #               plots_tall_wide=None, #tuple with number of subplots tall and wide
      #               sub_titles=['']*len(images), 
      #               figsize = (6, 8), #(width, tall)
      #               dpi = 100,
      #               sharex = 'all',
      #               sharey = 'all',
      #               title='',
      #               remove_ticks=True,
      #               save_path_name = None,
      #               fill_images_columnwise=False,
      #               max_images_limit=100,
      #               subplot_title_text_size=5,
      #               use_tight_padding=True,
      #               use_col_row_names=False,
      #               col_names=None,
      #               row_names=None,
      #               col_name_font_weight=None,
      #               col_name_font_size=None,
      #               row_name_font_weight=None,
      #               row_name_font_size=None)

    return images


  def create_six_edit_subplots_figure(self, 
                                      image_path,
                                      savefig=False):

    input_image, result_image, base_latent = self.embed_input_image(
                                                                image_path, 
                                                                show_plots=False)

    start_latent, _, _ = self.add_avg_glasses(input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True)
    
    num_pcs = 6

    images=[]
    titles=[]
    for PC_num in range(num_pcs):
      lower_bound, upper_bound, lower_edit_desc, upper_edit_desc = self.edit_bounds(
                                                          pc_num=PC_num,
                                                          return_edit_desc=True)
      for edit_desc, bias in zip((lower_edit_desc, upper_edit_desc),(lower_bound, upper_bound)):
        img, _ = self.e4e.run_gen_add_pc_direction_bias(start_latent = start_latent, 
                                                        fitted_pca = self.fitted_pca, 
                                                        bias = bias, 
                                                        PC_num = PC_num) 
        blends, _, _ = self.blend_in_edits(edits=[img], 
                                          input_image=input_image)
        images.append(blends[0])
        titles.append(f'{edit_desc}')

    # plot_images_subplot(images, #list of images or image paths
    #                     plots_tall_wide=(2,num_pcs), #tuple with number of subplots tall and wide
    #                     sub_titles=titles, 
    #                     figsize = (1.0*num_pcs, 1.2*2), # (width, tall)
    #                     dpi = 500,
    #                     sharex = 'all',
    #                     sharey = 'all',
    #                     title='',
    #                     remove_ticks=True,
    #                     save_path_name = None,
    #                     fill_images_columnwise=True,
    #                     subplot_title_text_size=5)
    assert len(images) == num_pcs*2

    if savefig:
      save_path_name = (f'{base_working_dir}/sixeditfig_'
                        f'{image_path.split("/")[-1].split(".")[0]}.jpg')  
    else:
      save_path_name=None

    plot_images_subplot(images=images, #list of images or image paths
                        plots_tall_wide=(2, num_pcs), #tuple with number of subplots tall and wide
                        sub_titles=titles, 
                        figsize = (1.0*num_pcs, 1.2*2), #(width, tall)
                        dpi = 500,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = save_path_name,
                        fill_images_columnwise=True,
                        max_images_limit=100,
                        subplot_title_text_size=5,
                        use_tight_padding=True,
                        use_col_row_names=False,
                        col_names=None,
                        row_names=None,
                        col_name_font_weight=None,
                        col_name_font_size=None,
                        row_name_font_weight=None,
                        row_name_font_size=None)

    return images


  def create_average_glasses_plot(self, 
                                  image_paths,
                                  ave_add_glasses_diff_clear_fp,
                                  ave_add_glasses_diff_tint_fp):
    '''
    Show Average sunglasses and clearglasses for a few different people
    '''
    old_ave_add_glasses_diff = self.ave_add_glasses_diff.copy()
    old_use_full_glasses_or_frames_mask = self.use_full_glasses_or_frames_mask 

    images=[]
    # titles=[]
    col_names = ['Original','Clear','Tinted'] 

    for image_path in image_paths:

      input_image, result_image, base_latent = self.embed_input_image(
                                                                  image_path, 
                                                                  show_plots=False)
      
      # print(input_image.shape)
      images.append(np.array(input_image))
      # titles.append('Original')

      # descs=('Clear','Tinted')
      mask_types = ('frames','full_glasses')
      ave_add_glasses_diff_fps = (ave_add_glasses_diff_clear_fp, ave_add_glasses_diff_tint_fp)


      for mask_type, ave_add_glasses_diff_fp in zip(mask_types, ave_add_glasses_diff_fps):
        self.ave_add_glasses_diff = np.load(ave_add_glasses_diff_fp)
        self.use_full_glasses_or_frames_mask = mask_type

        _, blend, _ = self.add_avg_glasses(
                                        input_image=input_image, 
                                        base_latent=base_latent,
                                        base_bias=1,
                                        show_plots=False,
                                        auto_pick_bias=True,
                                        return_blended_image=True)
        images.append(blend)
        # titles.append(desc)

    self.ave_add_glasses_diff = old_ave_add_glasses_diff
    self.use_full_glasses_or_frames_mask = old_use_full_glasses_or_frames_mask

    # plot_images_subplot(images, #list of images or image paths
    #                     plots_tall_wide=(len(image_paths),3), #tuple with number of subplots tall and wide
    #                     sub_titles=titles, 
    #                     figsize = (1.66*3, 1.5*len(image_paths)), # (width, tall)
    #                     dpi = 500,
    #                     sharex = 'all',
    #                     sharey = 'all',
    #                     title='',
    #                     remove_ticks=True,
    #                     save_path_name = None,
    #                     fill_images_columnwise=False,
    #                     subplot_title_text_size=5)

    assert images[0].shape == (1024,1024,3)


    plot_images_subplot(images=images, #list of images or image paths
                        plots_tall_wide=((len(images)//3),3), #tuple with number of subplots tall and wide
                        sub_titles=['']*(len(images)//3)* 3, 
                        figsize = (2*3, 2*(len(images)//3)), #(width, tall)
                        dpi = 500,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = None,
                        fill_images_columnwise=False,
                        max_images_limit=100,
                        subplot_title_text_size=5,
                        use_tight_padding=True,
                        use_col_row_names=True,
                        col_names=col_names,
                        row_names=['']*(len(images)//3),
                        col_name_font_weight=None,
                        col_name_font_size=20,
                        row_name_font_weight=None,
                        row_name_font_size=None)


    return images

  def prep_images_for_continuous_multi_style_edit(self, 
                                                  image_path,
                                                  edit_directions_per_side,
                                                  blend=True):


    input_image, result_image, base_latent = self.embed_input_image(
                                                              image_path, 
                                                              show_plots=False)
    
    input_image = np.array(input_image)
    assert input_image.shape == (1024,1024,3)

    start_latent, _, _ = self.add_avg_glasses(input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True)
    
    # edit_directions_per_side = 3
    edit_steps = edit_directions_per_side * 2 + 1
    num_pcs=6

    edit_images=[]
    for pc_num in range(num_pcs):
      lower_bound, upper_bound, lower_edit_desc, upper_edit_desc = self.edit_bounds(
                                                        pc_num,
                                                        return_edit_desc=True)
      pca_vector_multipliers_lower = np.linspace(lower_bound, 0, edit_directions_per_side+1)
      pca_vector_multipliers_upper = np.linspace(0, upper_bound, edit_directions_per_side+1)
      pca_vector_multipliers = np.concatenate((pca_vector_multipliers_lower,
                                               pca_vector_multipliers_upper[1:]))
      
      # print(lower_bound)
      # print(upper_bound)
      # print(pca_vector_multipliers_lower)
      # print(pca_vector_multipliers_upper)
      # print(pca_vector_multipliers)

      assert pca_vector_multipliers.ndim == 1
      PCX_edit_images, _, edit_desc = self.embedding_image_PC_direction_edits(
                                start_latent=start_latent, 
                                PC_num=pc_num,
                                pca_vector_multipliers=pca_vector_multipliers)

      # print(edit_desc)
      if blend:
        blends, _, _ = self.blend_in_edits(edits=PCX_edit_images, 
                                          input_image=input_image)
      else:
        blends=PCX_edit_images

      assert type(blends) == list
      assert blends[0].shape == (1024,1024,3)
      assert len(blends) == edit_steps
      edit_images = edit_images + blends

      # raise Exception('done')

    assert len(edit_images) == (edit_steps * num_pcs)
    assert edit_images[0].shape == (1024,1024,3)

    return edit_images, num_pcs, edit_steps

  def create_continuous_multi_style_edit_doublecol_figure(self, 
                                                          image_path,
                                                          savefig=False):
    ## figure with rows as edit magnitude, cols as different edit directions

    edit_directions_per_side = 3

    edit_images, num_pcs, edit_steps = self.prep_images_for_continuous_multi_style_edit(
                                                       image_path,
                                                       edit_directions_per_side,
                                                       blend=True)

    col_names = ['---', '--', '-', 'Average Glasses', '+', '++', '+++']
    row_names = ['Size','Height','Squareness','Round/Shrink','Cateye','Thicken']

    if savefig:
      save_path_name = (f'{base_working_dir}/contmultiedit_'
                        f'{image_path.split("/")[-1].split(".")[0]}.png')  
    else:
      save_path_name=None

    plot_images_subplot(images=edit_images, #list of images or image paths
                        plots_tall_wide=(num_pcs, edit_steps), #tuple with number of subplots tall and wide
                        sub_titles=['']*edit_steps*num_pcs, 
                        figsize = (1.28*edit_steps, 1.3*num_pcs), #(width, tall)
                        dpi = 500,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = save_path_name,
                        fill_images_columnwise=False,
                        max_images_limit=100,
                        subplot_title_text_size=10,
                        use_tight_padding=True,
                        use_col_row_names=True,
                        col_names=col_names,
                        row_names=row_names,
                        col_name_font_weight=None,
                        col_name_font_size=None,
                        row_name_font_weight=None,
                        row_name_font_size=None)

    return


  def create_continuous_multi_style_edit_without_blending(self, 
                                                          image_path,
                                                          savefig=False):
    ## figure with rows as edit magnitude, cols as different edit directions

    edit_directions_per_side = 3

    input_image, result_image, base_latent = gvton.embed_input_image(image_path, 
                                                                show_plots=True)

    edit_images, num_pcs, edit_steps = self.prep_images_for_continuous_multi_style_edit(
                                                       image_path,
                                                       edit_directions_per_side,
                                                       blend=False)

    col_names = ['---', '--', '-', 'Average Glasses', '+', '++', '+++']
    row_names = ['Size','Height','Squareness','Round/Shrink','Cateye','Thicken']

    if savefig:
      save_path_name = (f'{base_working_dir}/sixeditfig_'
                        f'{image_path.split("/")[-1].split(".")[0]}.jpg')  
    else:
      save_path_name=None

    plot_images_subplot(images=edit_images, #list of images or image paths
                        plots_tall_wide=(num_pcs, edit_steps), #tuple with number of subplots tall and wide
                        sub_titles=['']*edit_steps*num_pcs, 
                        figsize = (1.28*edit_steps, 1.3*num_pcs), #(width, tall)
                        dpi = 250,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = save_path_name,
                        fill_images_columnwise=False,
                        max_images_limit=100,
                        subplot_title_text_size=10,
                        use_tight_padding=True,
                        use_col_row_names=True,
                        col_names=col_names,
                        row_names=row_names,
                        col_name_font_weight=None,
                        col_name_font_size=None,
                        row_name_font_weight=None,
                        row_name_font_size=None)

    return


  def create_continuous_multi_style_edit_singlecol_figure(self, 
                                                          image_path,
                                                          savefig=False):
    ## figure with rows as edit magnitude, cols as different edit directions

    edit_directions_per_side = 2

    edit_images, num_pcs, edit_steps = self.prep_images_for_continuous_multi_style_edit(
                                                       image_path,
                                                       edit_directions_per_side)

    col_names = ['--', '-', 'Average Glasses', '+', '++']
    row_names = ['Size','Height','Squareness','Round/Shrink','Cateye','Thicken']

    if savefig:
      save_path_name = (f'{base_working_dir}/contmultiedit_'
                        f'{image_path.split("/")[-1].split(".")[0]}.jpg')  
    else:
      save_path_name=None


    plot_images_subplot(images=edit_images, #list of images or image paths
                        plots_tall_wide=(num_pcs, edit_steps), #tuple with number of subplots tall and wide
                        sub_titles=['']*edit_steps*num_pcs, 
                        figsize = (1.28*edit_steps, 1.3*num_pcs), #(width, tall)
                        dpi = 500,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = save_path_name,
                        fill_images_columnwise=False,
                        max_images_limit=100,
                        subplot_title_text_size=10,
                        use_tight_padding=True,
                        use_col_row_names=True,
                        col_names=col_names,
                        row_names=row_names,
                        col_name_font_weight=None,
                        col_name_font_size=None,
                        row_name_font_weight=None,
                        row_name_font_size=None)

    return

  def define_return_ablation_study_images(self,
                                          chosen_cnum_tuning='1002',
                                          chosen_dataset_tuning='celebhq_dataset_full',
                                          chosen_cnum_blending='1002',
                                          chosen_dataset_blending='celebhq_dataset_full'):

    ablation_examples = {}

    ### With and without subspace tuning
    fp_index = np.argwhere([str(cnum)==chosen_cnum_tuning for cnum in cnums_all[chosen_dataset_tuning]])[0][0]
    assert type(fp_index) == np.int64
    image_path = cpfs_all[chosen_dataset_tuning][fp_index]
    assert type(image_path) == str

    input_image, _, base_latent = self.embed_input_image(image_path, 
                                                         show_plots=False)
    ablation_examples['subspace_tuning','original'] = np.array(input_image)
    _, ablation_examples['subspace_tuning','without'], _ = self.add_avg_glasses(
                                          input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=False,
                                           return_blended_image=True)
    _, ablation_examples['subspace_tuning','with'], _ = self.add_avg_glasses(
                                          input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True,
                                           return_blended_image=True)
    

    ### With and without blending
    fp_index = np.argwhere([str(cnum)==chosen_cnum_blending for cnum in cnums_all[chosen_dataset_blending]])[0][0]
    assert type(fp_index) == np.int64
    image_path = cpfs_all[chosen_dataset_blending][fp_index]
    assert type(image_path) == str

    input_image, result_image, base_latent = self.embed_input_image(
                                                              image_path, 
                                                              show_plots=False)
    ablation_examples['blending','original'] = np.array(input_image)
    _, ablation_examples['blending','without'], _ = self.add_avg_glasses(
                                          input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True,
                                           return_blended_image=False)
    _, ablation_examples['blending','with'], _ = self.add_avg_glasses(
                                          input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True,
                                           return_blended_image=True)
    
    for image in ablation_examples.values():
      assert image.shape == (1024,1024,3)

    return ablation_examples

  def define_return_limitations_study_images(self):

    descs = []
    image_ids = []

    ### Idenitify eyebrow failures
    image_ids.append(('celebhq_dataset_full','1030', 5, 6.7))
    image_ids.append(('siblingsdb','34', 3, 10))
    descs+=['Eyebrow Issues']*2

    ### Idenitify hair overlap failures
    image_ids.append(('celebhq_dataset_full','1005', 5, 6.7))
    image_ids.append(('celebhq_dataset_full','1031', 0, 0))
    image_ids.append(('siblingsdb','241', 5, 6.7))
    image_ids.append(('siblingsdb','309', 5, 6.7))
    descs+=['Hair Overlap']*4

    ### Idenitify light tint sunglasses
    image_ids.append(('siblingsdb','41', 4, -10))
    descs+=['Light Tint Glasses']

    ## Idenitify dissapearing glasses failures
    image_ids.append(('celebhq_dataset_full','1034', 4, 13.3))
    descs+=['Dissapearing Glasses']

    images = self.generate_images_of_specific_samples(image_ids)

    # plot_images_subplot(images=images, #list of images or image paths
    #               plots_tall_wide=None, #tuple with number of subplots tall and wide
    #               sub_titles=descs, 
    #               figsize = (6, 8), #(width, tall)
    #               dpi = 500,
    #               sharex = 'all',
    #               sharey = 'all',
    #               title='',
    #               remove_ticks=True,
    #               save_path_name = None,
    #               fill_images_columnwise=False,
    #               max_images_limit=100,
    #               subplot_title_text_size=5,
    #               use_tight_padding=True,
    #               use_col_row_names=False,
    #               col_names=None,
    #               row_names=None,
    #               col_name_font_weight=None,
    #               col_name_font_size=None,
    #               row_name_font_weight=None,
    #               row_name_font_size=None)

    return images, descs


  def create_sequential_edits_figure(self,
                                  image_path):

    per_edit_number_steps = 4

    input_image, result_image, base_latent = self.embed_input_image(
                                                              image_path, 
                                                              show_plots=False)

    start_latent, _, _ = self.add_avg_glasses(input_image=input_image, 
                                           base_latent=base_latent,
                                           base_bias=1,
                                           show_plots=False,
                                           auto_pick_bias=True)

    PC_nums = [2, 5, 0]
    PC_limits = [20, 25, 10]

    titles=[]
    pca_vector_multiplier_sets = []

    first_stage=True
    for PC_num, PC_limit in zip(PC_nums, PC_limits):
      pca_vector_multiplier_sets.append(np.linspace(0, PC_limit, per_edit_number_steps).tolist())
      _, _, lower_edit_desc, upper_edit_desc = self.edit_bounds(
                                                          pc_num=PC_num,
                                                          return_edit_desc=True)
      edit_desc = lower_edit_desc if PC_limit<0 else upper_edit_desc
      if first_stage:
        first_col_title = 'Beginning Image'
        first_stage=False
      else:
        first_col_title = 'From Previous Stage'
        
      

      titles = titles + [first_col_title] + [f'{"+"*(edit_multiplier+1)} {edit_desc}' for edit_multiplier in range(per_edit_number_steps-1)]

    ### Apply edits in the desired amount of stages
    PCX_edit_images, edit_desc = self.apply_edits_in_stages(
                            PC_nums=PC_nums, 
                            pca_vector_multiplier_sets=pca_vector_multiplier_sets, 
                            start_latent=start_latent)
    
    blends, seg_maps, blur_masks = self.blend_in_edits(edits=PCX_edit_images, 
                                                    input_image=input_image)
    
    # plot_images_subplot(images=blends, #list of images or image paths
    #                   plots_tall_wide=(len(PC_nums), per_edit_number_steps), #tuple with number of subplots tall and wide
    #                   sub_titles=titles, 
    #                   figsize = (1.66*per_edit_number_steps, 1.5*len(PC_nums)), # (width, tall)
    #                   dpi = 500,
    #                   sharex = 'all',
    #                   sharey = 'all',
    #                   title='',
    #                   remove_ticks=True,
    #                   save_path_name = None,
    #                   fill_images_columnwise=False,
    #                   subplot_title_text_size=5)

    plot_images_subplot(images=blends, #list of images or image paths
                        plots_tall_wide=(len(PC_nums), per_edit_number_steps), #tuple with number of subplots tall and wide
                        sub_titles=titles, 
                        figsize = (1.30*per_edit_number_steps, 1.5*len(PC_nums)), #(width, tall)
                        dpi = 500,
                        sharex = 'all',
                        sharey = 'all',
                        title='',
                        remove_ticks=True,
                        save_path_name = None,
                        fill_images_columnwise=False,
                        max_images_limit=100,
                        subplot_title_text_size=10,
                        use_tight_padding=True,
                        use_col_row_names=False,
                        col_names=None,
                        row_names=None,
                        col_name_font_weight=None,
                        col_name_font_size=None,
                        row_name_font_weight=None,
                        row_name_font_size=None)


    return blends




    # edit_descs = {0: ('Shrink','Enlarge'),
    #               1: ('Lower','Raise'),
    #               2: ('Square','Round'),
    #               3: ('Shrink+Round','Enlarge+Square'),
    #               4: ('Reverse Cateye','Cateye'),
    #               5: ('Thicken+Darken Frames','Thin+Lighten Frames')}


    # bounds = {0: (-5,10),
    #           1: (-10,4),
    #           2: (-4,20),
    #           3: (-5,15),
    #           4: (-15,20),
    #           5: (-10,10)}


