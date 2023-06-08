import numpy as np
import imageio
import os
from tqdm import tqdm
from glob import glob
import shutil
from skimage import transform

def copy_file(source_path, 
              dst_folder, 
              dst_filename,
              overwrite=False):
  
  if overwrite and os.path.exists(f'{dst_folder}/{dst_filename}'):
    os.remove(f'{dst_folder}/{dst_filename}')

  if os.path.exists(f'{dst_folder}/{dst_filename}'):
    print(f'File already exists ({dst_folder}/{dst_filename})')
  else:
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src=source_path,
                dst=f'{dst_folder}/{dst_filename}')
  return

def write_to_text_file(lines,
                       destination_folder,
                       filename):
  ''' Example use
  write_to_text_file(lines=["This is Delhi \n",
                            "This is Paris \n",
                            "This is London \n"],
                    destination_folder='/content',
                    filename='test.txt')
  '''
  file1 = open(f"{destination_folder}/{filename}","w")
  file1.writelines(lines)
  file1.close()
  return

class deeplab_segmenter_inference:
  def __init__(self, 
               temp_save_folder, #full path to temporary save folder
               resume_model_ckpt,
               deeplab_script_location,
               chosen_deeplab_epoch,
			   auto_clean=False):
    
    self.temp_save_folder = temp_save_folder
    self.deeplab_script_location = deeplab_script_location
    self.chosen_deeplab_epoch = chosen_deeplab_epoch

    #make temp folder
    new_dir = temp_save_folder
    if os.path.isdir(new_dir):
      if auto_clean:
        self.clean_up_temp_folder()
      else:
        if input('A clean folder is required to proceed. Remove old temporary files? (y/n): ') == 'y':
          self.clean_up_temp_folder()
        else:
          raise Exception('Clean folder required')
    os.makedirs(f'{temp_save_folder}/input_images', 
                exist_ok=False)
    
    #Copy model folder to temp location
    print('\nCopying model to temporary working directory')
    copy_file(source_path=resume_model_ckpt,
              dst_folder=f'{temp_save_folder}/model',
              dst_filename=os.path.basename(resume_model_ckpt))
    
    #Remove files that are already in output directory
    extra_files = glob(f'{self.temp_save_folder}/model/validation/*.npy')
    if len(extra_files) > 0:
      for extra_file in extra_files:
        os.remove(extra_file)
    
    # #Copy in custom test_deeplab_cross_validation.py file
    # copy_file(source_path=test_deeplab_cross_validation_script_fp, 
    #           dst_folder=f'{base_working_dir}/datasetGAN_release/datasetGAN', 
    #           dst_filename='test_deeplab_cross_validation.py',
    #           overwrite=True)
    
    #Create params file in temp location
    lines = ['{ \n',
    '"exp_dir": "model_dir/face_34", \n',
    '"batch_size": 64, \n',
    '"category": "face", \n',
    '"debug": false, \n',
    '"dim": [512, 512, 5088], \n',
    '"deeplab_res": 512, \n',
    '"number_class": 4, \n',
    '"testing_data_number_class": 4, \n',
    '"max_training": 7, \n',
    '"stylegan_ver": "1", \n',
    '"annotation_data_from_w": false, \n',
    '"annotation_mask_path":  "./custom_data/annotation/training_data", \n',
    f'"testing_path": "{temp_save_folder}/input_images", \n',
    '"average_latent": "./custom_data/training_latent/avg_latent_stylegan1.npy", \n',
    '"annotation_image_latent_path": "./custom_data/training_latent/latent_stylegan1.npy", \n',
    '"stylegan_checkpoint": "./checkpoints/stylegan_pretrain/karras2019stylegan-ffhq-1024x1024_old_serialization.pt", \n',
    '"model_num": 10, \n',
    '"upsample_mode":"bilinear" \n',
    '}']
    write_to_text_file(lines=lines,
                      destination_folder=self.temp_save_folder,
                      filename='face_34.json')
    
  def run_segmentation(self, faces):

    ## Assume input is cropped face

    ## Seg output key
    ## 0:background
    ## 1:frames
    ## 2:right lens
    ## 3:left lens

    #check to make sure that no leftover face images are in the target folders
    if len(glob(f'{self.temp_save_folder}/input_images/*')) > 0:
      raise Exception('Input temp directory not empty')
    if len(glob(f'{self.temp_save_folder}/model/validation/*.npy')) > 0:
      raise Exception('Intermediate seg map directory not empty')



    #save image and fake groundtruth maps to temporary file location
    print('\nSaving image and fake groundtruth maps to temporary file location')
    fp_ims = []
    dummy_fp_masks = []
    for i, face in tqdm(enumerate(faces)):
      assert face.dtype == np.uint8

      if face.shape[0] == 3:
        print('Reording input image axis...')
        face = np.moveaxis(face, (0,1,2), (2,0,1))

      if face.shape[0] == 4:
        print('Reording input image axis...')
        face = np.moveaxis(face, (0,1,2), (2,0,1)) 

      if face.shape[-1] == 4:
        print('Removeing input image alpha channel')
        face = face[:,:,:3]

      assert face.shape[-1] == 3
      assert len(face.shape) == 3 
      assert face.shape[0] == face.shape[1]

      fp_im = f'{self.temp_save_folder}/input_images/temp_face_{i}.png'
      dummy_fp_mask = f'{self.temp_save_folder}/input_images/mask_placeholder_{i}.npy'
      # print(f'Saving image')
      # print(f'{face.shape}')
      # print(f'{face.dtype}')
      if face.shape != (512,512,3):
        print('Resizing image to (512,512,3)')
        face = face.astype(float) / 255
        face = transform.resize(image=face, 
                                output_shape=(512,512,3),
                                preserve_range=False)
        face = (face*255).astype(np.uint8)
      # print(f'{face.shape}')
      # print(f'{face.dtype}')
      imageio.imwrite(fp_im, face)
      # print('Saving dummy mask')
      np.save(dummy_fp_mask, np.zeros((10,10)))
      fp_ims.append(fp_im)
      dummy_fp_masks.append(dummy_fp_mask)

    #change to script directory
    cwd = os.getcwd()
    os.chdir(self.deeplab_script_location)

    #run deeplab
    print('\nRunning deeplab')
    cmd = (f'python test_deeplab_cross_validation.py '
          f'--exp {self.temp_save_folder}/face_34.json '
          f'--resume {self.temp_save_folder}/model '
          f'--validation_number 0 '
          f'--chosen_deeplab_epoch {self.chosen_deeplab_epoch}' )
    # !{cmd}
    out=os.popen(cmd).read()
    print(out)

    os.chdir(cwd)

    #load the processed images
    print('\nLoading the processed images')
    seg_map_files = glob(f'{self.temp_save_folder}/model/validation/*.npy')
    sort_func = lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[2])
    # print(seg_map_files)
    seg_map_files = sorted(seg_map_files, 
                           key=sort_func)
    # print(seg_map_files)
    seg_maps = [np.load(seg_map_file) for seg_map_file in seg_map_files]

    #clean up temp images
    for fp_im, dummy_fp_mask, seg_map_file in zip(fp_ims, dummy_fp_masks, seg_map_files):
      os.remove(fp_im)
      os.remove(dummy_fp_mask)
      os.remove(seg_map_file)

    return seg_maps

  def test_segmenter(self):
    faces = [np.zeros((1000,1000,3),dtype=np.uint8), 
             np.zeros((1000,1000,3),dtype=np.uint8)]
    seg_maps = self.run_segmentation(faces=faces)
    print(f'\n{len(seg_maps)} seg maps of size {seg_maps[0].shape}')
    print('Success')
    return


  def clean_up_temp_folder(self):
    print('\nRemoving temporary folder')
    shutil.rmtree(self.temp_save_folder)
    return

  