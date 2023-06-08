import os
import dlib
import shutil
import torchvision.transforms as transforms
import torch
from argparse import Namespace
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import time
import numpy as np

### Module must be imported in encoder4editing directory
assert os.path.basename(os.getcwd()) == 'encoder4editing'

### Load needed modules from e4e encoder repo
import sys
sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im
from models.psp import pSp  
from scripts.align_faces_parallel import align_face

def make_dir_copy_file(src_filepath,
                       dst_folder,
                       dst_filename,
                       pass_existing_dir=True):
  if pass_existing_dir and os.path.isdir(dst_folder):
    print(f'Dir already exists ({dst_folder})')     
  else:
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src=src_filepath,
                dst=f'{dst_folder}/{dst_filename}')
  return


class e4e_encoder_inference:
  def __init__(self, 
               base_working_dir, 
               load_loc):
    
    self.base_working_dir = base_working_dir

    ### Initalization must happen in encoder4editing directory
    cwd = os.getcwd()
    os.chdir(f'{base_working_dir}/encoder4editing')
    
    ### Load needed files
    #src=f'{load_loc}/e4e_files/e4e_ffhq_encode.pt'
    #dst=f'{base_working_dir}/encoder4editing/pretrained_models/e4e_ffhq_encode.pt'
    make_dir_copy_file(src_filepath=f'{load_loc}/e4e_ffhq_encode.pt',
                       dst_folder=f'{base_working_dir}/encoder4editing/pretrained_models',
                       dst_filename = 'e4e_ffhq_encode.pt',
                       pass_existing_dir=False)
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
      print('Downloading files for aligning face image...')
      os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
      os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
      print('Done.')

    ### Initalize dlib predictor
    self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    ### Set params
    self.EXPERIMENT_ARGS = {
            "model_path": "pretrained_models/e4e_ffhq_encode.pt",
            "image_path": "notebooks/images/input_img.jpg",
            "transform":  transforms.Compose([
                          transforms.Resize((256, 256)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    self.resize_dims = (256, 256)

    model_path = self.EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    self.opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    self.opts['checkpoint_path'] = model_path
    self.opts= Namespace(**self.opts)
    self.net = pSp(self.opts)
    self.net.eval()
    self.net.cuda()
    print('Model successfully loaded!')

    os.chdir(cwd)


  def run_alignment(self, image_path):
    aligned_image = align_face(filepath=image_path, predictor=self.predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

  def run_test_1(self):
    cwd=os.getcwd()
    os.chdir(f'{self.base_working_dir}/encoder4editing')
    image_path = self.EXPERIMENT_ARGS["image_path"]
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    input_image = self.run_alignment(image_path)
    input_image.resize(self.resize_dims)
    plt.figure()
    plt.imshow(input_image)
    os.chdir(cwd)
    return

  def run_test_2(self, image_path):
    input_image, transformed_image = self.pre_process_image(image_path, run_allign=True)
    result_image, final_latent= self.run_inference(transformed_image)
    coupled_res, final_rec = self.visualize_results(result_image, transformed_image)

    plt.figure()
    plt.title('Original and StyleGAN2 Embedded Image')
    plt.imshow(coupled_res)
    
    _, embedded_image_pil = self.run_generator(final_latent)
    plt.figure()
    plt.title('Generator output from learned latent code')
    plt.imshow(embedded_image_pil)

    return input_image, transformed_image, result_image, final_latent, coupled_res, final_rec, embedded_image_pil

  def pre_process_image(self,
                        img_fp_or_array,
                        run_allign=True):

    input_is_filepath = type(img_fp_or_array) == str      

    if run_allign:
      if not input_is_filepath:
        raise TypeError('Image input must be str (filepath) when running allignment')
      else:
        input_image = self.run_alignment(img_fp_or_array)
      # input_image.resize((256, 256))
    else:
      if input_is_filepath:
        input_image = PIL.Image.open(img_fp_or_array)
      else:
        input_image = Image.fromarray(img_fp_or_array)


    img_transforms = self.EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    return input_image, transformed_image

  def run_inference(self, transformed_image):
    self.opts.resize_outputs = False  # generate outputs at full resolution
    with torch.no_grad():
      tic = time.time()

      codes = self.net.encoder(x=transformed_image.unsqueeze(0).to("cuda").float())
      codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
      latent_code_tensor = codes
      final_latent = latent_code_tensor.clone().detach().cpu().numpy()

      

      # images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
      # result_image, final_latent = images[0], latents[0]

      # result_batch, result_latents, results_deltas = run_inversion(transformed_image.unsqueeze(0).cuda(), 
      #                                                 net, 
      #                                                 opts,
      #                                                 return_intermediate_results=True)
      toc = time.time()
      print('Inference took {:.4f} seconds.'.format(toc - tic))
      # result_latents = result_latents[0]
      # final_latent = result_latents[-1]
      # results_deltas = results_deltas[0]
      # final_delta = results_deltas[-1]

    result_image, _ = self.run_generator(final_latent)

    return result_image, final_latent

  @staticmethod
  def get_coupled_results(result_batch, 
                          transformed_image, 
                          resize_amount):
      result_tensors = result_batch  # there's one image in our batch
      final_rec = tensor2im(result_tensors).resize(resize_amount)
      input_im = tensor2im(transformed_image).resize(resize_amount)
      coupled_res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
      coupled_res = Image.fromarray(coupled_res)
      return coupled_res, final_rec


  def visualize_results(self,
                        result_batch, 
                        transformed_image,
                        resize_outputs=True):
    resize_amount = (256, 256) if resize_outputs else (1024, 1024)
    coupled_res, final_rec = self.get_coupled_results(result_batch, 
                                                      transformed_image, 
                                                      resize_amount)
    return coupled_res, final_rec


  def run_generator(self,
                    latent_code):

    assert latent_code.shape == (1,18,512)
    assert latent_code.dtype == 'float32' or latent_code.dtype == torch.float32

    if type(latent_code) == np.ndarray:
      # print(f'converting latent code from type {type(latent_code)}')
      # print(latent_code.shape)
      latent_code = torch.from_numpy(latent_code).to("cuda")

    #assuming the weights delta are a list of numpy arrays
    # weights_delta = [torch.from_numpy(w).to("cuda") if w is not None else w for w in weights_delta]

    with torch.no_grad():
      images, _ = self.net.decoder([latent_code], 
                              input_is_latent = True,
                              randomize_noise = False,
                              return_latents = False)
    embedded_image = images[0]
    embedded_image_pil = tensor2im(embedded_image)
    return embedded_image, embedded_image_pil

  def run_on_batch(self, inputs):
    images, latents = self.net(inputs.to("cuda").float(), 
                          randomize_noise=False, 
                          return_latents=True)
    # latents = latents + net.latent_avg.repeat(latents.shape[0], 1, 1)

    return images, latents

  @staticmethod
  def display_alongside_source_image(result_image, source_image, resize_dims):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

  @staticmethod
  def to_cuda_float(x):
    return torch.tensor(x).to('cuda').float()

  def run_gen_add_pc_direction_bias(self,
                       start_latent, #Latent code to begin with
                       fitted_pca, #Fitted pca 
                       bias, # Amount to move in PC direction
                       PC_num, #PC direction number
                       run_gen=True): 

    # vec_to_add = norm_pc[PC_num, :, :] * bias

    try:
      n_components = fitted_pca.n_components_
    except:
      n_components = len(fitted_pca.eigenvalues_)
    one_hot = np.zeros((1,n_components)) #create one-hot-vector
    one_hot[0,PC_num] = bias #modulate magnitude of direction along PC vector
    vec_to_add = fitted_pca.inverse_transform(one_hot) #Determine vector in original space
    vec_to_add = vec_to_add.reshape(start_latent.squeeze().shape) #Reshape back to normal W+ shape

    new_vec = start_latent + vec_to_add
    
    if run_gen:
      img = np.array(self.run_generator( self.to_cuda_float(new_vec) )[1])
    else:
      img = None
    
    return img, new_vec
