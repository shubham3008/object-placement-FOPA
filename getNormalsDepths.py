import os, imageio 
import numpy as np
from skimage.transform import resize
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals
from boosted_depth.depth_util import create_depth_models, get_depth
from omnidata_tools.model_util import load_omni_model
# from PIL import Image
import matplotlib.pyplot as plt
from chrislib.general import (
    round_32,
)

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

print('loading depth model')
dpt_model = create_depth_models()

print('loading normals model')
nrm_model = load_omni_model()

bg_dir = '../FOPA-Fast-Object-Placement-Assessment/data/data/bg/'
bg_list = os.listdir(bg_dir)
print('number of backgrounds: ',len(bg_list))
depth_dir = '../FOPA-Fast-Object-Placement-Assessment/data/data/bg_depth/'
normal_dir = '../FOPA-Fast-Object-Placement-Assessment/data/data/bg_normals/'

if not os.path.exists(depth_dir):
    os.makedirs(depth_dir)

if not os.path.exists(normal_dir):
    os.makedirs(normal_dir)

for curr_bg in bg_list:
    bg_img = load_image(bg_dir+curr_bg)
    bg_h, bg_w = bg_img.shape[:2]
    max_dim = max(bg_h, bg_w)

    nrm_scale = 512 / max_dim
    small_bg_img = rescale(bg_img, nrm_scale)
    small_bg_nrm = get_omni_normals(nrm_model, small_bg_img)
    bg_nrm = resize(small_bg_nrm, (bg_h, bg_w))
    
    if max_dim > 1024:
        dpt_scale = 1024 / max_dim
    else:
        dpt_scale = 1.0
    small_bg_img = rescale(bg_img, dpt_scale, r32=True)
    bg_depth = get_depth(small_bg_img, dpt_model)
    if dpt_scale != 1:
        bg_depth = resize(bg_depth, (bg_h, bg_w))

    # depth_img = Image.fromarray(255*bg_depth.astype(np.uint8))
    # normal_img = Image.fromarray(255*bg_nrm.astype(np.uint8))
    # depth_img.save(depth_dir+curr_bg)
    # normal_img.save(normal_dir+curr_bg)
    # print(bg_img.shape, nrm_scale, small_bg_img.shape, dpt_scale, small_bg_img.shape, depth.shape)

    plt.imsave(depth_dir+curr_bg,bg_depth, cmap='gray')
    plt.imsave(normal_dir+curr_bg,bg_nrm)
    # plt.imshow(bg_img)
    # plt.show()
    # plt.imshow(bg_nrm)
    # plt.show()
    # plt.imshow(depth)
    # plt.show()
    # exit()