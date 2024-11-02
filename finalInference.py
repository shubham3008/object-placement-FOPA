import os
import argparse
import torch
import numpy as np
from PIL import Image
from pprint import pprint
from torchvision import transforms
from tqdm import tqdm

from network.ObPlaNetND import *
from config import arg_config
from data.OBdatasetND import make_composite_PIL
from data.all_transforms import ComposeND, JointResizeND

# depth normal 

import imageio 
from skimage.transform import resize
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals
from boosted_depth.depth_util import create_depth_models, get_depth
from omnidata_tools.model_util import load_omni_model
from chrislib.general import (
    round_32,
)

# harmonization

from pathlib import Path

from chrislib.general import (
    invert, 
    uninvert, 
    view, 
    np_to_pil, 
    to2np, 
    add_chan, 
    show, 
    round_32,
    tile_imgs
)

from intrinsic.pipeline import run_gray_pipeline, load_models

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def compute_composite_normals(img, msk, model, size):
    
    bin_msk = (msk > 0)

    bb = get_bbox(bin_msk)
    bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

    # create the crop around the object in the image to send through normal net
    img_crop = img[bb[0] : bb[1], bb[2] : bb[3], :]

    crop_scale = 1024 / max(bb_h, bb_w)
    img_crop = rescale(img_crop, crop_scale)
        
    # get normals of cropped and scaled object and resize back to original bbox size
    nrm_crop = get_omni_normals(model, img_crop)
    nrm_crop = resize(nrm_crop, (bb_h, bb_w))

    h, w, c = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim
    else:
        scale = 1.0
    
    # resize to the final output size as specified by input args
    out_img = rescale(img, scale, r32=True)
    out_msk = rescale(msk, scale, r32=True)
    out_bin_msk = (out_msk > 0)
    
    # compute normals for the entire composite image at it's output size
    out_nrm_bg = get_omni_normals(model, out_img)
    
    # now the image is at a new size so the parameters of the object crop change.
    # in order to overlay the normals, we need to resize the crop to this new size
    out_bb = get_bbox(out_bin_msk)
    bb_h, bb_w = out_bb[1] - out_bb[0], out_bb[3] - out_bb[2]
    
    # now resize the normals of the crop to this size, and put them in empty image
    out_nrm_crop = resize(nrm_crop, (bb_h, bb_w))
    out_nrm_fg = np.zeros_like(out_img)
    out_nrm_fg[out_bb[0] : out_bb[1], out_bb[2] : out_bb[3], :] = out_nrm_crop

    # combine bg and fg normals with mask alphas
    # print(out_nrm_fg.shape, out_msk.shape, out_nrm_bg.shape)
    out_nrm = (out_nrm_fg * out_msk[:, :, None]) + (out_nrm_bg * (1.0 - out_msk[:, :, None]))
    return out_nrm


class Evaluator:
    def __init__(self, args, checkpoint_path, reproduce_paper):
        super(Evaluator, self).__init__()
        self.args = args
        self.dev = torch.device("cuda:0")
        self.to_pil = transforms.ToPILImage()

        self.checkpoint_path = checkpoint_path
        pprint(self.args)

        print('load pretrained weights from ', checkpoint_path)
        self.net = nn.DataParallel(ObPlaNet_resnet18(
            pretrained=False)).to(self.dev)
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.dev), strict=False)
        self.net = self.net.to(self.dev).eval()
        self.softmax = torch.nn.Softmax(dim=1)

        # dpt normal model
        print('loading depth model')
        self.dpt_model = create_depth_models()
        print('loading normals model')
        self.nrm_model = load_omni_model()

        print('loading intrinsic decomposition model')
        self.int_model = load_models('paper_weights')

        print('loading albedo model')
        self.alb_model = load_albedo_harmonizer()

        print('loading reshading model')
        if reproduce_paper:
            self.shd_model = load_reshading_model('paper_weights')
        else:
            self.shd_model = load_reshading_model('further_trained')


        # image transforms
        self.train_triple_transform = ComposeND([JointResizeND(256)])
        self.train_img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
            ]
        )
        self.train_mask_transform = transforms.ToTensor()
        
    def get_nrm_dpt(self, bg_path):
        bg_img = load_image(bg_path)
        bg_h, bg_w = bg_img.shape[:2]
        max_dim = max(bg_h, bg_w)

        nrm_scale = 512 / max_dim
        small_bg_img = rescale(bg_img, nrm_scale)
        small_bg_nrm = get_omni_normals(self.nrm_model, small_bg_img)
        bg_nrm = resize(small_bg_nrm, (bg_h, bg_w))

        if max_dim > 1024:
            dpt_scale = 1024 / max_dim
        else:
            dpt_scale = 1.0
        small_bg_img = rescale(bg_img, dpt_scale, r32=True)
        bg_depth = get_depth(small_bg_img, self.dpt_model)
        if dpt_scale != 1:
            bg_depth = resize(bg_depth, (bg_h, bg_w))
        return bg_nrm, bg_depth
        

    def get_heatmap_multi_scales(self, fg_scale_num, bg_path, fg_path, mask_path, example_dir):
        '''
        generate heatmap for each pair of scaled foreground and background  
        '''
        fg_scales = list(range(1, fg_scale_num+1))
        fg_scales = [i/(1+fg_scale_num+1) for i in fg_scales]

        bg_img = Image.open(bg_path)  
        bg_nrm, bg_dpt = self.get_nrm_dpt(bg_path)
        bg_nrm = Image.fromarray(np.uint8(bg_nrm*255))
        bg_dpt = Image.fromarray(np.uint8(bg_dpt*255))

        if len(bg_img.split()) != 3: 
            bg_img = bg_img.convert("RGB")

        if len(bg_nrm.split()) != 3: 
            bg_nrm = bg_nrm.convert("RGB")

        if len(bg_dpt.split()) == 3:
            bg_dpt = bg_dpt.convert("L")

        bg_img_aspect =  bg_img.height/bg_img.width
        fg_tocp = Image.open(fg_path).convert("RGB")
        mask_tocp = Image.open(mask_path).convert("RGB")
        fg_tocp_aspect = fg_tocp.height/fg_tocp.width

        for index, fg_scale in enumerate(fg_scales):
            if fg_tocp_aspect>bg_img_aspect:
                new_height = bg_img.height*fg_scale
                new_width = new_height/fg_tocp.height*fg_tocp.width
            else:
                new_width = bg_img.width*fg_scale
                new_height = new_width/fg_tocp.width*fg_tocp.height

            new_height = int(new_height)
            new_width = int(new_width) 

            top = int((bg_img.height-new_height)/2)
            bottom = top+new_height
            left = int((bg_img.width-new_width)/2)
            right = left+new_width

            fg_img_ = fg_tocp.resize((new_width, new_height))
            mask_ = mask_tocp.resize((new_width, new_height))

            save_name = f'{example_dir}/heatmap' + '_' + str(fg_scale) + '.jpg'

            fg_img_ = np.array(fg_img_)       
            mask_ = np.array(mask_)
            fg_img = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            mask  = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            fg_img[top:bottom, left:right, :] = fg_img_
            mask[top:bottom, left:right, :] = mask_
            
            # print(bg_img, fg_img, mask, bg_dpt, bg_nrm)
            fg_img = Image.fromarray(np.uint8(fg_img*255))
            mask = Image.fromarray(np.uint8(mask*255))

            if len(fg_img.split()) == 3:
                fg_img = fg_img.convert("RGB")
            if len(mask.split()) == 3:
                mask = mask.convert("L")

            bg_t, bg_nrm_t, bg_dpt_t, fg_t, mask_t = self.train_triple_transform(bg_img, bg_nrm, bg_dpt, fg_img, mask)
            mask_t = self.train_mask_transform(mask_t)[None,:,:,:]
            fg_t = self.train_img_transform(fg_t)[None,:,:,:]
            bg_t = self.train_img_transform(bg_t)[None,:,:,:]
            bg_nrm_t = self.train_mask_transform(bg_nrm_t)[None,:,:,:]
            bg_dpt_t = self.train_mask_transform(bg_dpt_t)[None,:,:,:]
            # print(bg_t.shape, fg_t.shape, mask_t.shape, bg_dpt_t.shape, bg_nrm_t.shape)

            test_outs, _ = self.net(bg_t, bg_nrm_t, bg_dpt_t, fg_t, mask_t, 'test')
            test_outs = self.softmax(test_outs)
    
            test_outs = test_outs[0,1,:,:] 
            test_outs = transforms.ToPILImage()(test_outs)
            test_outs.save(save_name)

        
    def generate_composite_multi_scales(self, fg_scale_num, composite_num, bg_path, fg_path, mask_path, example_dir):
        '''
        generate composite images for each pair of scaled foreground and background 
        '''
        
        fg_scales = list(range(1, fg_scale_num+1))
        fg_scales = [i/(1+fg_scale_num+1) for i in fg_scales]

        icount = 0

        bg_img = Image.open(bg_path)  
        if len(bg_img.split()) != 3:  
            bg_img = bg_img.convert("RGB")

        # fg_tocp = Image.open(fg_path_2).convert("RGB")
        # mask_tocp = Image.open(mask_path_2).convert("RGB")

        bg_img_aspect =  bg_img.height/bg_img.width
        fg_tocp = Image.open(fg_path).convert("RGB")
        mask_tocp = Image.open(mask_path).convert("RGB")
        fg_tocp_aspect = fg_tocp.height/fg_tocp.width
        heatmap_center_list = []
        fg_size_list = []

        for index, fg_scale in enumerate(fg_scales):
            if fg_tocp_aspect>bg_img_aspect:
                new_height = bg_img.height*fg_scale
                new_width = new_height/fg_tocp.height*fg_tocp.width
            else:
                new_width = bg_img.width*fg_scale
                new_height = new_width/fg_tocp.width*fg_tocp.height

            h = int(new_height)
            w = int(new_width) 

            top = int((bg_img.height-h)/2)
            bottom = top+h
            left = int((bg_img.width-w)/2)
            right = left+w

            fg_img_ = fg_tocp.resize((w, h))
            mask_ = mask_tocp.resize((w, h))

            save_name = f'{example_dir}/heatmap' + '_' + str(fg_scale) + '.jpg'

            fg_img_ = np.array(fg_img_)       
            mask_ = np.array(mask_)
            fg_img = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            mask  = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            fg_img[top:bottom, left:right, :] = fg_img_
            mask[top:bottom, left:right, :] = mask_
            fg_img = Image.fromarray(np.uint8(fg_img*255))
            mask = Image.fromarray(np.uint8(mask*255))

            if len(fg_img.split()) == 3:
                fg_img = fg_img.convert("RGB")
            if len(mask.split()) == 3:
                mask = mask.convert("L")
            
            icount += 1
            heatmap = Image.open(os.path.join(save_name))
            os.remove(save_name)
            heatmap = np.array(heatmap)
            # exclude boundary

            heatmap_center = np.zeros_like(heatmap, dtype=np.float64)
            hb= int(h/bg_img.height*heatmap.shape[0]/2)
            wb = int(w/bg_img.width*heatmap.shape[1]/2)
            heatmap_center[hb:-hb, wb:-wb] = heatmap[hb:-hb, wb:-wb]
            heatmap_center_list.append(heatmap_center)
            fg_size_list.append((h,w))
            
            if icount==fg_scale_num:
                icount = 0
                heatmap_center_stack = np.stack(heatmap_center_list)
                # print(heatmap_center_stack.shape)
                # sort pixels in a descending order based on the heatmap 
                sorted_indices = np.argsort(-heatmap_center_stack, axis=None)
                sorted_indices = np.unravel_index(sorted_indices, heatmap_center_stack.shape)
                for i in range(composite_num):
                    iscale, y_, x_ = sorted_indices[0][i], sorted_indices[1][i], sorted_indices[2][i]
                    curr_h, curr_w = fg_size_list[iscale]
                    x_ = x_/heatmap.shape[1]*bg_img.width
                    y_ = y_/heatmap.shape[0]*bg_img.height
                    x = int(x_ - curr_w / 2)
                    y = int(y_ - curr_h / 2)
                    # make composite image with foreground, background, and placement 
                    composite_img, composite_msk = make_composite_PIL(fg_tocp, mask_tocp, bg_img, [x, y, curr_w, curr_h], return_mask=True)
                    save_img_path = os.path.join(f'{example_dir}/composite.png')
                    save_msk_path = os.path.join(f'{example_dir}/mask.png')
                    save_bg_path = os.path.join(f'{example_dir}/bg.jpeg')
                    # save_img_path = os.path.join(f'test_composite_{x}_{y}_{curr_w}_{curr_h}.png')
                    # save_msk_path = os.path.join(f'test_mask_{x}_{y}_{curr_w}_{curr_h}.png')
                    composite_img.save(save_img_path)
                    composite_msk.save(save_msk_path)
                    bg_img.save(save_bg_path)
                    # print(save_img_path)

    def harmonize_image(self, example_dir, args):
        bg_img = load_image(f'{example_dir}/bg.jpeg')
        comp_img = load_image(f'{example_dir}/composite.png')
        mask_img = load_image(f'{example_dir}/mask.png')

        if len(mask_img.shape) != 2:
            mask_img = mask_img[:,:,0]

        scene_name = Path(example_dir).stem

        # to ensure that normals are globally accurate we compute them at
        # a resolution of 512 pixels, so resize our shading and image to compute 
        # rescaled normals, then run the lighting model optimization
        bg_h, bg_w = bg_img.shape[:2]
        max_dim = max(bg_h, bg_w)
        scale = 512 / max_dim
        
        small_bg_img = rescale(bg_img, scale)
        small_bg_nrm = get_omni_normals(self.nrm_model, small_bg_img)
        
        result = run_gray_pipeline(
            self.int_model,
            small_bg_img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )
        
        small_bg_shd = result['gry_shd'][:, :, None]
        
        
        coeffs, lgt_vis = get_light_coeffs(
            small_bg_shd[:, :, 0], 
            small_bg_nrm, 
            small_bg_img
        )

        # now we compute the normals of the entire composite image, we have some logic
        # to generate a detailed estimation of the foreground object by cropping and 
        # resizing, we then overlay that onto the normals of the whole scene
        comp_nrm = compute_composite_normals(comp_img, mask_img, self.nrm_model, args.inference_size)

        # now compute depth and intrinsics at a specific resolution for the composite image
        # if the image is already smaller than the specified resolution, leave it
        h, w, c = comp_img.shape
        
        max_dim = max(h, w)
        if max_dim > args.inference_size:
            scale = args.inference_size / max_dim
        else:
            scale = 1.0
        
        # resize to specified size and round to 32 for network inference
        img = rescale(comp_img, scale, r32=True)
        msk = rescale(mask_img, scale, r32=True)
        
        depth = get_depth(img, self.dpt_model)
        
        result = run_gray_pipeline(
            self.int_model,
            img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )
        
        inv_shd = result['gry_shd']
        # inv_shd = rescale(inv_shd, scale, r32=True)

        # compute the harmonized albedo, and the subsequent color harmonized image
        alb_harm = harmonize_albedo(img, msk, inv_shd, self.alb_model, reproduce_paper=args.reproduce_paper) ** 2.2
        harm_img = alb_harm * uninvert(inv_shd)[:, :, None]

        # run the reshading model using the various composited components,
        # and our lighting coefficients computed from the background
        comp_result = compute_reshading(
            harm_img,
            msk,
            inv_shd,
            depth,
            comp_nrm,
            alb_harm,
            coeffs,
            self.shd_model
        )
        
        if args.intermediate:
            tile_imgs([
                img, 
                msk, 
                view(alb_harm),
                1-inv_shd, 
                depth, 
                comp_nrm, 
                view(generate_shd(comp_nrm, coeffs, msk, viz=True)[1]),
                1-invert(comp_result['reshading'])
            ], save=f'{args.output_dir}/{scene_name}_intermediate.jpeg', rescale=0.75)

        np_to_pil(comp_result['composite']).save(f'{example_dir}/harmonized.png')

        # print(f'finished ({i+1}/{len(examples)}) - {scene_name}')




if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', type=str, default= "test_bg.jpeg")
    parser.add_argument('--fg', type=str, default= "test_fg.jpeg")
    parser.add_argument('--mask', type=str, default= "test_mask.jpeg")
    parser.add_argument('--inference_size', type=int, default=1024, help='size to perform inference (default 1024)')
    parser.add_argument('--intermediate', action='store_true', help='whether or not to save visualization of intermediate representations')
    parser.add_argument('--reproduce_paper', action='store_true', help='whether or not to use code and weights from the original paper implementation')

    args = parser.parse_args()
    
    fg_scale_num = 16
    composite_num = 1

    full_path = 'nrm_dpt_weight.pth'
    # full_path = os.path.join('output', args.path, 'pth', f'{args.epoch}_state_final.pth')
    bg_path = args.bg
    fg_path = args.fg
    mask_path = args.mask
    output_dir= 'result'
#     full_path = args.path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(full_path):
        print(f'{full_path} does not exist!')
    else:
        evaluator = Evaluator(arg_config, checkpoint_path=full_path, reproduce_paper=args.reproduce_paper) 
        # if args.mode== "heatmap":
        evaluator.get_heatmap_multi_scales(fg_scale_num, bg_path, fg_path, mask_path, output_dir)
        # elif args.mode== "composite":
        evaluator.generate_composite_multi_scales(fg_scale_num, composite_num, bg_path, fg_path, mask_path, output_dir) 
        evaluator.harmonize_image(output_dir, args)