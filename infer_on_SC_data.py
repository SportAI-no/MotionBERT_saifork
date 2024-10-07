from typing import Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset


def run_2d_to_3d(json_path: str, size: Tuple[int, int] = None): 

    config_path = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
    evaluate_bin_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
    # json_path = '/home/martin/repos/sportai/sc_project/motionbert_input/P001_L_inside_accur_LR_view2_031_h36m.json'
    # vid_path = '/home/martin/repos/sportai/sc_project/Resultater/GoPro/P001_L_inside_accur_LR_view2_031.MP4'
    focus = None
    clip_len = 243

    args = get_config(config_path)
    model_backbone = load_backbone(args)
    checkpoint = torch.load(evaluate_bin_path, map_location=lambda storage, loc: storage)

    for key in list(checkpoint['model_pos'].keys()):
        checkpoint['model_pos'][key.replace('module.', '')] = checkpoint['model_pos'].pop(key)

    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model_backbone)
    #     model_backbone = model_backbone.cuda()

    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }


    if size is not None:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(json_path, clip_len=clip_len, vid_size=size, scale_range=None, focus=focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(json_path, clip_len=clip_len, scale_range=[1,1], focus=focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    if size is not None:
        # Convert to pixel coordinates
        results_all = results_all * (min(size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(size) / 2.0

    return results_all

if __name__ == '__main__': 
    json_path = '/home/martin/repos/sportai/sc_project/motionbert_input/P001_L_inside_accur_LR_view2_031_h36m.json'

    json_paths = list(Path('/home/martin/repos/sportai/sc_project/motionbert_input').rglob('*.json'))
    output_root = Path('/home/martin/repos/sportai/sc_project/Resultater/mmpose_motionbert_3d')
    for json_path in tqdm(json_paths): 
        output_path = output_root / f'{json_path.stem}.npy'

        np.save(str(output_path), run_2d_to_3d(str(json_path), (1920, 1080)))
        