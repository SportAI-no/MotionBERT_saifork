import os
import json
from typing import Tuple

import numpy as np
from tqdm import tqdm
from pathlib import Path

from mediautil import Vid

import typer

import torch
from lib.utils.tools import *
from lib.utils.learning import *
from torch.utils.data import DataLoader
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset

from anno import AnnotationInstance, AnnotationFrame, VideoEntry
from anno.formats import H36MFormat

app = typer.Typer()


def coco2h36m(x: np.ndarray, conf: np.ndarray):
    """
    Input: x (M x T x V x C)

    COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}

    H36M:
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    7: 'belly',
    8: 'neck',
    9: 'nose',
    10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
    """
    y = np.zeros(x.shape)
    y_con = np.zeros(conf.shape)
    y_con[0] = np.mean(conf[[11, 12]])
    y_con[1] = conf[12]
    y_con[2] = conf[14]
    y_con[3] = conf[16]
    y_con[4] = conf[11]
    y_con[5] = conf[13]
    y_con[6] = conf[15]
    y_con[8] = np.mean(conf[[5, 6]])
    y_con[7] = np.mean(y_con[[0, 8]])
    y_con[9] = conf[0]
    y_con[10] = np.mean(conf[[1, 2]])
    y_con[11] = conf[5]
    y_con[12] = conf[7]
    y_con[13] = conf[9]
    y_con[14] = conf[6]
    y_con[15] = conf[8]
    y_con[16] = conf[10]

    y[0, :] = (x[11, :] + x[12, :]) * 0.5
    y[1, :] = x[12, :]
    y[2, :] = x[14, :]
    y[3, :] = x[16, :]
    y[4, :] = x[11, :]
    y[5, :] = x[13, :]
    y[6, :] = x[15, :]
    y[8, :] = (x[5, :] + x[6, :]) * 0.5
    y[7, :] = (y[0, :] + y[8, :]) * 0.5
    y[9, :] = x[0, :]
    y[10, :] = (x[1, :] + x[2, :]) * 0.5
    y[11, :] = x[5, :]
    y[12, :] = x[7, :]
    y[13, :] = x[9, :]
    y[14, :] = x[6, :]
    y[15, :] = x[8, :]
    y[16, :] = x[10, :]
    return y, y_con


def get_h36m_json(entry: VideoEntry):
    h, w = Vid(entry._source_path).hw

    tracked_keypoints = {}

    for frame in entry:
        for instance in frame:
            instance: AnnotationInstance = instance
            keypoints = instance.get_keypoints()
            bbox = instance.get_bbox() * np.array([w, h, w, h])
            # scale keypoints to frame size
            keypoints[:, 0] *= w
            keypoints[:, 1] *= h
            confidences = instance.get_confidence()

            keypoints, confidences = coco2h36m(keypoints, confidences)

            tracked_keypoints.setdefault(instance.get_tracking_id(), []).append(
                {
                    "image_id": str(frame.frame_nr) + ".jpg",
                    "category_id": 1,
                    "keypoints": np.vstack([keypoints.T, confidences])
                    .T.flatten()
                    .tolist(),  # [x1,y1,c1,...,xk,yk,ck],
                    "score": instance.get_box_confidence(),
                    "box": bbox.tolist(),
                    "idx": [0.0],
                }
            )
    return tracked_keypoints


def run_2d_to_3d(config_path: str, weights_path: str,
                 json_path: str, size: Tuple[int, int] = None):
    focus = None
    clip_len = 243

    args = get_config(config_path)
    model_backbone = load_backbone(args)
    checkpoint = torch.load(weights_path, 
                            map_location=lambda storage, loc: storage, 
                            weights_only=False)

    for key in list(checkpoint["model_pos"].keys()):
        checkpoint["model_pos"][key.replace("module.", "")] = checkpoint[
            "model_pos"
        ].pop(key)

    model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)

    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model_backbone)
    #     model_backbone = model_backbone.cuda()

    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "drop_last": False,
    }

    if json_path is None: # just warmup
        return None

    if size is not None:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(
            json_path, clip_len=clip_len, vid_size=size, scale_range=None, focus=focus
        )
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(
            json_path, clip_len=clip_len, scale_range=[1, 1], focus=focus
        )

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
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    if size is not None:
        # Convert to pixel coordinates
        results_all = results_all * (min(size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(size) / 2.0

    return results_all


def motionbert_to_anno(kps: np.ndarray, tracking_id: int) -> VideoEntry:
    R = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    kps = np.einsum("ij,knj->kni", R, kps)

    anno_frames = []
    for i in range(kps.shape[0]):
        frame = AnnotationFrame(i)
        frame.add_instance(
            AnnotationInstance(kps[i], tracking_id=tracking_id, annotation_format=H36MFormat)
        )
        anno_frames.append(frame)

    return anno_frames


@app.command()
def run_entry(
    config_path: str,
    weights_path: str,
    entry_path: Path = None, 
    output_path: Path = None, 
    temp_root: Path = Path('~/tmp/')
):
    if entry_path is None:  #  just warmup for test
        run_2d_to_3d(config_path, weights_path, None)
        return
    
    entry_2d = VideoEntry.from_file(entry_path)
    h36m = get_h36m_json(entry_2d)

    # Copy entry with 2d-pose to new entry, keeping frame info with timestamps
    entry = entry_2d.copy(copy_frames=True, copy_instances=False)
    entry._annotation_format = H36MFormat
    
    # Old code to create new entry:
    # entry = VideoEntry(entry.get_source_path(), annotation_format=H36MFormat)
    # for i, _ in enumerate(entry.get_source()):
    #     entry.add_frame(AnnotationFrame(i))

    for k, v in h36m.items():
        json_output_path = Path(temp_root) / f"{entry_path.stem}_{k}_h36m.json"
        with json_output_path.open("w") as f:
            json.dump(v, f)

        result_numpy = run_2d_to_3d(config_path, weights_path, str(json_output_path))
        for frame in motionbert_to_anno(result_numpy, tracking_id=int(k)):
            if frame.frame_nr >= len(entry):
                continue
            for i in frame:
                entry[frame.frame_nr].add_instance(i)
        os.remove(str(json_output_path))

    entry.to_file(output_path)


if __name__ == "__main__":
    app()