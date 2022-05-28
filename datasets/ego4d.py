import os
import json
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
import time
import ffmpeg
import numpy as np
import random
import pdb


class VideoModulatedSTGrounding(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        transforms,
        is_train=False,
        video_max_len=200,
        video_max_len_train=100,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        stride=0,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """
        self.vid_folder = vid_folder
        print("loading annotations into memory...")
        tic = time.time()
        annotations = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self._transforms = transforms
        self.is_train = is_train
        self.video_max_len = video_max_len
        self.video_max_len_train = video_max_len_train
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = (
            {}
        )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]
        self.stride = stride
        pos_idxs, neg_idxs = [], []
        for i, ann in enumerate(annotations):
            start_frame, end_frame = 0, 200
            video_id, clip_id, query_id, duration, timestamps = ann[:5]
            is_pos = ann[-1]
            pos_idxs.append(i) if is_pos == "positive" else neg_idxs.append(i)
            frame_ids = [fid for fid in range(start_frame, end_frame)]
            tube_start_frame = int(timestamps[0]/duration*200)
            tube_end_frame = int(timestamps[1]/duration*200)
    
            if tube_start_frame == tube_end_frame:
                if tube_end_frame>=frame_ids[-1]:
                    tube_start_frame = tube_start_frame-1     
                else:
                    tube_end_frame = tube_end_frame+1

            inter_frames = [frame_id for frame_id in frame_ids if tube_start_frame<=frame_id<=tube_end_frame]
            # frames in the annotated moment
            self.vid2imgids[video_id+"-%d"%query_id] = [frame_ids, inter_frames]
       
        _ratio = int(len(neg_idxs)/len(pos_idxs))
        train_ratio = 5
        neg_idxs = random.sample(neg_idxs, len(pos_idxs)*min(_ratio, train_ratio))
        sampled_idxs = pos_idxs #+neg_idxs
        self.annotations = [annotations[i] for i in sampled_idxs]
        
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video_id, clip_id, query_id, duration, timestamps, caption = self.annotations[idx][:6]
        hw, is_positive = self.annotations[idx][-2:]
        h, w = hw[0], hw[1]
        clip_start, clip_end = 0, 200
        frame_ids, inter_frames = self.vid2imgids[video_id+"-%d"%query_id]
        
        # ffmpeg decoding
        video_id = "00a192ce-4468-4700-9a1a-f44eff5e29cc"
        clip_id = 2
        vid_path = os.path.join(self.vid_folder, "%s_%d.mp4"%(video_id, clip_id))
        cmd = ffmpeg.input(vid_path)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) == len(frame_ids)
        
        # prepare frame-level targets
        targets_list = [] 
        inter_idx = list(inter_frames) # list of indexes of the frames in the annotated moment
        for img_id in frame_ids:
            target = dict()
            target["label"] = 0 if img_id not in inter_idx else 1
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list  # 3,200,H,W;  200
        raw_idx = inter_idx
        # temporal crop
        if self.tmp_crop:
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx]
                images = images[:, new_start_idx:new_end_idx+1]  # CTHW
                targets = [
                    x
                    for i, x in enumerate(targets)
                    if new_start_idx <= i <= new_end_idx
                ]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]
        #import pdb;pdb.set_trace()
        # video level annotations
        tmp_target = {
            "video_id": video_id,
            "inter_idx": [inter_idx[0], inter_idx[-1]]
                    if inter_idx else [-100, -100],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "caption": caption,
        }
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args):
    vid_dir = Path(args.mount_dir+"/"+args.ego4d_vid_path)
   
    if args.test:
        ann_file = Path(args.mount_dir+"/"+args.ego4d_ann_path) / f"test_clip_nlq.json"
    elif image_set == "val":
        ann_file = Path(args.mount_dir+"/"+args.ego4d_ann_path) / f"test_clip_nlq.json"
    else:
        ann_file = (
            Path(args.mount_dir+"/"+args.ego4d_ann_path) / f"train_clip_nlq.json"
            if args.video_max_len_train == 200 or (not args.sted)
            else Path(args.mount_dir+"/"+args.vidstg_ann_path) / f"train_{args.video_max_len_train}.json"
        )

    dataset = VideoModulatedSTGrounding(
        vid_dir,
        ann_file,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        video_max_len=args.video_max_len,
        video_max_len_train=args.video_max_len_train,
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        stride=args.stride,
    )
    return dataset
