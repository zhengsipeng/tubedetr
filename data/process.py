import json
import os
from time import time
import decord
import sys
import numpy as np
from torch.multiprocessing import Pool
from tqdm import tqdm
from decord import VideoReader, cpu
#decord.bridge.set_bridge("torch")
from torchvision.io import write_video


def slice_clip(start, end, num_works=5, root_dir=""):
    videos = []
    dura_dict = dict()
    for split in ["train", "test"]:
        with open(root_dir+"/tsg/ego4d/%s_data_gcn.json"%split) as f:
            anns = json.load(f)
        videos += [ann[0] for ann in anns]
        for ann in anns:
            if ann[0] not in dura_dict:
                dura_dict[ann[0]] = ann[1] 
    videos = list(set(videos))
    saved_dir = root_dir+"/tsg/ego4d/nlq_clip_videos"
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    print(len(videos))
    videos = videos[start:end]
    
    pool = Pool(num_works)
    data_dict_list = pool.map(slice_single_video,  [(i, video_id, dura_dict[video_id], saved_dir) for i, video_id in enumerate(videos)])
    pool.close()
    pool.join()


def slice_single_video(params):
        idx, video_id, duration, saved_dir = params
        print(idx)
        vr = VideoReader(root_dir+"/tsg/ego4d/nlq_raw_videos/%s.mp4"%video_id, ctx=cpu(0))
        num_frame = len(vr)
        fps = num_frame/duration
        window, step = 100, 50
        clip_id = 0
        frame_window = window*fps
        frame_step = step*fps

        while(True):        
            start_fid = int(clip_id*frame_step)
            end_fid = min(int(start_fid+frame_window), num_frame)
            if int(start_fid+frame_window)-frame_step>num_frame:
                break
            vr = VideoReader("../../data/tsg/ego4d/nlq_raw_videos/%s.mp4"%video_id, ctx=cpu(0))
            frame_idx = np.linspace(start_fid, end_fid-1, num=200).astype(int)
            ratio = (end_fid-start_fid)/200.
            img_arrays = vr.get_batch(frame_idx)  # 512,384,384,3
            #import pdb;pdb.set_trace()
            saved_path = os.path.join(saved_dir, "{}_{}.mp4".format(video_id, clip_id))
            #print(saved_path)
            write_video(saved_path, img_arrays.asnumpy(), fps/ratio)
            clip_id += 1
      
def slice_annotation(split):
    with open("ego4d/new_%s_nlq.json"%split) as f:
        video_anns = json.load(f)

    num_vid = 3
    clip_anns = []
    query_id = 0
    for vid_ann in tqdm(video_anns):
        num_vid+=1
        video_id, duration, timestamps, sentence, _split = vid_ann[:5]
        vr = VideoReader(r"../../data/tsg/ego4d/nlq_raw_videos/%s.mp4"%video_id, ctx=cpu(0))[0]
        video_h, video_w, _ = vr.shape
        text_emb = vid_ann[-1]
        
        window, step = 100, 50
        clip_id = 0
        while(True):
            start_sec = clip_id*step
            end_sec = min(start_sec+window, duration)
            if start_sec+window-step>duration:
                break
            #print(start_sec, end_sec, duration)
            new_duration = end_sec-start_sec
            
            if timestamps[0]==timestamps[1]:
                if timestamps[1]<start_sec or timestamps[0]>end_sec:
                    new_timestamps = [-1,-1]
                else:
                    new_timestamps = [timestamps[0]-start_sec, timestamps[1]-start_sec]
            else:
                if timestamps[1]<=start_sec or timestamps[0]>=end_sec:
                    new_timestamps = [-1,-1]
                else:
                    new_start = max(start_sec, timestamps[0]) - start_sec
                    new_end = min(end_sec, timestamps[1]) - start_sec
                    new_timestamps = [new_start, new_end]
            is_pos = "positive" if new_timestamps[0]!=-1 else "negative"
            clip_ann = [video_id, clip_id, query_id, new_duration, new_timestamps, sentence, _split, text_emb, [video_h, video_w], is_pos]
            clip_anns.append(clip_ann)
            query_id+=1
            clip_id+=1
        
        #import pdb;pdb.set_trace()
    print("../../data/tsg/ego4d/%s_clip_nlq.json"%split)
    with open("../../data/tsg/ego4d/%s_clip_nlq.json"%split, "w") as f:
        json.dump(clip_anns, f)


if __name__ == "__main__":
    #start, end, num_works = sys.argv[1:4]
    #root_dir = sys.argv[4]  # ../../data/
    #slice_clip(int(start), int(end), int(num_works), root_dir)
    slice_annotation("train")
    slice_annotation("test")