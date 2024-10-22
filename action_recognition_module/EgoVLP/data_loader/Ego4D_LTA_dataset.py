import os
import pdb
import sys
import json
import pandas as pd

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

from tqdm import tqdm

class LongTermAnticipation(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'fho_lta_train.json',
            'val': 'fho_lta_val.json',            # there is no test
            'test': 'fho_lta_test_unannotated.json'
        }
        assert self.subsample in ['video', 'text']

        self.metadata = pd.DataFrame(columns=['video_uid', 'clip_uid',
                                              'video_start_sec', 'video_end_sec',
                                              'query'])

        target_split_fp = split_files[self.split]
        ann_file = os.path.join(self.meta_dir, target_split_fp)
        with open(ann_file) as f:
            anno_json = json.load(f)

        # forward clip features
        if self.subsample == 'video':
            for anno_clip in anno_json["clips"]:
                if anno_clip["clip_uid"] == "440656ae-cb82-464e-b320-25c8e693ad84":
                    continue 
                clip_times = float(anno_clip["action_clip_start_sec"]), float(anno_clip["action_clip_end_sec"])
                clip_duration = clip_times[1] - clip_times[0]
                new = pd.DataFrame({
                    'video_uid': anno_clip['clip_uid'],
                    'clip_uid': anno_clip['clip_uid'] + "_" + str(anno_clip["action_idx"]),
                    'video_start_sec': clip_times[0],
                    'video_end_sec': clip_times[1]}, index=[1])
                self.metadata = self.metadata.append(new, ignore_index=True)

        elif self.subsample == 'text':            
            taxonomy_file = os.path.join(self.meta_dir, 'fho_lta_taxonomy.json')
            with open(taxonomy_file) as f:
                taxonomy_json = json.load(f) 
            for i, verb in tqdm(enumerate(taxonomy_json['verbs'])):  
                for j, noun in enumerate(taxonomy_json['nouns']): 
                    prompt = f'#C C {verb} {noun}.' 
                    new = pd.DataFrame({
                        'video_uid': f'{i}_{j}',
                        'clip_uid': 1,
                        'video_start_sec': 1,
                        'video_end_sec': 1,
                        'query': prompt,}, index=[1])
                    self.metadata = self.metadata.append(new, ignore_index=True)
        

        self.transforms = init_video_transform_dict()['test']

    def _get_video_path(self, sample):
        rel_video_fp = sample[0]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp + '.mp4')
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        caption = sample['query']
        return caption

    def _get_video_feats(self, item):
        sample = self.metadata.iloc [item]
        video_fp, rel_fp = self._get_video_path(sample)
        frame_sample = 'rand'
        if self.split == 'test':
            frame_sample = 'uniform'

        fps = 1.87
        try:
            imgs, idxs = self.video_reader(video_fp, sample[2]*30, sample[3]*30,
                                               (sample[3]-sample[2]) * fps * self.video_params['num_frames'], frame_sample)
        except Exception as e:
            print(e)
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'data': video_fp}
        data = {'video': imgs, 'meta' : meta_arr}
        return data

    def _get_text_feats(self, item):
        sample = self.metadata.iloc [item]
        text = self._get_caption(sample)
        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'dataset': self.dataset_name}
        data = {'text': text, 'meta' : meta_arr}
        return data

    def __getitem__(self, item):
        if self.subsample == 'video':
            return self._get_video_feats(item)
        if self.subsample == 'text':
            return self.get_noun_verb_from_ego4d(item)

    def select_noun_verb(self, item):
        sample = self.total_text_info[item]
        clip_uid = sample[0]
        prompts = sample[1]
        action_idx = sample[2]

        data = {'clip_uid': clip_uid, 'prompts' : prompts, 'action_idx' : action_idx }
        return data

    def get_noun_verb_from_ego4d(self, item):
        sample = self.metadata.iloc[item]
        data = {'text': sample[-1], 'meta' : sample[0] }
        return data

if __name__ == "__main__":
    split = 'val'
    kwargs = dict(
        dataset_name="Ego4d_LTA",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="dataset/ego4d_256/data",
        meta_dir="dataset/ego4d/benchmark_splits/lta/",
        tsfms=init_video_transform_dict()['test'],
        reader='decord_start_end',
        subsample='text',
        split=split,
    )
    dataset = NaturalLanguageQueries(**kwargs)
    print(len(dataset))
