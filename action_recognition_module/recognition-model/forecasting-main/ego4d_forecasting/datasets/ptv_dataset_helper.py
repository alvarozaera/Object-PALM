from __future__ import annotations
import gc
import time

import json
import logging
import os
import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

import pickle
import numpy as np

from collections import Counter

logger = logging.getLogger(__name__)


# NOTE: Variables to configure the experiment to run
EGOVLP_FEATURES_PATH_TRAIN = '/cluster/project/cvg/students/azaera/ego_vlp_feats/'
EGOVLP_FEATURES_PATH_VAL = '/cluster/project/cvg/students/azaera/ego_vlp_feats/'
EGOVLP_FEATURES_PATH_TEST = '/cluster/project/cvg/students/azaera/ego_vlp_feats/'
RAM_PLUS_OUTPUTS_PATH = '/cluster/project/cvg/students/azaera/ram_plus_outputs/'

# Options for the experiment
# A - Use the default thresholds defined in RAM++
# B - Use an average of the default thresholds of RAM++ and the 95% quantile found in the dataset
# C - Use the confidence scores after applying the linear transformation and selecting the top-N nouns without threshold
# D - Use the confidence scores after applying the linear transformation and use 0.91 as threshold
# POC - Performs the Proof-of-Concept experiment (it directly loads the noun indices from a file so it can be used for other experiments as a general case with a slight adaptation of the padding mask treatment)
CASE_SELECTED = 'C'

# Json files with the recognized nouns for each action segment used in the POC case
POC_NOUNS_PATH_TRAIN = '/cluster/project/cvg/students/azaera/poc_10_nouns_6_past_3_rand_no_rep_train.json'
POC_NOUNS_PATH_VAL = '/cluster/project/cvg/students/azaera/poc_10_nouns_6_past_3_rand_no_rep_val.json'

# Specific variables for the RAM++ cases
THRESHOLDS_PATH_A = '/cluster/project/cvg/students/azaera/ram_analysis/default_thresholds.pt'
THRESHOLDS_PATH_B = '/cluster/project/cvg/students/azaera/ram_analysis/avg-095-quantile-default-thresholds.pt'
GLOBAL_THRESHOLD_D = 0.91
NO_RESCALED_SCORES_NAME = 'aggregated_logits_hand_boxes_mapped_ego4d.pt' #'trunc-mean-3_logits_hand_boxes_max-per-frame_mapped_ego4d.pt'
RESCALED_SCORES_NAME = 'aggregated_logits_hand_boxes_mapped_ego4d_rescaled.pt' #'trunc-mean-3_logits_hand_boxes_max-per-frame_mapped_ego4d_rescaled.pt'


class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self.path_handler = VideoPathHandler()

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clips = None
        self._next_clip_start_time = 0.0

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        video_index = next(self._video_sampler_iter)

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            try:
                video_path, info_dict = self._labeled_videos[video_index]
                video = self.path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                )
            except Exception as e:
                logger.debug(
                    "Failed to load video with error: {}; trial {}".format(e, i_try)
                )
                continue

            clips = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if not isinstance(clips, list):
                clips = [clips]

            decoded_clips = []
            video_is_null = False
            for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
                clip = video.get_clip(clip_start, clip_end)
                video_is_null = clip is None or clip["video"] is None
                if video_is_null:
                    break
                decoded_clips.append(clip)

            self._next_clip_start_time = clip_end

            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                video.close()
                self._next_clip_start_time = 0.0

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue


            if len(decoded_clips) == 1:
                frames = decoded_clips[0]["video"]
                audio_samples = decoded_clips[0]["audio"]
            else:
                clip_frames = [
                    uniform_temporal_subsample(x["video"], num_samples=64)
                    for x in decoded_clips
                ]
                frames = torch.stack(clip_frames, dim=0)

                clip_audio = [x["audio"] for x in decoded_clips]
                audio_samples = None
                if None not in clip_audio:
                    audio_samples = torch.stack(clip_audio, dim=0)

            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

class LabeledFeatureDataset(torch.utils.data.IterableDataset):

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        filename,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        logger.info("Loading labeled feature dataset...")

        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self.path_handler = VideoPathHandler()

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clips = None
        self._next_clip_start_time = 0.0

        self.filename = filename

        if CASE_SELECTED == 'POC':
            with open(POC_NOUNS_PATH_TRAIN, 'r') as f:
                self.json_poc_train = json.load(f)

            with open(POC_NOUNS_PATH_VAL, 'r') as f:
                self.json_poc_val = json.load(f)

        if CASE_SELECTED == 'A':
            self.class_thresholds = torch.load(THRESHOLDS_PATH_A)
        elif CASE_SELECTED == 'B':
            self.class_thresholds = torch.load(THRESHOLDS_PATH_B)
        elif CASE_SELECTED == 'D':
            self.class_thresholds = GLOBAL_THRESHOLD_D # Global threshold instead of per-class threshold

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def load_egovlp_feature(self, clip_uid, action_idx):
        if 'train' in self.filename:
            filename_egovlp = f'{EGOVLP_FEATURES_PATH_TRAIN}/{clip_uid}_{action_idx}.pt'
        elif 'val' in self.filename:
            filename_egovlp = f'{EGOVLP_FEATURES_PATH_VAL}/{clip_uid}_{action_idx}.pt'
        else:
            filename_egovlp = f'{EGOVLP_FEATURES_PATH_TEST}/{clip_uid}_{action_idx}.pt'

        egovlp_feature = torch.load(filename_egovlp)

        non_zero = (egovlp_feature[..., 0] != 0).sum().item()
        padded = torch.zeros(256 * 12)
        padded[:non_zero * 256] = egovlp_feature[:non_zero].reshape(-1)
        return padded
    

    ### Function used to load the noun indices for the POC case ####
    def load_recognized_noun_indices_poc(self, clip_uid, action_idx):
        MAX_NOUNS = 10

        if 'train' in self.filename:
            nouns = self.json_poc_train[f"{clip_uid}_{action_idx}"]
        else:
            nouns = self.json_poc_val[f"{clip_uid}_{action_idx}"]

        return torch.tensor(nouns, dtype=torch.long), torch.zeros(MAX_NOUNS)


    
    #### Function used to load the noun indices using scores and thresholds obtained using RAM++ ####
    def load_recognized_noun_indices(self, clip_uid, action_idx):
        MAX_NOUNS = 15

        ram_plus_base_outputs_path = f'{RAM_PLUS_OUTPUTS_PATH}/{clip_uid}_{action_idx}/'

        # Load scores without rescaling
        if CASE_SELECTED == 'A' or CASE_SELECTED == 'B':
            try:
                logits = torch.load(os.path.join(ram_plus_base_outputs_path, NO_RESCALED_SCORES_NAME))
                logits.requires_grad = False
            except:
                logger.debug(f'Failed to load scores without rescaling for {clip_uid}_{action_idx}: using a full padding mask')
                return torch.zeros(MAX_NOUNS, dtype=torch.long), torch.ones(MAX_NOUNS)
        

        # Load scores rescaled using piecewise linear transformation
        if CASE_SELECTED == 'C' or CASE_SELECTED == 'D':
            try:
                logits = torch.load(os.path.join(ram_plus_base_outputs_path, RESCALED_SCORES_NAME))
                logits.requires_grad = False
            except:
                logger.debug(f'Failed to load scores rescaled using piecewise linear transformation for {clip_uid}_{action_idx}: using a full padding mask')
                return torch.zeros(MAX_NOUNS, dtype=torch.long), torch.ones(MAX_NOUNS)


        # Ignore indices of very common ego4d nouns: 2 (arm) 198 (hand) 164 (foot) 504 (person) 230 (leg)
        ignore_indices = torch.tensor([2, 198, 164, 504, 230])
        logits[ignore_indices] = -float('inf')


        # CASE C: Select the top MAX_NOUNS nouns without threshold
        if CASE_SELECTED == 'C':
            top_nouns = torch.topk(logits, MAX_NOUNS)
            top_nouns_indices = top_nouns.indices
            top_nouns_confidences = top_nouns.values
            padding_mask = torch.zeros(MAX_NOUNS)

        # CASES A, B, and D: Select the top MAX_NOUNS nouns using thresholds
        if CASE_SELECTED == 'A' or CASE_SELECTED == 'B' or CASE_SELECTED == 'D':
            # CASE D: self.class_thresholds is a global threshold (a single value)
            # CASES A and B: self.class_thresholds is a tensor with the thresholds for each class
            top_nouns_indices = torch.where(logits >= self.class_thresholds)[0]
            n_nouns = len(top_nouns_indices)
            # Pad with zeros if there are less than MAX_NOUNS nouns
            if len(top_nouns_indices) < MAX_NOUNS:
                top_nouns_indices = torch.cat((top_nouns_indices, torch.zeros(MAX_NOUNS - n_nouns, dtype=torch.long)))
            elif len(top_nouns_indices) > MAX_NOUNS:
                # Select the top MAX_NOUNS nouns based on confidence
                top_indices_among_selected = torch.topk(logits[top_nouns_indices], MAX_NOUNS).indices
                top_nouns_indices = top_nouns_indices[top_indices_among_selected]
            padding_mask = torch.zeros(MAX_NOUNS)
            padding_mask[n_nouns:] = 1 # 1 where is padded


        return top_nouns_indices, padding_mask



    def __next__(self) -> dict:
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        video_index = next(self._video_sampler_iter)
        video_path, info_dict = self._labeled_videos[video_index]

        egovlp_feature = []
        padding_mask_bert = []
        top_nouns_indices = []
        for input_clip in info_dict["input_clips"]:
            find = False
            for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
                try: 
                    clip_uid = input_clip["clip_uid"]
                    action_idx = input_clip["action_idx"]
                    if action_idx < self._MAX_CONSECUTIVE_FAILURES:
                        action_idx += i_try
                    else:
                        action_idx -= i_try
                    loaded_feature = self.load_egovlp_feature(clip_uid, action_idx) 
                    egovlp_feature.append(loaded_feature)
                    if CASE_SELECTED == 'POC':
                        top_nouns_ind, padding_mask = self.load_recognized_noun_indices_poc(clip_uid, action_idx)
                    else:
                        top_nouns_ind, padding_mask = self.load_recognized_noun_indices(clip_uid, action_idx)
                    padding_mask_bert.append(padding_mask)
                    top_nouns_indices.append(top_nouns_ind)
                    find = True 
                    break
                except Exception as e:
                    logger.debug("Failed to load EgoVLP feature with error: {}; trial {}".format(e, i_try))
            if not find:
                print("Impute {} with a fixed one".format(input_clip["clip_uid"]))
                loaded_feature = self.load_egovlp_feature("5085ff96-926e-4462-adb9-57f91215e881", 6)
                egovlp_feature.append(loaded_feature)
                top_nouns_ind, padding_mask = self.load_recognized_noun_indices("5085ff96-926e-4462-adb9-57f91215e881", 6)
                padding_mask_bert.append(padding_mask)
                top_nouns_indices.append(top_nouns_ind)


        sample_dict = {
            "feature": egovlp_feature,
            "padding_mask_bert": padding_mask_bert,
            "top_nouns_indices": top_nouns_indices,
            "video_index": video_index,
            **info_dict,
        }

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict


    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset



class UntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_end_boundary - clip_start_boundary

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
        return ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        )


class ForecastingClipSampler:
    def __init__(self, clip_sampler: ClipSampler) -> None:
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> List[ClipInfo]:
        clip_infos = []
        for input_clip in clip_info["input_clips"]:
            clip_start_boundary = input_clip["clip_start_sec"]
            clip_end_boundary = input_clip["clip_end_sec"]
            duration = clip_end_boundary - clip_start_boundary

            # Sample between 0 and duration of untrimmed clip, then add back start boundary.
            clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
            clip_infos.append(
                ClipInfo(
                    clip_info.clip_start_sec + clip_start_boundary,
                    clip_info.clip_end_sec + clip_start_boundary,
                    clip_info.clip_index,
                    clip_info.aug_index,
                    clip_info.is_last_clip,
                )
            )
        return clip_infos


def clip_recognition_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    
    assert os.path.exists(data_path), 'Please run data/parse_ego4d_json.py first. Will change this later'

    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                annotations = json.load(f)['clips']
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict). For recognition, the annotation_dict contains
        # the verb and noun label, and the annotation boundaries.
        untrimmed_clip_annotations = []

        for entry in annotations:
            if "test_unannotated" in data_path:
                untrimmed_clip_annotations.append(
                    (
                        os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
                        {
                            "clip_start_sec": entry['action_clip_start_sec'],
                            "clip_end_sec": entry['action_clip_end_sec'],
                            "action_idx": entry['action_idx'],
                        },
                    )
                )
            else:
                untrimmed_clip_annotations.append(
                    (
                        os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
                        {
                            "clip_start_sec": entry['action_clip_start_sec'],
                            "clip_end_sec": entry['action_clip_end_sec'],
                            "noun_label": entry['noun_label'],
                            "verb_label": entry['verb_label'],
                            "action_idx": entry['action_idx'],
                        },
                    )
                )

    else:
        raise FileNotFoundError(f"{data_path} not found.")

    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        UntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

def clip_forecasting_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    num_input_actions: int,
    num_future_actions: int,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    filename = data_path.split('/')[-1]
    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                entries = json.load(f)['clips']
        except Exception as e:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset. {e}")

        # if entries do not have verb/noun labels (test set) then give dummy ones
        for entry in entries:
            if 'verb_label' not in entry:
                entry.update({'verb_label': -1, 'noun_label': -1})

        # rename keys for pytorchvideo
        for entry in entries:
            entry.update({
                'clip_start_sec': entry.pop('action_clip_start_sec'),
                'clip_end_sec': entry.pop('action_clip_end_sec'),
            })


        # group annotations by clip_uid
        annotations = collections.defaultdict(list)
        for entry in entries:
            annotations[entry['clip_uid']].append(entry)

        # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
        annotations = {
            clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
            for clip_uid in annotations
        }

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict). For forecasting, annotation_dict contains
        # the input boundaries to be decoded, any observed clip annotations within
        # those boundaries, and a list of num_future_actions clip annotations (including
        # labels and boundaries).

        #IDK why. but clips/ directory doesn't include following clip .mp4 file
        not_found_clip_ids = ['440656ae-cb82-464e-b320-25c8e693ad84']

        untrimmed_clip_annotations = []
        for clip_uid, video_clips in annotations.items():
            if clip_uid in not_found_clip_ids:
                continue
            video_path = os.path.join(video_path_prefix, f'{clip_uid}.mp4')
            if len(video_clips) <= 0:
                continue

            # Extract forecasting annotations from video clips.
            for i in range(len(video_clips) - num_input_actions ): #+1??
                input_clips = copy.deepcopy(video_clips[i : i + num_input_actions])
                forecast_clips = copy.deepcopy(video_clips[i : i + num_input_actions])
                untrimmed_clip_annotations.append(
                    (
                        video_path,
                        {
                            "input_clips": input_clips,
                            "forecast_clips": forecast_clips,
                        },
                    )
                )
    else:
        raise FileNotFoundError(f"{data_path} not found.")
    
    dataset = LabeledFeatureDataset(
        filename,
        untrimmed_clip_annotations,
        ForecastingClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset



