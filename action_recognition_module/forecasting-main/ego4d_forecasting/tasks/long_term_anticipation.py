import torch
import pdb
import numpy as np
import itertools
from fvcore.nn.precise_bn import get_bn_modules
import json

from ..evaluation import lta_metrics as metrics
from ..utils import distributed as du
from ..utils import logging
from ..utils import misc
from ..tasks.video_task import VideoTask

logger = logging.get_logger(__name__)

import json 
import collections
from tqdm import tqdm

import torch.nn as nn

import torchnet as tnt

class MultiTaskClassificationTask(VideoTask):
    checkpoint_metric = "val_top1_noun_err"

    def training_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        loss1 = self.loss_fun(preds[0], labels[:, 0])
        loss2 = self.loss_fun(preds[1], labels[:, 1])
        loss = loss1 + loss2
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        step_result = {
            "loss": loss,
            "train_loss": loss.item(),
            "train_top1_verb_err": top1_err_verb.item(),
            "train_top5_verb_err": top5_err_verb.item(),
            "train_top1_noun_err": top1_err_noun.item(),
            "train_top5_noun_err": top5_err_noun.item(),
        }

        return step_result

    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )
        return {
            "val_top1_verb_err": top1_err_verb.item(),
            "val_top5_verb_err": top5_err_verb.item(),
            "val_top1_noun_err": top1_err_noun.item(),
            "val_top5_noun_err": top5_err_noun.item(),
        }

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        inputs, labels, clip_id, _ = batch
        preds = self.forward(inputs)
        return {
            "preds_verb": preds[0],
            "preds_noun": preds[1],
            "labels": labels,
            "clip_ids": clip_id,
        }

    def test_epoch_end(self, outputs):
        preds_verbs = torch.cat([x["preds_verb"] for x in outputs])
        preds_nouns = torch.cat([x["preds_noun"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        clip_ids = [x["clip_ids"] for x in outputs]
        clip_ids = [item for sublist in clip_ids for item in sublist]

        # Gather all labels from distributed processes.
        preds_verbs = torch.cat(du.all_gather([preds_verbs]), dim=0)
        preds_nouns = torch.cat(du.all_gather([preds_nouns]), dim=0)
        labels = torch.cat(du.all_gather([labels]), dim=0)
        clip_ids = list(itertools.chain(*du.all_gather_unaligned(clip_ids)))

        # Ensemble multiple predictions of the same view together. This relies on the
        # fact that the dataloader reads multiple clips of the same video at different
        # spatial crops.
        video_labels = {}
        video_verb_preds = {}
        video_noun_preds = {}
        assert preds_verbs.shape[0] == preds_nouns.shape[0]
        for i in range(preds_verbs.shape[0]):
            vid_id = clip_ids[i]
            video_labels[vid_id] = labels[i]
            if vid_id not in video_verb_preds:
                video_verb_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[0]),
                    device=preds_verbs.device,
                    dtype=preds_verbs.dtype,
                )
                video_noun_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[1]),
                    device=preds_nouns.device,
                    dtype=preds_nouns.dtype,
                )

            if self.cfg.DATA.ENSEMBLE_METHOD == "sum":
                video_verb_preds[vid_id] += preds_verbs[i]
                video_noun_preds[vid_id] += preds_nouns[i]
            elif self.cfg.DATA.ENSEMBLE_METHOD == "max":
                video_verb_preds[vid_id] = torch.max(
                    video_verb_preds[vid_id], preds_verbs[i]
                )
                video_noun_preds[vid_id] = torch.max(
                    video_noun_preds[vid_id], preds_nouns[i]
                )

        video_verb_preds = torch.stack(list(video_verb_preds.values()), dim=0)
        video_noun_preds = torch.stack(list(video_noun_preds.values()), dim=0)
        video_labels = torch.stack(list(video_labels.values()), dim=0)
        top1_verb_err, top5_verb_err = metrics.topk_errors(
            video_verb_preds, video_labels[:, 0], (1, 5)
        )
        top1_noun_err, top5_noun_err = metrics.topk_errors(
            video_noun_preds, video_labels[:, 1], (1, 5)
        )
        errors = {
            "top1_verb_err": top1_verb_err,
            "top5_verb_err": top5_verb_err,
            "top1_noun_err": top1_noun_err,
            "top5_noun_err": top5_noun_err,
        }
        for k, v in errors.items():
            self.log(k, v.item())


class LongTermAnticipationTask(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpoint_metric = f"val_top1_acc"

    def forward(self, inputs, tgts=None):
        return self.model(inputs, tgts=tgts)
    
    def training_step(self, batch, batch_idx):
        # Labels is tensor of shape (batch_size, time, label_dim)
        input, labels, observed_labels, _, _ = batch
        # Preds is a list of tensors of shape (B, Z, C), where
        # - B is batch size,
        # - Z is number of future predictions,
        # - C is the class
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.forward(input)
        assert len(preds) == len(self.cfg.MODEL.NUM_CLASSES), len(preds)
        
        loss = 0
        step_result = {}

        #independent CE loss for all 8 head (4 for verb and 4 for noun)
        for head_idx, pred_head in enumerate(preds):
            for seq_idx in range(pred_head.shape[1]):

                loss += self.loss_fun(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx]
                )
                #print(f'loss_calculation{head_idx}_{seq_idx}: {pred_head[:, seq_idx].shape}, {labels[:, seq_idx, head_idx].shape}')
                #print(pred_head[0, seq_idx])

                top1_err, top5_err = metrics.distributed_topk_errors(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx], (1, 5)
                )

                step_result[f"train_{seq_idx}_{head_idx}_top1_err"] = top1_err.item()
                step_result[f"train_{seq_idx}_{head_idx}_top5_err"] = top5_err.item()

        for head_idx in range(len(preds)):
            step_result[f"train_{head_idx}_top1_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top1" in k]
            )
            step_result[f"train_{head_idx}_top5_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top5" in k]
            )

        step_result["loss"] = loss
        step_result["train_loss"] = loss.item()

        return step_result       


    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx):
        input, forecast_labels, _, last_clip_ids, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.model.generate(input, k=k)  # [(B, K, Z)]
        
        return {
            'last_clip_ids': last_clip_ids,
            'verb_preds': preds[0],
            'noun_preds': preds[1],
        }

    def validation_epoch_end(self, outputs):
        test_outputs = {}
        for key in ['verb_preds', 'noun_preds']:
            preds = torch.cat([x[key] for x in outputs], 0)
            preds = self.all_gather(preds).unbind()
            test_outputs[key] = torch.cat(preds, 0)

        last_clip_ids = [x['last_clip_ids'] for x in outputs]
        last_clip_ids = [item for sublist in last_clip_ids for item in sublist]
        last_clip_ids = list(itertools.chain(*du.all_gather_unaligned(last_clip_ids)))
        test_outputs['last_clip_ids'] = last_clip_ids

        if du.get_local_rank() == 0:
            pred_dict = {}
            for idx in range(len(test_outputs['last_clip_ids'])):
                pred_dict[test_outputs['last_clip_ids'][idx]] = {
                    'verb': test_outputs['verb_preds'][idx].cpu().tolist(),
                    'noun': test_outputs['noun_preds'][idx].cpu().tolist(),
                }       

            pred_action = pred_dict

            annotation_folder = self.cfg.DATA.PATH_TO_DATA_DIR


            def get_dset(split):
                with open("{}/fho_lta_{}.json".format(annotation_folder, split), "r") as f:
                    dset = json.load(f)

                annotations = collections.defaultdict(list)
                for entry in dset["clips"]:
                    annotations[entry['clip_uid']].append(entry)

                # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
                annotations = {
                    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
                    for clip_uid in annotations
                }
                return annotations

            annotations = get_dset("val")

            nprev = 4
            torun = []
            for clip_uid in annotations:
                l = len(annotations[clip_uid])
                for i in range(l - nprev):
                    torun.append((clip_uid, i))

            pred_action_aggregated = collections.defaultdict(lambda : {"verb": [], "noun": []})
            for k in pred_action:
                clip_uid, idx = k.split("_")
                idx = int(idx)
                for vpd, npd in zip(pred_action[k]["verb"], pred_action[k]["noun"]):
                    offset = len(vpd)
                    for i in range(offset):
                        #if idx > 7 and i != offset - 1:
                        #    continue
                        pred_action_aggregated["{}_{}".format(clip_uid, idx + i - offset + 1)]["verb"].append(vpd[i])
                        pred_action_aggregated["{}_{}".format(clip_uid, idx + i - offset + 1)]["noun"].append(npd[i])

            for k in pred_action_aggregated:
                get_most_common = lambda x : max(set(x), key=x.count)
                pred_action_aggregated[k]["verb_agg"] = get_most_common(pred_action_aggregated[k]["verb"])
                pred_action_aggregated[k]["noun_agg"] = get_most_common(pred_action_aggregated[k]["noun"])


            total_actions = len(torun)
            top1_verb = 0
            top1_noun = 0
            top1_action = 0

            top5_verb = 0
            top5_noun = 0
            top5_action = 0

            fail=0
            for arg in tqdm(torun, total=len(torun)):
                try:
                    clip_uid, i = arg

                    idx = annotations[clip_uid][i]["action_idx"]

                    verb = pred_action_aggregated[clip_uid + "_" + str(idx)]["verb_agg"]
                    noun = pred_action_aggregated[clip_uid + "_" + str(idx)]["noun_agg"]

                    true_verb = annotations[clip_uid][i]["verb_label"]
                    true_noun = annotations[clip_uid][i]["noun_label"]

                    if true_verb == verb and true_noun==noun:
                        top1_verb +=1
                        top1_noun +=1
                        top1_action +=1
                    elif true_verb == verb:
                        top1_verb +=1
                    elif true_noun==noun:
                        top1_noun +=1

                    verb_list = pred_action_aggregated[clip_uid + "_" + str(idx)]["verb"]
                    noun_list = pred_action_aggregated[clip_uid + "_" + str(idx)]["noun"]

                    if true_verb in verb_list and true_noun in noun_list:
                        top5_verb +=1
                        top5_noun +=1
                        top5_action +=1  
                    elif true_verb in verb_list:
                        top5_verb +=1
                    elif true_noun in noun_list:
                        top5_noun +=1

                except:
                    fail +=1
                    continue      

            top1_verb_acc = round(top1_verb/total_actions*100, 2)
            top1_noun_acc = round(top1_noun/total_actions*100, 2)
            top1_action_acc = round(top1_action/total_actions*100, 2)

            top5_verb_acc = round(top5_verb/total_actions*100, 2)
            top5_noun_acc = round(top5_noun/total_actions*100, 2)
            top5_action_acc = round(top5_action/total_actions*100, 2)

            #print(output_dir)
            print(f'Top 1: Verb Acc: {top1_verb_acc}, Noun Acc: {top1_noun_acc}, Action Acc: {top1_action_acc}')
            print(f'Top 5: Verb Acc: {top5_verb_acc}, Noun Acc: {top5_noun_acc}, Action Acc: {top5_action_acc}')
            print(f'failure cases: {fail}')

            self.log("val_top1_acc", top1_action_acc)


    def test_step(self, batch, batch_idx):
        input, forecast_labels, _, last_clip_ids, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.model.generate(input, k=k)  # [(B, K, Z)]
        
        return {
            'last_clip_ids': last_clip_ids,
            'verb_preds': preds[0],
            'noun_preds': preds[1],
        }

    def test_epoch_end(self, outputs):
        test_outputs = {}
        for key in ['verb_preds', 'noun_preds']:
            preds = torch.cat([x[key] for x in outputs], 0)
            preds = self.all_gather(preds).unbind()
            test_outputs[key] = torch.cat(preds, 0)

        last_clip_ids = [x['last_clip_ids'] for x in outputs]
        last_clip_ids = [item for sublist in last_clip_ids for item in sublist]
        last_clip_ids = list(itertools.chain(*du.all_gather_unaligned(last_clip_ids)))
        test_outputs['last_clip_ids'] = last_clip_ids

        if du.get_local_rank() == 0:
            pred_dict = {}
            for idx in range(len(test_outputs['last_clip_ids'])):
                pred_dict[test_outputs['last_clip_ids'][idx]] = {
                    'verb': test_outputs['verb_preds'][idx].cpu().tolist(),
                    'noun': test_outputs['noun_preds'][idx].cpu().tolist(),
                }       
            json.dump(pred_dict, open('outputs.json', 'w'))

            pred_action = pred_dict

            annotation_folder = self.cfg.DATA.PATH_TO_DATA_DIR


            def get_dset(split):
                with open("{}/fho_lta_{}.json".format(annotation_folder, split), "r") as f:
                    dset = json.load(f)

                annotations = collections.defaultdict(list)
                for entry in dset["clips"]:
                    annotations[entry['clip_uid']].append(entry)

                # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
                annotations = {
                    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
                    for clip_uid in annotations
                }
                return annotations

            annotations = get_dset("val")

            nprev = 4
            torun = []
            for clip_uid in annotations:
                l = len(annotations[clip_uid])
                for i in range(l - nprev):
                    torun.append((clip_uid, i))

            pred_action_aggregated = collections.defaultdict(lambda : {"verb": [], "noun": []})
            for k in pred_action:
                clip_uid, idx = k.split("_")
                idx = int(idx)
                for vpd, npd in zip(pred_action[k]["verb"], pred_action[k]["noun"]):
                    offset = len(vpd)
                    for i in range(offset):
                        #if idx > 7 and i != offset - 1:
                        #    continue
                        pred_action_aggregated["{}_{}".format(clip_uid, idx + i - offset + 1)]["verb"].append(vpd[i])
                        pred_action_aggregated["{}_{}".format(clip_uid, idx + i - offset + 1)]["noun"].append(npd[i])

            for k in pred_action_aggregated:
                get_most_common = lambda x : max(set(x), key=x.count)
                pred_action_aggregated[k]["verb_agg"] = get_most_common(pred_action_aggregated[k]["verb"])
                pred_action_aggregated[k]["noun_agg"] = get_most_common(pred_action_aggregated[k]["noun"])


            total_actions = len(torun)
            top1_verb = 0
            top1_noun = 0
            top1_action = 0

            top5_verb = 0
            top5_noun = 0
            top5_action = 0

            fail=0
            for arg in tqdm(torun, total=len(torun)):
                try:
                    clip_uid, i = arg

                    idx = annotations[clip_uid][i]["action_idx"]

                    verb = pred_action_aggregated[clip_uid + "_" + str(idx)]["verb_agg"]
                    noun = pred_action_aggregated[clip_uid + "_" + str(idx)]["noun_agg"]

                    true_verb = annotations[clip_uid][i]["verb_label"]
                    true_noun = annotations[clip_uid][i]["noun_label"]

                    if true_verb == verb and true_noun==noun:
                        top1_verb +=1
                        top1_noun +=1
                        top1_action +=1
                    elif true_verb == verb:
                        top1_verb +=1
                    elif true_noun==noun:
                        top1_noun +=1

                    verb_list = pred_action_aggregated[clip_uid + "_" + str(idx)]["verb"]
                    noun_list = pred_action_aggregated[clip_uid + "_" + str(idx)]["noun"]

                    if true_verb in verb_list and true_noun in noun_list:
                        top5_verb +=1
                        top5_noun +=1
                        top5_action +=1  
                    elif true_verb in verb_list:
                        top5_verb +=1
                    elif true_noun in noun_list:
                        top5_noun +=1

                except:
                    fail +=1
                    continue      

            top1_verb_acc = round(top1_verb/total_actions*100, 2)
            top1_noun_acc = round(top1_noun/total_actions*100, 2)
            top1_action_acc = round(top1_action/total_actions*100, 2)

            top5_verb_acc = round(top5_verb/total_actions*100, 2)
            top5_noun_acc = round(top5_noun/total_actions*100, 2)
            top5_action_acc = round(top5_action/total_actions*100, 2)

            #print(output_dir)
            print(f'Top 1: Verb Acc: {top1_verb_acc}, Noun Acc: {top1_noun_acc}, Action Acc: {top1_action_acc}')
            print(f'Top 5: Verb Acc: {top5_verb_acc}, Noun Acc: {top5_noun_acc}, Action Acc: {top5_action_acc}')
            print(f'failure cases: {fail}')

