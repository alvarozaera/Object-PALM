import os
os.environ["OMP_NUM_THREADS"] = "64" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "64" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "64" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "64" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "64" # export NUMEXPR_NUM_THREADS=1
import sys
import tqdm
import argparse
import numpy as np
import torch
torch.set_num_threads(16)
import transformers
from sacred import Experiment

import torch
sys.path.append("./")
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser


from model.model import sim_matrix

ex = Experiment('test')

@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride
    config._config['data_loader']['args']['subsample'] = args.subsample

    data_loader = config.initialize('data_loader', module_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    import model.model as module_arch
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(len(data_loader))

    if not os.path.exists(args.save_feats):
        os.mkdir(args.save_feats)

    # extract clip features
    if args.subsample == 'video':
        num_frame = config.config['data_loader']['args']['video_params']['num_frames']
        dim = config.config['arch']['args']['projection_dim']
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                # leave this for now since not doing anything on the gpu
                # pdb.set_trace()
                if os.path.exists(os.path.join(args.save_feats, data['meta']['clip_uid'][0]+'.pt')):
                    print(f"{data['meta']['clip_uid']} is already.")
                    continue
                # this implementation is cautious, we use 4f video-encoder to extract featurs of whole clip.
                f, c, h, w = data['video'].shape[1], data['video'].shape[2], data['video'].shape[3], data['video'].shape[4]

                data['video'] = data['video'][0][:(f // num_frame * num_frame), ]
                data['video'] = data['video'].reshape(-1, num_frame, c, h, w)

                data['video'] = data['video'].to(device)
                outs = torch.zeros(data['video'].shape[0], dim)

                batch = 4
                times = data['video'].shape[0] // batch
                for j in range(times):
                    start = j*batch
                    if (j+1) * batch > data['video'].shape[0]:
                        end = data['video'].shape[0]
                    else:
                        end = (j+1)*batch

                    outs[start:end,] = \
                        model.compute_video(data['video'][start:end,])

                torch.save(outs, os.path.join(args.save_feats, data['meta']['clip_uid'][0]+'.pt'))

    if args.subsample == 'text':
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                raw_text = data['text'][0]
                #print(data['text'], data['meta'])

                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}


                if args.token:
                    text_embed = model.compute_text_tokens(data['text'])[0]
                    num_words = data['text']['attention_mask'][0].sum()
                    text_embed = text_embed[1 : num_words-1]
                else:
                    text_embed = model.compute_text(data['text'])

                torch.save(text_embed, os.path.join(args.save_feats, f'{data["meta"][0]}.pt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default=None,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('-subsample', '--subsample', default='video', type=str, # 0 for vidoe while 1 for text.
                      help='source data from video or text.')
    args.add_argument('--token', default=False, type=bool,
                      help='whether use token features to represent sentence.')
    args.add_argument('--save_feats', default='./egovlplta_raw',
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args, test=True, eval_mode='lta')
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)
    ex.run()
