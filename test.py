import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
 
from coco_loader import *
from torchvision import models                                                                     
from convcap import convcap_G
from vggfeats import Vgg16Feats
from evaluate import language_eval
from resnet101feats import ResNet101
from convcap import convcap_D

def save_test_json(preds, resFile):
  print('Writing %d predictions' % (len(preds)))
  json.dump(preds, open(resFile, 'w')) 

def test(args, split, modelfn=None, model_convcap=None, model_imgcnn=None, model_resnet=None):
  """Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""

  t_start = time.time()



  test_data_gcn = gcn_loader(split=split, seq_per_img=1)
  print('')
  print('[DEBUG] Finding %d gcn_images to split %s' % (len(test_data_gcn.split_ix[split]), split))
  #print(len(train_data.split_ix['train']))   # gcn_loader  ->  image numbers    train - > 113287

  vocab_size = test_data_gcn.vocab_size    #print(vocab_size)   # vocab_size = 14964
  seq_length = test_data_gcn.seq_length    #print (seq_length)   # seq_lenght = 16



  print('...')
  data = coco_loader(args.coco_root, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  data_loader = DataLoader(dataset=data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=False, drop_last=True)

  batchsize = args.batchsize
  max_tokens = data.max_tokens
  num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
  print('[DEBUG] Running inference on %s with %d batches' % (split, num_batches))


  if(modelfn is not None):
    model_imgcnn = Vgg16Feats()
    model_imgcnn.cuda()

    model_convcap = convcap_D(data.numwords, args.num_layers, is_attention=args.attention)
    model_convcap.cuda()

    model_resnet = ResNet101()
    model_resnet.cuda()

    print('[DEBUG] Loading checkpoint %s' % modelfn)
    checkpoint = torch.load(modelfn)
    model_convcap.load_state_dict(checkpoint['state_dict'])
    model_imgcnn.load_state_dict(checkpoint['img_state_dict'])
    model_resnet.load_state_dict(checkpoint['img_state_dict_resnet'])
  else:
    model_imgcnn = model_imgcnn
    model_convcap = model_convcap
    model_resnet = model_resnet

  model_imgcnn.train(False)
  model_convcap.train(False)
  model_resnet.train(False)

  pred_captions = []
  #Test epoch
  for batch_idx, (imgs, _, _, _, img_ids) in \
    tqdm(enumerate(data_loader), total=num_batches):


    use_rela = False
    data1 = test_data_gcn.get_batch(split=split, batch_size=args.batchsize, seq_per_img=1)
    # print(data)   # include get_batch all data[] except ssg_data

    # tmp = [data1['fc_feats'], data1['labels'], data1['masks']]
    # tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    # fc_feats, labels, masks = tmp

    tmp = [data1['att_feats'], data1['att_masks'],
           data1['rela_rela_matrix'], data1['rela_rela_masks'],
           data1['rela_attr_matrix'], data1['rela_attr_masks']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]

    att_feats, att_masks, rela_rela_matrix, rela_rela_masks, \
    rela_attr_matrix, rela_attr_masks = tmp

    # fc_feats = Variable(fc_feats)
    # att_masks = Variable(att_masks)
    # att_feats = Variable(att_feats)

    rela_data = {}
    # rela_data['att_feats'] = att_feats
    # rela_data['att_masks'] = att_masks
    rela_data['rela_matrix'] = rela_rela_matrix
    rela_data['rela_masks'] = rela_rela_masks
    rela_data['attr_matrix'] = rela_attr_matrix
    rela_data['attr_masks'] = rela_attr_masks


    #rela_rela_matrix = rela_rela_matrix.cuda()
    #rela_attr_matrix = Variable(rela_attr_matrix)

    imgs = imgs.view(batchsize, 3, 224, 224)
    imgs_v = Variable(imgs.cuda())

    imgsfc = model_resnet(imgs_v)     ## (20, 2048)
    imgsfeats = model_imgcnn(imgs_v)  ## (20, 512, 7, 7)

    imgsfc = imgsfc.view(batchsize, 2048)

    _, featdim, feat_h, feat_w = imgsfeats.size()

    wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
    wordclass_feed[:,0] = data.wordlist.index('<S>')

    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, _ = model_convcap(imgsfeats, imgsfc, wordclass, rela_data)

      wordact = wordact[:,:,:-1]
      wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)

      wordprobs = F.softmax(wordact_t, dim=1).cpu().data.numpy()
      wordids = np.argmax(wordprobs, axis=1)
      #print(wordprobs.size())
      #print(wordids.size())

      for k in range(batchsize):
        word = data.wordlist[wordids[j+k*(max_tokens-1)]]
        outcaps[k].append(word)
        if(j < max_tokens-1):
          wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]

    for j in range(batchsize):
      num_words = len(outcaps[j]) 
      if 'EOS' in outcaps[j]:
        num_words = outcaps[j].index('EOS')
      outcap = ' '.join(outcaps[j][:num_words])
      pred_captions.append({'image_id': img_ids[j], 'caption': outcap})

  scores = language_eval(pred_captions, args.model_dir, split)

  model_imgcnn.train(True)
  model_convcap.train(True)
  model_resnet.train(True)

  return scores 
 
