import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models                                                                     

from coco_loader import *
from convcap import convcap_G
from convcap import convcap_D
from vggfeats import Vgg16Feats
from resnet101feats import ResNet101
from tqdm import tqdm 
from test import test 

def repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img):
  """Repeat image features ncap_per_img times"""

  batchsize, featdim, feat_h, feat_w = imgsfeats.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfeats = imgsfeats.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim, feat_h, feat_w)
  imgsfeats = imgsfeats.contiguous().view(\
    batchsize_cap, featdim, feat_h, feat_w)

  batchsize, featdim = imgsfc7.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfc7 = imgsfc7.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim)
  imgsfc7 = imgsfc7.contiguous().view(\
    batchsize_cap, featdim)

  return imgsfeats, imgsfc7


def train(args):
  """Trains model for args.nepochs (default = 30)"""
  use_rela = False
  t_start = time.time()


  train_data_gcn = gcn_loader(split='train', seq_per_img=args.seq_per_img)
  print('[DEBUG] Finding %d gcn_images to split train' % (len(train_data_gcn.split_ix['train'])))
  #print(len(train_data.split_ix['train']))   # gcn_loader  ->  image numbers    train - > 113287

  # vocab_size = train_data_gcn.vocab_size    #print(vocab_size)   # vocab_size = 14964
  # seq_length = train_data_gcn.seq_length    #print(seq_length)   # seq_lenght = 16



  print('...')
  train_data_coco = coco_loader(args.coco_root, split='train', ncap_per_img=args.seq_per_img)
  print('[DEBUG] Finding %d cap_images to split train' % (len(train_data_coco.ids)))
  #print(len(train_data_coco.ids))  # train split data 113287

  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
  train_data_coco_loader = DataLoader(dataset=train_data_coco, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=True, drop_last=True)



  model_resnet = ResNet101()
  model_resnet.cuda()
  model_resnet.train(True)

  #Load pre-trained imgcnn
  model_imgcnn = Vgg16Feats()
  model_imgcnn.cuda()
  model_imgcnn.train(True)

  #Convcap model
  model_convcap_G = convcap_G(train_data_coco.numwords, args.num_layers, is_attention=args.attention)
  model_convcap_G.cuda()
  model_convcap_G.train(True)

  model_convcap_D = convcap_D(train_data_coco.numwords, args.num_layers, is_attention=args.attention)
  model_convcap_D.cuda()
  model_convcap_D.train(True)

  optimizer_G = optim.Adam(model_convcap_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
  scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=args.lr_step_size, gamma=.1)
  optimizer_D = optim.Adam(model_convcap_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
  scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step_size, gamma=.1)
  img_optimizer = None
  img_optimizer1 = None


  batchsize = args.batchsize
  seq_per_img = args.seq_per_img
  batchsize_cap = batchsize * seq_per_img
  max_tokens = train_data_coco.max_tokens
  nbatches = np.int_(np.floor((len(train_data_coco.ids) * 1.) / batchsize))
  bestscore = .0


  for epoch in range(args.epochs):
    #loss_train = 0.

    if (epoch == args.finetune_after):
      img_optimizer = optim.RMSprop(model_imgcnn.parameters(), lr=1e-5)
      img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=args.lr_step_size, gamma=.1)

      img_optimizer1 = optim.RMSprop(model_resnet.parameters(), lr=1e-5)
      img_scheduler1 = lr_scheduler.StepLR(img_optimizer1, step_size=args.lr_step_size, gamma=.1)

    scheduler_G.step()
    scheduler_D.step()
    if(img_optimizer):
      img_scheduler.step()

    if (img_optimizer1):
      img_scheduler1.step()

    # One epoch of train
    for batch_idx, (imgs, captions, wordclass, mask, _) in \
      tqdm(enumerate(train_data_coco_loader), total=nbatches):

      imgs = imgs.view(batchsize, 3, 224, 224)

      wordclass = wordclass.view(batchsize_cap, max_tokens)

      #rela_rela_matrix = rela_rela_matrix.view(20, 147)
      #print(rela_rela_matrix)
      #print(rela_rela_matrix.shape)
      #print(type(rela_rela_matrix))

      mask = mask.view(batchsize_cap, max_tokens)

      imgs_v = Variable(imgs).cuda()
      wordclass_v = Variable(wordclass).cuda()

      wordclass_feed = np.zeros((batchsize_cap, max_tokens), dtype='int64')
      wordclass_feed[:,0] = train_data_coco.wordlist.index('<S>')
      wordclass_zeros = Variable(torch.from_numpy(wordclass_feed)).cuda()

      # imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
      imgsfeats = model_imgcnn(imgs_v)
      imgsfc = model_resnet(imgs_v)

      imgsfc = imgsfc.view(batchsize, 2048)

      #print(imgsfc.size())
      imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc, args.seq_per_img)
      _, _, feat_h, feat_w = imgsfeats.size()

      # print(imgsfeats.size())
      # print(imgsfc7.size())


      data = train_data_gcn.get_batch(split=args.train_split, batch_size=args.batchsize, seq_per_img=args.seq_per_img)
      #print(data)   # include get_batch all data[] except ssg_data

      # tmp = [data['fc_feats'], data['labels'], data['masks']]
      # tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
      # fc_feats, labels, masks = tmp

      tmp = [data['att_feats'], data['att_masks'],
             data['rela_rela_matrix'], data['rela_rela_masks'],
             data['rela_attr_matrix'], data['rela_attr_masks']]
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
      #
      # #rela_data = Variable(rela_data)



      '''  Train the discriminator  '''

      # Loss for real
      optimizer_D.zero_grad()                           # wordclass_v (100, 15)
      if(img_optimizer):
        img_optimizer.zero_grad()

      if (img_optimizer1):
        img_optimizer1.zero_grad()


      if (args.attention == True):
        # rela_data, use_rela, imgsfeats, imgsfc7, wordclass_v
        wordact_D_real, attn = model_convcap_D(imgsfeats, imgsfc7, wordclass_v, rela_data)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        wordact_D_real, _ = model_convcap_D(imgsfeats, imgsfc7, wordclass_v, rela_data)

      wordact = wordact_D_real[:,:,:-1]    #
      wordclass_change = wordclass_v[:,1:]
      mask = mask[:,1:].contiguous()

      wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize_cap * (max_tokens - 1), -1)
      wordclass_d_real = wordclass_change.contiguous().view(batchsize_cap * (max_tokens - 1), 1)

      maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

      if (args.attention == True):
        # Cross-entropy loss and attention loss of Show, Attend and Tell
        loss_D_real = F.cross_entropy(wordact_t[maskids, ...], wordclass_d_real[maskids, ...].contiguous().view(maskids.shape[0])) \
               + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) \
               / (batchsize_cap * feat_h * feat_w)
      else:
        loss_D_real = F.cross_entropy(wordact_t[maskids, ...], wordclass_d_real[maskids, ...].contiguous().view(maskids.shape[0]))

      loss_D_real.backward(retain_graph=True)



      '''Loss for fake'''
      # generator output the fake
      if (args.attention == True):
        fake_D, attn = model_convcap_G(imgsfeats, imgsfc7, wordclass_zeros, rela_data)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        fake_D, _ = model_convcap_G(imgsfeats, imgsfc7, wordclass_zeros, rela_data)

      outcaps_G_fake = np.empty((batchsize_cap, 0)).tolist()

      fake_D = fake_D[:,:,:-1]   # (100, 9221, 14)

      for j in range(max_tokens-1):
        wordact_t1 = fake_D.permute(0, 2, 1).contiguous().view(batchsize_cap*(max_tokens-1), -1)    # (100*14, 9221)
        wordprobs = F.softmax(wordact_t1, dim=1).cpu().data.numpy()
        wordids = np.argmax(wordprobs, axis=1)     # 1400

        for k in range(batchsize_cap):
          word = wordids[j+k*(max_tokens-1)]
          outcaps_G_fake[k].append(word)

      outcaps_G_fake = np.matrix(outcaps_G_fake)
      outcaps_G_fake = Variable(torch.from_numpy(outcaps_G_fake)).cuda()    # (100, 14)

      word_start = Variable(torch.Tensor(batchsize_cap, 1).fill_(50)).cuda()     ## 50 = <S>
      outcaps_G_fake = outcaps_G_fake.float()
      outcaps_G_fake1 = torch.cat([word_start, outcaps_G_fake], 1)
      outcaps_G_fake1 = outcaps_G_fake1.long()


      # input the fake to discriminator
      if (args.attention == True):
        wordact_D_fake, attn = model_convcap_D(imgsfeats, imgsfc7, outcaps_G_fake1, rela_data)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        wordact_D_fake, _ = model_convcap_D(imgsfeats, imgsfc7, outcaps_G_fake1, rela_data)

      wordact1 = wordact_D_fake[:,:,:-1]
      wordclass_change1 = wordclass_v[:, 1:]
      mask1 = mask[:, 1:].contiguous()

      wordact_t2 = wordact1.permute(0, 2, 1).contiguous().view(batchsize_cap * (max_tokens - 1), -1)
      wordclass_d_fake = wordclass_change1.contiguous().view(batchsize_cap * (max_tokens - 1), 1)

      maskids = torch.nonzero(mask1.view(-1)).numpy().reshape(-1)

      if (args.attention == True):
        # Cross-entropy loss and attention loss of Show, Attend and Tell
        loss_D_fake = F.cross_entropy(wordact_t2[maskids, ...], wordclass_d_fake[maskids, ...].contiguous().view(maskids.shape[0])) \
                + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) \
                / (batchsize_cap * feat_h * feat_w)
      else:
        loss_D_fake = F.cross_entropy(wordact_t2[maskids, ...], wordclass_d_fake[maskids, ...].contiguous().view(maskids.shape[0]))

      loss_D_fake.backward(retain_graph=True)
      loss_D = (loss_D_real + loss_D_fake) / 2
      #loss_D.backward(retain_variables=True)
      optimizer_D.step()





      '''  Train the Generator  '''

      # generator output real
      optimizer_G.zero_grad()

      if (args.attention == True):
        fake_G, attn = model_convcap_G(imgsfeats, imgsfc7, wordclass_zeros, rela_data)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        fake_G, _ = model_convcap_G(imgsfeats, imgsfc7, wordclass_zeros, rela_data)

      fake_G = fake_G[:,:,:-1]               # (100, 9221, 14)
      outcaps_G_real = np.empty((batchsize_cap, 0)).tolist()

      for j in range(max_tokens-1):
        wordact_t3 = fake_G.permute(0, 2, 1).contiguous().view(batchsize_cap*(max_tokens-1), -1)
        wordprobs1 = F.softmax(wordact_t3, dim=1).cpu().data.numpy()
        wordids1 = np.argmax(wordprobs1, axis=1)

        for k in range(batchsize_cap):
          word = wordids1[j+k*(max_tokens-1)]
          outcaps_G_real[k].append(word)

      outcaps_G_real = np.matrix(outcaps_G_real)
      outcaps_G_real = Variable(torch.from_numpy(outcaps_G_real)).cuda()

      word_start1 = Variable(torch.Tensor(batchsize_cap, 1).fill_(50)).cuda()
      outcaps_G_real = outcaps_G_real.float()
      outcaps_G_real1 = torch.cat([word_start1, outcaps_G_real], 1)
      outcaps_G_real1 = outcaps_G_real1.long()


      # input the real to discriminator
      if (args.attention == True):
        wordact_G_real, attn = model_convcap_D(imgsfeats, imgsfc7, outcaps_G_real1, rela_data)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        wordact_G_real, _ = model_convcap_D(imgsfeats, imgsfc7, outcaps_G_real1, rela_data)

      wordact2 = wordact_G_real[:,:,:-1]
      wordclass_change2 = wordclass_v[:, 1:]
      mask1 = mask[:, 1:].contiguous()

      wordact_t3 = wordact2.permute(0, 2, 1).contiguous().view(batchsize_cap * (max_tokens - 1), -1)
      wordclass_g_real = wordclass_change2.contiguous().view(batchsize_cap * (max_tokens - 1), 1)

      maskids = torch.nonzero(mask1.view(-1)).numpy().reshape(-1)

      if (args.attention == True):
        # Cross-entropy loss and attention loss of Show, Attend and Tell
        loss_G = F.cross_entropy(wordact_t3[maskids, ...], wordclass_g_real[maskids, ...].contiguous().view(maskids.shape[0])) \
               + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) \
               / (batchsize_cap * feat_h * feat_w)
      else:
        loss_G = F.cross_entropy(wordact_t3[maskids, ...], wordclass_g_real[maskids, ...].contiguous().view(maskids.shape[0]))

      loss_G.backward()
      ## loss_D_fake.backward(retain_graph=True)
      optimizer_G.step()

      if(img_optimizer):
        img_optimizer.step()

      if (img_optimizer1):
        img_optimizer1.step()


    print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_D))

    modelfn = osp.join(args.model_dir, 'model.pth')

    if(img_optimizer):
      img_optimizer_dict = img_optimizer.state_dict()
    else:
      img_optimizer_dict = None

    if (img_optimizer1):
      img_optimizer1_dict = img_optimizer1.state_dict()
    else:
      img_optimizer1_dict = None

    torch.save({
        'epoch': epoch,
        'state_dict': model_convcap_D.state_dict(),
        'img_state_dict': model_imgcnn.state_dict(),
        'img_state_dict_resnet': model_resnet.state_dict(),
        'optimizer' : optimizer_D.state_dict(),
        'img_optimizer' : img_optimizer_dict,
        'img_optimizer1' : img_optimizer1_dict,
      }, modelfn)

    #Run on validation and obtain score
    scores = test(args, 'val', model_convcap=model_convcap_D, model_imgcnn=model_imgcnn, model_resnet=model_resnet)
    score = scores[0][args.score_select]

    if(score > bestscore):
      bestscore = score
      print('[DEBUG] Saving model at epoch %d with %s score of %f'\
        % (epoch, args.score_select, score))
      bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
      os.system('cp %s %s' % (modelfn, bestmodelfn))
