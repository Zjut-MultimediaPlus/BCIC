import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json
import h5py
import random
import argparse
#from models.ass_fun import *
import random
from PIL import Image
import multiprocessing
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable



class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img):
    return img.resize((self.size[1], self.size[0]), self.interpolation)

class gcn_loader(Dataset):
    """Loads train/val/test splits of coco dataset"""
    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split==self.split)
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size   ## opt.vocab_size = loader.vocab_size  train_mem.py line37

    def get_rela_dict_size(self):
        return self.rela_dict_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length


    def __init__(self, split='train', max_tokens=15, seq_per_img=5):
        self.max_tokens = max_tokens
        self.seq_per_img = seq_per_img
        #self.coco_root = coco_root
        self.split = split

        self.use_att = True
        #self.use_box = False
        self.use_rela = True
        self.norm_att_feat = False
        self.norm_box_feat = False
        self.train_only = 0
        self.batch_size = 20
        #self.seq_per_img = 5

        self.input_json = 'data/cocobu2.json'
        self.input_fc_dir = 'data/cocobu_fc'
        self.input_att_dir = 'data/cocobu_att'
        self.input_label_h5 = 'data/cocobu2_label.h5'
        self.input_rela_dir = 'data/coco_img_sg'
        self.sg_dict_path= 'data/spice_sg_dict2.npz'
        self.rela_dict_dir = 'data/rela_dict.npy'
        self.input_ssg_dir = 'data/coco_spice_sg2'

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', self.input_json)
        self.info = json.load(open(self.input_json))

        print('using new dict')
        if self.sg_dict_path == 'data/spice_sg_dict2.npz':
            sg_dict_info = np.load(self.sg_dict_path)['spice_dict'][()]
            self.ix_to_word = sg_dict_info['ix_to_word']
        else:
            sg_dict_info = np.load(self.sg_dict_path)['sg_dict'][()]
            self.ix_to_word = sg_dict_info['sg_ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        if self.use_rela:
            self.rela_dict_dir = self.rela_dict_dir
            rela_dict_info = np.load(self.rela_dict_dir)
            rela_dict = rela_dict_info[()]['rela_dict']
            self.rela_dict_size = len(rela_dict)
            print('rela dict size is {0}'.format(self.rela_dict_size))


        # open the hdf5 file
        print('DataLoader loading h5 file: ', self.input_fc_dir, self.input_att_dir, self.input_label_h5)
        self.h5_label_file = h5py.File(self.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.input_fc_dir
        self.input_att_dir = self.input_att_dir
        self.input_rela_dir = self.input_rela_dir


        # load in the sequence data                        ## self.h5_label_file = opt.input_lable_h5
        seq_size = self.h5_label_file['labels'].shape      ##
        print("seq_size:{0}".format(seq_size))             ##
        self.seq_length = seq_size[1]                      ##
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]     ##
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'train_sg': [], 'val_sg': [], 'test_sg': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                #self.split_ix['train'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif self.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))
        print('assigned %d images to split train_sg' % len(self.split_ix['train_sg']))
        print('assigned %d images to split val_sg' % len(self.split_ix['val_sg']))
        print('assigned %d images to split test_sg' % len(self.split_ix['test_sg']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'train_sg':0, 'val_sg': 0, 'test_sg': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)


    def get_captions(self, ix, seq_per_img):    ## seq_per_img  default = 5
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:      ## 
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)    ## 
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
                    ## self.h5_label_file = opt.input_label_h5
        return seq

    def get_batch(self, split, batch_size=None, seq_per_img= None):
        batch_size = batch_size or self.batch_size         ## batch_size = 50
        seq_per_img = seq_per_img or self.seq_per_img      ## seq_per_img = 5

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
                                                                    ## opt.fc_feat_size = 2048
                                                                    ## opt.att_feat_size = 2048
        rela_rela_batch = []        ## if self.use_rela
        rela_attr_batch = []

        '''
        ssg_rela_batch = []
        ssg_obj_batch = []
        ssg_attr_batch = []
        '''

                                ## (50x5, seq_length+2)
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):    ## 50
            # fetch image
            tmp_fc, tmp_att, tmp_rela,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            #  captions, wordclass, sentence_mask, img_id, tmp_ssg,
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att) 

            ## relationship + attribute
            rela_rela_batch.append(tmp_rela['rela_rela_matrix'])
            rela_attr_batch.append(tmp_rela['rela_attr_matrix'])



            ## get captions
            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        #                                           
        #  zip() ->,   sorted() -> ,   key = lambda x:0 -> ,   reverse = True -> 
        fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: 0, reverse=True))
        data = {}                # np.vsplit()   label_batch -> batch_size * seq_per_img  seq_per_img
                                                                                # label_batch -> (50 * 5, 15+2)
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*seq_per_img for _ in fc_batch]))  # reduce(lambda x,y:x+y, ...)  sum
                        ## np.stack(a, axis=0)  
        ## max_att_len -> att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])  ##
                                            ## att_batch -> (50 *5, 14, 14, 2048)
        # merge att_feats           # len()                # 
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):      ## len(att_batch) -> batch_size * seq_per_img
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        ## --use_rela 0  ->  If use rela information
        if self.use_rela:
            max_rela_len = max([_.shape[0] for _ in rela_rela_batch])
            data['rela_rela_matrix'] = np.zeros([len(att_batch)*seq_per_img, max_rela_len, 3])
            for i in range(len(rela_rela_batch)):
                data['rela_rela_matrix'][i*seq_per_img:(i+1)*seq_per_img,0:len(rela_rela_batch[i]),:] = rela_rela_batch[i]
            data['rela_rela_masks'] = np.zeros(data['rela_rela_matrix'].shape[:2], dtype='float32')
            for i in range(len(rela_rela_batch)):
                data['rela_rela_masks'][i*seq_per_img:(i+1)*seq_per_img,:rela_rela_batch[i].shape[0]] = 1

            max_attr_obj_len = max(_.shape[0] for _ in rela_attr_batch)
            max_attr_each_len = max(_.shape[1] for _ in rela_attr_batch)
            data['rela_attr_masks'] = np.zeros([len(att_batch)*seq_per_img, max_attr_obj_len, max_attr_each_len], dtype='float32')
            data['rela_attr_matrix'] = np.zeros([len(att_batch)*seq_per_img, max_attr_obj_len, max_attr_each_len], dtype='float32')
            for i in range(len(rela_attr_batch)):
                data['rela_attr_matrix'][i*seq_per_img:(i+1)*seq_per_img,0:len(rela_attr_batch[i]),:rela_attr_batch[i].shape[1]] = \
                    rela_attr_batch[i]
            for i in range(len(rela_attr_batch)):
                attr_obj_len = rela_attr_batch[i].shape[0]
                for j in range(attr_obj_len):
                    attr_each_len = np.sum(rela_attr_batch[i][j,:]>=0)
                    data['rela_attr_masks'][i*seq_per_img:(i+1)*seq_per_img,j,:attr_each_len] = 1
     
        ## rela = None
        else:
            data['rela_rela_matrix'] = None
            data['rela_rela_masks'] = None
            data['rela_attr_matrix'] = None
            data['rela_attr_masks'] = None



        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data    

    def __getitem__(self, index):
        #img_id = self.ids[index]
        #print(img_id)
        ix = index
        #img_id = self.split_ix[idx]

        rela_data = {}
        rela_data['rela_rela_matrix'] = []
        rela_data['rela_attr_matrix'] = []


        ## use_att = True
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])

            if self.use_rela:
                path_temp = os.path.join(self.input_rela_dir, str(self.info['images'][ix]['id']) + '.npy')
                if os.path.isfile(path_temp):
                    rela_info = np.load(os.path.join(path_temp))
                    rela_data['rela_rela_matrix'] = rela_info[()]['rela_matrix']
                    rela_data['rela_attr_matrix'] = rela_info[()]['obj_attr']
                else:
                    #if we do not have rela_matrix, this matrix is set to be [0,3] zero matrix
                    rela_data = {}
                    rela_data['rela_rela_matrix'] = []
                    rela_data['rela_attr_matrix'] = []
            else:
                rela_data = {}
                rela_data['rela_rela_matrix'] = []
                rela_data['rela_attr_matrix'] = []


        else:
            att_feat = np.zeros((1,1,1))


        #return img, captions, wordclass, sentence_mask, img_id 
        return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
                att_feat,
                rela_data,
                ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[3] == ix, "ix not equal"

        return tmp + [wrapped]


class coco_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, coco_root, split='train', max_tokens=15, ncap_per_img=5):
    self.max_tokens = max_tokens
    self.ncap_per_img = ncap_per_img
    self.coco_root = coco_root
    self.split = split
    self.input_rela_dir = 'data/coco_img_sg'
    self.input_json = 'data/cocobu2.json'
    self.info = json.load(open(self.input_json))
    #Splits from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    self.get_split_info('data/dataset_coco.json')

    worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
    wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
    self.wordlist = ['EOS'] + sorted(wordlist)
    self.numwords = len(self.wordlist)
    print('[DEBUG] #words in wordlist: %d' % (self.numwords))

    self.img_transforms = transforms.Compose([
      Scale([224, 224]),
      transforms.ToTensor(),
      transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        std = [ 0.229, 0.224, 0.225 ])
      ])

  def get_split_info(self, split_file):
    print('Loading annotation file...')
    with open(split_file) as fin:
      split_info = json.load(fin)
    annos = {}
    for item in split_info['images']:
      if self.split == 'train':
        if item['split'] == 'train' or item['split'] == 'restval':
          annos[item['cocoid']] = item
      elif item['split'] == self.split:
        annos[item['cocoid']] = item
    self.annos = annos
    self.ids = list(self.annos.keys())
    print('Found %d images in split: %s'%(len(self.ids), self.split))

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    anno = self.annos[img_id]

    captions = [caption['raw'] for caption in anno['sentences']]

    imgpath = '%s/%s/%s'%(self.coco_root, anno['filepath'], anno['filename'])
    img = Image.open(os.path.join(imgpath)).convert('RGB')
    img = self.img_transforms(img)

    if(self.split != 'train'):
      r = np.random.randint(0, len(captions))
      captions = [captions[r]]

    if(self.split == 'train'):
      if(len(captions) > self.ncap_per_img):
        ids = np.random.permutation(len(captions))[:self.ncap_per_img]
        captions_sel = [captions[l] for l in ids]
        captions = captions_sel
      assert(len(captions) == self.ncap_per_img)

    wordclass = torch.LongTensor(len(captions), self.max_tokens).zero_()
    sentence_mask = torch.ByteTensor(len(captions), self.max_tokens).zero_()

    for i, caption in enumerate(captions):
      words = str(caption).lower().translate(None, string.punctuation).strip().split()
      words = ['<S>'] + words
      num_words = min(len(words), self.max_tokens-1)
      sentence_mask[i, :(num_words+1)] = 1
      for word_i, word in enumerate(words):
        if(word_i >= num_words):
          break
        if(word not in self.wordlist):
          word = 'UNK'
        wordclass[i, word_i] = self.wordlist.index(word)

    # path_temp = os.path.join('./data/coco_pred_sg', str(img_id) + '.npy')
    # rela_info = np.load(os.path.join(path_temp))
    #
    # rela_rela_matrix = rela_info[()]['rela_matrix']
    # rela_attr_matrix = rela_info[()]['obj_attr']

    # rela_data = {}
    # rela_data['rela_rela_matrix'] = []
    # rela_data['rela_attr_matrix'] = []
    #
    # path_temp = os.path.join('./data/coco_pred_sg', str(img_id) + '.npy')
    # rela_info = np.load(os.path.join(path_temp))
    # rela_data['rela_rela_matrix'] = rela_info[()]['rela_matrix']
    # rela_data['rela_attr_matrix'] = rela_info[()]['obj_attr']


    #rela_rela_matrix = rela_rela_matrix


    return img, captions, wordclass, sentence_mask, img_id

  def __len__(self):
    return len(self.ids)