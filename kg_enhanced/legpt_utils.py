from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from configs import argparser

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from transformers import RobertaTokenizer, RobertaConfig
from get_kg_dict import KG_Dict


MASK_ID = -100


class Entity_Masking_Helper(object):
    """ """
    def __init__(self, tokenizer, kg_dict=None):
        self.tokenizer = tokenizer
        self.kg_dict = kg_dict

    def get_all_words(self, sent):
        words = word_tokenize(sent)
        return words

    def find_indices_naive(self, sent_ids, entity, nltk_words=None):
        # sent_aug = self.tokenizer.decode(sent_ids)
        # tokens = self.tokenizer.tokenize(sent_aug)
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in sent_ids.tolist()]
        # print (tokens)
        # print (self.tokenizer.decode(sent_ids))
        # print (sent_ids)
        # print (self.tokenizer.encode(self.tokenizer.decode(sent_ids))[1:-1])
        # print (len(sent_ids), len(tokens),
        #     len(self.tokenizer.tokenize(self.tokenizer.decode(sent_ids))),
        #     len(self.tokenizer.encode(self.tokenizer.decode(sent_ids))))
        assert len(tokens) == len(sent_ids)

        indices = []
        no_g_tokens = [t[1:] if "Ġ" in t else t for t in tokens]
        
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)+1):
                indices_tmp = list(range(i, j))
                token_tmp = ''.join(no_g_tokens[i:j])
                # print (token_tmp)
                # XXX: ignore cases
                if token_tmp.lower() == entity.lower():
                    indices.append(indices_tmp)

        return indices

    def find_indices(self, sent_ids, entity, nltk_words=None, verbose=False):
        # sent_aug = self.tokenizer.decode(sent_ids)
        # tokens = self.tokenizer.tokenize(sent_aug)
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in sent_ids.tolist()]
        if verbose:
            print (tokens)
        assert len(tokens) == len(sent_ids)

        indices = []
        idx = 0
        word_idx = 0

        idx += 1 # for neglecting <s>
        
        # print (nltk_words)
        while nltk_words:
            nltk_word = nltk_words[0]
            
            indices_tmp = []
            tokens_tmp = []
            for i in range(idx, len(tokens)):
                token_tmp = ''.join(tokens_tmp)
                if token_tmp == nltk_word:
                    nltk_words.pop(0)
                    break
                tkn = tokens[i]
                if tkn in ['<s>', '</s>', '<pad>']:
                    continue
                if "Ġ" in tkn:
                    tkn = tkn[1:]
                indices_tmp.append(i)
                tokens_tmp.append(tkn)
            # print (indices_tmp, i)
            if token_tmp == entity:
                indices.append(indices_tmp)
            idx = i

        return indices

    def find_entities(self, sent, nltk_words, skip_entities=[]):
        if self.kg_dict is None:
            raise ValueError("KG Dict not found!")

        entity_list = []

        for i in range(len(nltk_words)):
            word = nltk_words[i]
            word = word.lower()
            if word in self.kg_dict and word not in skip_entities:
                entity_list.append((word, i))

        return entity_list

    def clean_sentence_for_nltk(self, sent_ids):
        cleaned_sent_ids = []
        if type(sent_ids) != list:
            sent_ids = sent_ids.tolist()
        for idx in sent_ids:
            if idx != 0 and idx != 2:
                cleaned_sent_ids.append(idx)
        return cleaned_sent_ids

    def sample_entities(self, sent_ids, entity_count=1, skip_entities=[]):
        cleaned_sent_ids = self.clean_sentence_for_nltk(sent_ids)
        sent = self.tokenizer.decode(cleaned_sent_ids)
        nltk_words = self.get_all_words(sent)
        entities_list = self.find_entities(sent, nltk_words, skip_entities=skip_entities)

        # TODO: with uniform probability, choose one entity
        if len(entities_list) == 0:
            return []
        e_indices = range(len(entities_list))
        e_sampled_indices = np.random.choice(e_indices, entity_count, replace=False)
        entities = [entities_list[i] for i in e_sampled_indices]
        return entities

    def find_entities_indices(self, sent_ids, entities, verbose=False):
        if verbose:
            print (sent_ids)
        cleaned_sent_ids = self.clean_sentence_for_nltk(sent_ids)
        # print (cleaned_sent_ids)
        sent = self.tokenizer.decode(cleaned_sent_ids)
        sent = ' '.join(sent.split('<pad>'))
        if verbose:
            print (sent)
        nltk_words = self.get_all_words(sent)
        if verbose:
            print (nltk_words)
        entities_indices = []
        for entity in entities:
            if type(entity) == tuple:
                entity = entity[0]
            # entities_indices_tmp = self.find_indices(sent_ids, entity,
            #     nltk_words, verbose=verbose)
            entities_indices_tmp = self.find_indices_naive(sent_ids, entity, nltk_words)
            entities_indices += entities_indices_tmp
        if verbose:
            print (entities_indices)
        entities_indices = np.array(entities_indices)

        return entities_indices


class KG_Based_Masking(object):
    def __init__(self, args, tokenizer, standard_masking_prob=0):
        self.args = args
        self.tokenizer = tokenizer
        self.dummy_dict = {
            "birdy": [("birdy", "is capable of", "flying"), ("birdy", "is a", "bird")],
            "birdyy": [("birdyy", "is capable of", "flying"), ("birdyy", "is a", "bird")],
            "have": [("shiny cherry-red", "capable of", "have"),
                     ("clock keep time we", "not capable of", "have")],
            "hockey": [("hockey", "is capable of", "played"), ("hockey", "is a", "sport"),
                       ("hockey team", "is capable of", "formed"),
                       ("hockey team", "is a", "team"),
                       ("hockey team", "is not", "single man")],
        }
        self.kg_dict = KG_Dict(args)
        # self.kg_dict.d = self.dummy_dict
        self.helper = Entity_Masking_Helper(tokenizer, kg_dict=self.kg_dict.d)
        self.skip_entities = [
            "the",
            "have",
            "a",
        ]
        # self.skip_entities = []
        self.standard_masking_prob = standard_masking_prob
        
    def del_trailing_zeros_for_sent_ids_old(self, sent_ids):
        new_l = []
        if type(sent_ids) != list:
            sent_ids = sent_ids.tolist()
        for i in range(len(sent_ids)):
            idx = sent_ids[i]
            if i > 0 and idx == 0:
                continue
            new_l.append(idx)
        return new_l
        
    def del_trailing_zeros_for_sent_ids(self, sent_ids):
        new_l = sent_ids if type(sent_ids) == list else sent_ids.tolist()
        while new_l[-1] == 0 or new_l[-1] == 1:
            new_l.pop()
        return new_l

    def find_last_before_trailing_zeros(self, sent_ids):
        for i in range(len(sent_ids)):
            idx = sent_ids[i]
            if i > 0 and idx == 0:
                break
        return i

    def augemnt_kg_triplet_to_sent_ids(self, sent_ids, kg_triplets,
            type_ids=None, attn_mask=None, kg_pos=None):
        sent_ids = self.del_trailing_zeros_for_sent_ids(sent_ids)

        # TODO: for token type ids and attention mask
        if type_ids is not None:
            type_ids = type_ids if type(type_ids) == list else type_ids.tolist()
            type_ids = type_ids[:len(sent_ids)]
        if attn_mask is not None:
            attn_mask = attn_mask if type(attn_mask) == list else attn_mask.tolist()
            attn_mask = attn_mask[:len(sent_ids)]

        for kg_triplet in kg_triplets:
            if kg_pos is not None:
                if kg_pos != 0:
                    kg_triplet = "I " + kg_triplet
                    kg_ids = self.tokenizer.encode(kg_triplet)
                    kg_ids = [1] + [kg_ids[0]] + kg_ids[2:]
                else:
                    kg_ids = [1] + self.tokenizer.encode(kg_triplet)
            else:
                kg_ids = [1] + self.tokenizer.encode(kg_triplet)
            sent_ids += kg_ids

            # TODO: for token type ids and attention mask
            if type_ids is not None and attn_mask is not None:
                type_ids += [0] * len(kg_ids)
                attn_mask += [1] * len(kg_ids)
        sent_ids = torch.tensor(sent_ids)
        
        # TODO: for token type ids and attention mask
        type_ids = torch.tensor(type_ids)
        attn_mask = torch.tensor(attn_mask)

        # print (sent_ids)
        # print (self.tokenizer.decode(sent_ids))
        # print (self.tokenizer.tokenize(self.tokenizer.decode(sent_ids)))
        # raise
        if type_ids is not None and attn_mask is not None:
            return sent_ids, type_ids, attn_mask
        return sent_ids

    def masking_augemnt_kg_triplet_to_sent_ids_entity(self, sent_ids,
            entities, kg_triplets, verbose=False):
        if type(kg_triplets) == tuple:
            kg_triplets, kg_pos = kg_triplets
        sent_ids = self.augemnt_kg_triplet_to_sent_ids(
            sent_ids, kg_triplets, kg_pos)
        entities_indices = self.helper.find_entities_indices(sent_ids,
            entities, verbose=verbose)
        return sent_ids, entities_indices

    def masking_augemnt_kg_triplet_to_sent_ids_nokg(self, sent_ids,
            entities, verbose=False):
        entities_indices = self.helper.find_entities_indices(sent_ids,
            entities, verbose=verbose)
        return entities_indices

    def masking_augemnt_kg_triplet_to_sent_ids_rel(self, sent_ids, kg_triplets):
        kg_triplets = kg_triplets[0] if type(kg_triplets) == tuple else kg_triplets
        rel_indices = []
        sent_ids = self.del_trailing_zeros_for_sent_ids(sent_ids)
        curr_start_idx = len(sent_ids)
        for kg_triplet in kg_triplets:
            # get the e1 r e2 ids
            e1, r, e2 = kg_triplet
            r = "I " + r
            len_e1_ids = len(self.tokenizer.encode(e1)) - 2
            len_r_ids = len(self.tokenizer.encode(r)) - 2 - 1 # delete dummy "I"
            # print (len_r_ids, self.tokenizer.tokenize(r))
            len_e2_ids = len(self.tokenizer.encode(e2)) - 2
            rel_idx_lb = curr_start_idx + len_e1_ids + 2 # [1] and <s>
            rel_idx_ub = rel_idx_lb + len_r_ids
            rel_idx = list(range(rel_idx_lb, rel_idx_ub))
            rel_indices.append(rel_idx)
            kg_str = ' '.join(kg_triplet)
            kg_ids = [1] + self.tokenizer.encode(kg_str)
            sent_ids += kg_ids
            curr_start_idx = len(sent_ids)

        # print (rel_indices)
        # print (sent_ids)
        # print (self.tokenizer.decode(sent_ids))
        # print (self.tokenizer.tokenize(self.tokenizer.decode(sent_ids)))
        # raise
        sent_ids = torch.tensor(sent_ids)
        return sent_ids, rel_indices

    def standard_mlm_masking(self, sent_ids):
        label = sent_ids.clone()
        non_zero_entries = sent_ids.nonzero().t()[0][:-1]
        masked_indices_ = torch.bernoulli(torch.full(non_zero_entries.shape,
            self.args.mlm_probability)).bool()
        masked_indices_all_ = torch.full(sent_ids.shape, False)
        start = 1
        masked_indices_all_[start:len(non_zero_entries)+start] = masked_indices_
        zero_entries = torch.full(sent_ids.shape, 1)
        zero_entries[non_zero_entries] = 0
        ele_mult = masked_indices_all_ * zero_entries
        assert torch.sum(ele_mult) == 0
        masked_indices_all_ = masked_indices_all_.bool()
        label[~masked_indices_all_] = MASK_ID
        indices_replaced = torch.bernoulli(torch.full(label.shape, 0.8)).bool() \
            & masked_indices_all_
        sent_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(label.shape, 0.5)).bool() \
            & masked_indices_all_ & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), label.shape, dtype=torch.long)
        sent_ids[indices_random] = random_words[indices_random]
        return sent_ids, label

    def batch_masking_augemnt_kg_triplet_to_sent_ids(self, batch, verbose=False):
        max_len = 0
        new_batch = []
        org_sent = []
        for sent_ids in batch:
            # TODO: for each data ids, check and sample enities
            entities = self.helper.sample_entities(sent_ids, entity_count=1,
                skip_entities=self.skip_entities)
            if verbose:
                print (sent_ids)
                print (self.tokenizer.decode(sent_ids))
                print (entities)

            # TODO: Masking
            which_masking_method = np.random.rand()
            if len(entities) == 0 or which_masking_method < self.standard_masking_prob:
                org_sent.append(sent_ids.clone())
                last_non_zero = self.find_last_before_trailing_zeros(sent_ids)
                # TODO: do the standard masking here
                masked_sent_ids, label = self.standard_mlm_masking(sent_ids)
                new_batch.append((masked_sent_ids, label))
                if verbose:
                    print (masked_sent_ids)
                    print (label)
            else:
                # TODO: sample kg triplets
                # FIXME: currently only sample one entity at a time
                entity = entities[-1][0]
                entity_pos = entities[-1][1]
                entities = [entities[-1]]

                ####
                if self.args.special_masking == 'relation':
                    if_tuple = True
                else:
                    if_tuple = False
                kg_triplets = self.kg_dict.entity_sample(entity,
                    k=self.args.max_num_kg_sample, if_tuple=if_tuple)
                # TODO: for indentifying if the entity is first in a sentence
                kg_triplets = (kg_triplets, entity_pos)
                if verbose:
                    print (entity)
                    print (sent_ids)

                ####
                if self.args.special_masking == 'entity':
                    sent_ids, mask_indices = self.masking_augemnt_kg_triplet_to_sent_ids_entity(
                        sent_ids, entities, kg_triplets, verbose=verbose)
                elif self.args.special_masking == 'relation':
                    sent_ids, mask_indices = self.masking_augemnt_kg_triplet_to_sent_ids_rel(
                        sent_ids, kg_triplets)
                elif self.args.special_masking == 'entity_nokg':
                    mask_indices = self.masking_augemnt_kg_triplet_to_sent_ids_nokg(
                        sent_ids, entities, verbose=verbose)
                else:
                    raise NotImplementedError(
                        "Special Masking Method: {} not found!".format(
                            self.args.special_masking))

                # original sentence
                org_sent.append(sent_ids)
                mask_indices_flat = []
                for mask_indices_ in mask_indices:
                    for mask_idx in mask_indices_:
                        mask_indices_flat.append(mask_idx)
                mask_indices_flat = torch.tensor(mask_indices_flat).long()
                label = torch.full(sent_ids.size(), fill_value=MASK_ID).long()
                try:
                    label[mask_indices_flat] = sent_ids[mask_indices_flat]
                except:
                    print ("mask_indices_flat:", mask_indices_flat)
                    raise
                masked_sent_ids = sent_ids.clone()
                masked_sent_ids[mask_indices_flat] = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.mask_token)
                new_batch.append((masked_sent_ids, label))
                if verbose:
                    print (mask_indices)
                    print (mask_indices_flat)
                pass
            
            curr_len = len(sent_ids)
            max_len = max(curr_len, max_len)
            if verbose:
                print (curr_len)
                print ('-'*50)

        # if <pad>
        if not self.args.kg_augment_with_pad:
            max_len = 0
            for i in range(len(new_batch)):
                input_ids, labels = new_batch[i]
                no_pad_inputs = []
                no_pad_labels = []
                
                for j in range(len(input_ids)):
                    idx = input_ids[j].item()
                    lab_idx = labels[j].item()
                    if idx != 1:
                        no_pad_inputs.append(idx)
                        no_pad_labels.append(lab_idx)
                no_pad_inputs = torch.tensor(no_pad_inputs)
                no_pad_labels = torch.tensor(no_pad_labels)
                new_batch[i] = (no_pad_inputs, no_pad_labels)
                curr_len = len(no_pad_inputs)
                max_len = max(curr_len, max_len)

                # sent
                new_sent = org_sent[i]
                no_pad_sent = []
                for idx in new_sent:
                    if idx != 1:
                        no_pad_sent.append(idx)
                org_sent[i] = no_pad_sent

        # TODO: re-zero-padding again for max len
        if verbose:
            print ("Max Len = {}".format(max_len))
        n_batch = len(batch)
        input_ids = np.full((n_batch, max_len), fill_value=0, dtype=np.int64)
        label = np.full((n_batch, max_len), fill_value=MASK_ID, dtype=np.int64)
        for i in range(len(new_batch)):
            new_data = new_batch[i]
            curr_input_ids, curr_label = new_data
            curr_input_ids = curr_input_ids.numpy()
            curr_label = curr_label.numpy()
            len_data = len(curr_input_ids)
            input_ids[i][:len_data] = curr_input_ids
            label[i][:len_data] = curr_label

        # tensor out and original sentence
        input_ids = torch.tensor(input_ids)
        label = torch.tensor(label)
        if verbose:
            print (input_ids)
            print (label)
        for i in range(len(org_sent)):
            org_sent[i] = self.del_trailing_zeros_for_sent_ids(org_sent[i])

        batch = (input_ids, label, org_sent)
        return batch

    def batch_augemnt_kg_triplet_to_sent_ids(self, batch, type_ids_batch=None,
            attn_mask_batch=None, verbose=False):
        max_len = 0
        new_batch = []

        if type_ids_batch is not None and attn_mask_batch is not None:
            new_type_ids = []
            new_attn_mask = []

        for i in range(len(batch)):
            sent_ids = batch[i]
            if type_ids_batch is not None and attn_mask_batch is not None:
                type_ids = type_ids_batch[i]
                attn_mask = attn_mask_batch[i]
            # TODO: for each data ids, check and sample enities
            entities = self.helper.sample_entities(sent_ids, entity_count=1,
                skip_entities=self.skip_entities)
            if verbose:
                print (sent_ids)
                print (self.tokenizer.decode(sent_ids))
                print (entities)

            # TODO: Augmenting
            if True:
                # TODO: sample kg triplets
                # FIXME: currently only sample one entity at a time
                entity = entities[-1][0]
                entities = [entities[-1]]

                ####
                kg_triplets = self.kg_dict.entity_sample(entity,
                    k=self.args.max_num_kg_sample, if_tuple=False)
                # TODO: for indentifying if the entity is first in a sentence
                if verbose:
                    print (entity)
                    print (sent_ids)

                ####
                if type_ids_batch is not None and attn_mask_batch is not None:    
                    sent_ids, type_ids, attn_mask = self.augemnt_kg_triplet_to_sent_ids(
                        sent_ids, kg_triplets, type_ids, attn_mask, kg_pos=None)
                else:
                    sent_ids = self.augemnt_kg_triplet_to_sent_ids(
                        sent_ids, kg_triplets, kg_pos=None)

                # original sentence
            
            curr_len = len(sent_ids)
            max_len = max(curr_len, max_len)
            new_batch.append(sent_ids)
            if type_ids_batch is not None and attn_mask_batch is not None:
                new_type_ids.append(type_ids)
                new_attn_mask.append(attn_mask)
            if verbose:
                print (curr_len)
                print ('-'*50)

        # if <pad>
        if not self.args.kg_augment_with_pad:
            max_len = 0
            for i in range(len(new_batch)):
                input_ids  = new_batch[i]
                no_pad_inputs = []
                if type_ids_batch is not None and attn_mask_batch is not None:
                    type_ids = new_type_ids[i]
                    attn_mask = new_attn_mask[i]
                    no_pad_type_ids = []
                    no_pad_attn_mask = []
                
                for j in range(len(input_ids)):
                    idx = input_ids[j].item()
                    if idx != 1:
                        no_pad_inputs.append(idx)
                        if type_ids_batch is not None and attn_mask_batch is not None:
                            no_pad_type_ids.append(type_ids[j].item())
                            no_pad_attn_mask.append(attn_mask[j].item())
                no_pad_inputs = torch.tensor(no_pad_inputs)
                new_batch[i] = no_pad_inputs
                if type_ids_batch is not None and attn_mask_batch is not None:
                    no_pad_type_ids = torch.tensor(no_pad_type_ids)
                    no_pad_attn_mask = torch.tensor(no_pad_attn_mask)
                    new_type_ids[i] = no_pad_type_ids
                    new_attn_mask[i] = no_pad_attn_mask
                curr_len = len(no_pad_inputs)
                max_len = max(curr_len, max_len)

        # TODO: re-zero-padding again for max len
        if verbose:
            print ("Max Len = {}".format(max_len))
        n_batch = len(batch)
        input_ids = np.full((n_batch, max_len), fill_value=1, dtype=np.int64)
        if type_ids_batch is not None and attn_mask_batch is not None:
            token_type_ids = np.full((n_batch, max_len), fill_value=0, dtype=np.int64)
            # FIXME: currently don't have to deal with attention since it's always all 1
            attention_mask = np.full((n_batch, max_len), fill_value=1, dtype=np.int64)
        for i in range(len(new_batch)):
            new_data = new_batch[i]
            curr_input_ids = new_data
            curr_input_ids = curr_input_ids.numpy()
            len_data = len(curr_input_ids)
            input_ids[i][:len_data] = curr_input_ids

            if type_ids_batch is not None and attn_mask_batch is not None:
                curr_type_ids = new_type_ids[i]
                curr_type_ids = curr_type_ids.numpy()
                len_type_ids = len(curr_type_ids)
                token_type_ids[i][:len_type_ids] = curr_type_ids

        # tensor out and original sentence
        input_ids = torch.tensor(input_ids)
        if verbose:
            print (input_ids)

        if type_ids_batch is not None and attn_mask_batch is not None:
            token_type_ids = torch.tensor(token_type_ids)
            attention_mask = torch.tensor(attention_mask)
            return input_ids, token_type_ids, attention_mask
        return input_ids


# unit testing
if __name__ == "__main__":
    args = argparser()
    tokenizer = RobertaTokenizer.from_pretrained('large_roberta')
    masker = KG_Based_Masking(args, tokenizer, 0)

    sent1 = "Hockey like the I, and another hockey we have."
    sent2 = "I hate the hockey, and that too."
    sent3 = "I play with him all the time."
    sent_ids1 = tokenizer.encode(sent1)
    sent_ids2 = tokenizer.encode(sent2)
    sent_ids3 = tokenizer.encode(sent3)

    # print (len(sent_ids1))
    # print (len(sent_ids2))
    # print (len(sent_ids3))
    sent_ids1 += [0] * 2
    sent_ids2 += [0] * 5
    sent_ids3 += [0] * 6

    sent_ids1 = np.array(list(sent_ids1))
    sent_ids2 = np.array(list(sent_ids2))
    sent_ids3 = np.array(list(sent_ids3))
    sent_batch = [sent_ids1, sent_ids2, sent_ids3]
    sent_batch = np.asarray(sent_batch)
    # print (sent_batch)
    sent_batch_org = torch.tensor(sent_batch)

    sent_batch = masker.batch_masking_augemnt_kg_triplet_to_sent_ids(sent_batch_org, True)
    print (sent_batch[0][0])
    print (tokenizer.decode(masker.del_trailing_zeros_for_sent_ids(sent_batch[0][0])))
    print (tokenizer.decode(sent_batch[2][0]))
    
    sent_batch = masker.batch_augemnt_kg_triplet_to_sent_ids(sent_batch_org, True)
    print (sent_batch[0])
    print (tokenizer.decode(masker.del_trailing_zeros_for_sent_ids(sent_batch[0])))
