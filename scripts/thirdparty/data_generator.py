from collections import defaultdict
import scripts.thirdparty.constants as constants
import scripts.thirdparty.models as models
import scripts.thirdparty.pre_process as pre_process
from scripts.thirdparty.data_managers import CDRDataManager as data_manager
from scripts.thirdparty.feature_engineering.deptree.parsers import SpacyParser
from scripts.thirdparty.pre_process import opt as pre_opt
from scripts.thirdparty.readers import BioCreativeReader
import pickle
import itertools
import copy
import os
from scripts.thirdparty.feature_engineering.deptree.sdp import Finder
import re
import numpy as np
# from random import shuffle
from sklearn.utils import shuffle

parser = SpacyParser()


def process_one(doc):
    a = list()
    for sent in doc.sentences:
        deptree = parser.parse(sent)
        a.append(deptree)
    return a


def get_candidate(sent, entities):
    """
    :param models.Sentence sent:
    :param list of models.BioEntity entities:
    :return: list of (models.BioEntity, models.BioEntity)
    """
    chem_list = []
    dis_list = []

    min_offset = sent.doc_offset[0]
    max_offset = sent.doc_offset[1]

    for entity in entities:
        try:
            if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                    chem_list.append(entity)
                elif entity.type == constants.ENTITY_TYPE_DISEASE:
                    dis_list.append(entity)
        except:
            print(entity.content)

    return list(itertools.product(chem_list, dis_list))


def main(args=None):
    print('Start')
    pre_config = {
        pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
        pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
    }
    spd_finder = Finder()
    input_path = "data/cdr_data"
    output_path = "data/processed"

    datasets = ['train', 'dev', 'test']
    for dataset in datasets:
        print('Process dataset: ' + dataset)
        reader = BioCreativeReader(os.path.join(input_path, "cdr_" + dataset + ".txt"))
        raw_documents = reader.read()
        raw_entities = reader.read_entity()
        raw_relations = reader.read_relation()

        title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

        # Pre-process
        title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
        abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
        documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)

        # Generate data
        dict_nern = defaultdict(list)
        data_tree = defaultdict()
        for doc in documents:
            raw_entity = raw_entities[doc.id]

            for r_en in raw_entity:
                entity_obj = models.BioEntity(tokens=[], ids={})
                entity_obj.content = r_en[3]
                entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "Chemical" \
                    else constants.ENTITY_TYPE_DISEASE
                entity_obj.ids[constants.MESH_KEY] = r_en[5]

                for s in doc.sentences:
                    if s.doc_offset[0] <= int(r_en[1]) < s.doc_offset[1]:
                        for tok in s.tokens:
                            if (int(r_en[1]) <= tok.doc_offset[0] < int(r_en[2])
                                    or int(r_en[1]) < tok.doc_offset[1] <= int(r_en[2])
                                    or tok.doc_offset[0] <= int(r_en[1]) < int(r_en[2]) <= tok.doc_offset[1]):
                                entity_obj.tokens.append(tok)
                if len(entity_obj.tokens) == 0:
                    print(doc.id, r_en)
                dict_nern[doc.id].append(entity_obj)

            dep_tree = process_one(doc)
            data_tree[doc.id] = dep_tree

        with open(os.path.join(output_path, "sdp_data_acentors." + dataset + ".txt"), "w") as f:
            for doc in shuffle(sorted(documents, key=lambda x: x.id)):
                sdp_data = defaultdict(dict)
                deep_tree_doc = data_tree[doc.id]
                relation = raw_relations[doc.id]
                f.write(doc.id)
                f.write("\n")
                doc_sib = []
                for sent, deptree in zip(doc.sentences, deep_tree_doc):
                    sent_offset2idx = {}
                    for idx, token in enumerate(sent.tokens):
                        sent_offset2idx[token.sent_offset] = idx
                    pairs = get_candidate(sent, dict_nern[doc.id])
                    if len(pairs) == 0:
                        continue

                    for pair in pairs:
                        chem_entity = pair[0]
                        dis_entity = pair[1]

                        chem_token = chem_entity.tokens[-1]
                        dis_token = dis_entity.tokens[-1]

                        r_path = spd_finder.find_sdp(deptree, chem_token, dis_token)

                        new_r_path = copy.deepcopy(r_path)
                        for i, x in enumerate(new_r_path):
                            if i % 2 == 0:
                                x.content += "_" + str(sent_offset2idx[x.sent_offset])

                        path = spd_finder.parse_directed_sdp(new_r_path)

                        sent_path = '|'.join([token.content for token in sent.tokens])

                        if path:
                            chem_ids = chem_entity.ids[constants.MESH_KEY].split('|')
                            dis_ids = dis_entity.ids[constants.MESH_KEY].split('|')
                            rel = 'CID'
                            for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                if (doc.id, 'CID', chem_id, dis_id) not in relation:
                                    rel = 'NONE'
                                    break

                            for chem_id, dis_id in itertools.product(chem_ids, dis_ids):
                                key = '{}_{}'.format(chem_id, dis_id)

                                if rel not in sdp_data[key]:
                                    sdp_data[key][rel] = []

                                sdp_data[key][rel].append([path, sent_path])

                for pair_key in sdp_data:
                    if 'CID' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['CID'])):
                            sdp, sent_path = sdp_data[pair_key]['CID'][k]
                            f.write('{} {} {}\n'.format(pair_key, 'CID', sdp))

                    if 'NONE' in sdp_data[pair_key]:
                        for k in range(len(sdp_data[pair_key]['NONE'])):
                            sdp, sent_path = sdp_data[pair_key]['NONE'][k]
                            f.write('{} {} {}\n'.format(pair_key, 'NONE', sdp))
