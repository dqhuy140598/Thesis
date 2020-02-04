import itertools
import networkx as nx
from scipy.spatial import distance

import constants
from feature_engineering.deptree.models import DepTree
from feature_engineering.deptree.parsers import SpacyParser
from feature_engineering.word_embedding.nlplab import NLPLab
from module import wordnet

spp = SpacyParser()


def get_word_embedding_connect(document, threshold=1.5, step=1):
    """
    get token pair that  of word embedding
    :param int step:
    :param float threshold:
    :param models.Document document:
    :return:
    """
    pairs = []
    for i in range(len(document.sentences) - step):
        for tok1 in document.sentences[i].tokens:
            if tok1.is_stop_word() or tok1.is_symbol() or tok1.is_number():
                continue
            else:
                for tok2 in document.sentences[i + step].tokens:
                    if tok2.is_stop_word() or tok2.is_symbol() or tok2.is_number():
                        continue
                    else:
                        v1 = NLPLab.word2vec(tok1.content)
                        v2 = NLPLab.word2vec(tok2.content)

                        if not v1.any() or not v2.any():
                            continue
                        else:
                            o = distance.euclidean(v1, v2)

                            if 0.0 <= o <= threshold:
                                # print('distance {} to {} is {}'.format(tok1.content, tok2.content, o))
                                pairs.append((tok1, tok2))

    return pairs


def get_wordnet_synonym_connect(document, step=1):
    """

    :param int step:
    :param models.Document document:
    :return:
    """
    pairs = []
    for i in range(len(document.sentences) - step):
        for tok1 in document.sentences[i].tokens:
            if tok1.is_stop_word() or tok1.is_symbol() or tok1.is_number():
                continue
            else:
                tok1_synonyms = wordnet.synonym(tok1)

                for tok2 in document.sentences[i + step].tokens:
                    if tok2.is_stop_word() or tok2.is_symbol() or tok2.is_number():
                        continue
                    else:
                        if tok2.content.lower() != tok1.content.lower() and tok2.content.lower() in tok1_synonyms:
                            # print('synonym {} and {}'.format(tok1.content, tok2.content))
                            pairs.append((tok1, tok2))

    return pairs


def get_matched_id_entity_connect(entities):
    """

    :param models.Entities entities:
    :return:
    """
    pairs = []

    for pair in itertools.permutations(entities, 2):
        ent1 = pair[0]
        ent2 = pair[1]

        entity1_id = ent1.ids[constants.MESH_KEY]
        entity2_id = ent2.ids[constants.MESH_KEY]

        if entity1_id == entity2_id:
            # print('same id {} and {}'.format(ent1.content, ent2.content))
            pairs.append((ent1.tokens[-1], ent2.tokens[-1]))

    return pairs


def get_next_sentence_reference_connect(document, entities, step=1):
    """

    :param entities:
    :param int step:
    :param models.Document document:
    :return:
    """
    pairs = []
    for i in range(len(document.sentences) - step):
        # get node of entities
        min_offset = document.sentences[i].doc_offset[0]
        max_offset = document.sentences[i].doc_offset[1]

        chem_list = []
        dis_list = []
        for entity in entities:
            if min_offset <= entity.tokens[0].doc_offset[0] <= max_offset:
                if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                    chem_list.append(entity)
                elif entity.type == constants.ENTITY_TYPE_DISEASE:
                    dis_list.append(entity)

        if len(chem_list) == 0 and len(dis_list) == 0:
            continue
        else:
            for tok in document.sentences[i + step].tokens:
                if len(dis_list) != 0 and tok.is_disease_mention():
                    for ent in dis_list:
                        # print('Disease next sentence {} and {}'.format(ent.content, tok.content))
                        pairs.append((ent.tokens[-1], tok))
                elif len(chem_list) != 0 and tok.is_chemical_mention():
                    for ent in chem_list:
                        # print('Chemical next sentence {} and {}'.format(ent.content, tok.content))
                        pairs.append((ent.tokens[-1], tok))

    return pairs


def build_sentence_dependency_deptree(sentence, normalize=True, merge_entity=True):
    """
    build the DepTree for a sentence
    :param normalize:
    :param models.Sentence sentence:
    :return DepTree:
    """
    deptree_edges = spp.parse(sentence)
    if normalize:
        deptree_edges = spp.normalize_deptree(deptree_edges)

    # print('content', sentence.content)

    return DepTree(edges=deptree_edges)


def build_sentence_dependency_graph(sentence, core_node=None):
    """
    build the graph for a sentence
    :param list of (str, (int, int), (int, int)) core_node:
    :param models.Sentence sentence:
    :return DepTree:
    """
    deptree = build_sentence_dependency_deptree(sentence, normalize=False)
    g = nx.DiGraph()

    if not core_node:
        g.add_node(*deptree.get_root(data=True))
    else:
        for node in core_node:
            nearest_verb = deptree.find_nearest_verb(node)
            root_path_graph = deptree.find_root_tree(nearest_verb)
            subtree_graph = deptree.get_sub_tree(nearest_verb)

            g = nx.compose_all([g, root_path_graph.tree, subtree_graph.tree])

    return DepTree(tree=g)
