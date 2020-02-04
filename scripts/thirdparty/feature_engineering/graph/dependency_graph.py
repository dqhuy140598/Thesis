import itertools
import networkx as nx
from collections import defaultdict

import constants
from module import plotlib
from networkx.drawing.nx_agraph import graphviz_layout
from feature_engineering.graph import helper
from feature_engineering.deptree import opt
from module.spacy import Spacy


class DependencyGraph:
    def __init__(self, document, entities=None):
        """
        build the dependency graph for the document
        :param models.Document document:
        :param list of models.Entity entities:
        :return:
        """
        self.document = document
        self.entities = entities

        self.graph = nx.MultiDiGraph()
        self.root_map = {}
        self.sentence_entities = []
        for sentence in document.sentences:
            if len(sentence.tokens) == 1:
                token = sentence.tokens[0]
                r = Spacy.parse(token.content)
                token.metadata['pos_tag'] = r[0].tag_

                g = nx.MultiDiGraph()
                node = token.get_node()
                g.add_node(node, pos_tag=token.metadata['pos_tag'], label=token.content, token=token)
                self.graph = nx.compose(self.graph, g)
                self.root_map[sentence.doc_offset] = node

                min_offset = sentence.doc_offset[0]
                max_offset = sentence.doc_offset[1]

                sentence_entity = []
                for entity in self.entities:
                    if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                        sentence_entity.append(entity)
                self.sentence_entities.append(sentence_entity)
            else:
                core_node = []

                # get node of entities
                min_offset = sentence.doc_offset[0]
                max_offset = sentence.doc_offset[1]
                sentence_entity = []
                for entity in self.entities:
                    if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                        core_node.append(entity.tokens[-1].get_node())
                        sentence_entity.append(entity)
                self.sentence_entities.append(sentence_entity)

                # get node of pronouns
                pronoun_toks = sentence.get_pronoun()
                core_node.extend([tok.get_node() for tok in pronoun_toks])

                # get node of chemical mentions
                chem_toks = sentence.get_chemical_mention()
                core_node.extend([tok.get_node() for tok in chem_toks])

                # get node of disease mentions
                dis_toks = sentence.get_disease_mention()
                core_node.extend([tok.get_node() for tok in dis_toks])

                core_node = list(set(core_node))
                sentence_graph = helper.build_sentence_dependency_graph(sentence, core_node=core_node)
                sentence_graph.clean_leaf()
                self.graph = nx.compose(self.graph, nx.MultiDiGraph(sentence_graph.tree))
                self.root_map[sentence.doc_offset] = sentence_graph.get_root()

        self.edge_list = {opt.DEPENDENCY_EDGE: self.graph.edges()}

    def connect_root(self):
        root_list = [self.root_map[k] for k in self.root_map]
        self.edge_list[opt.ROOT_EDGE] = []
        for i in range(len(root_list) - 1):
            self.graph.add_edge(root_list[i], root_list[i + 1], type=opt.ROOT_EDGE, relation=opt.ROOT_EDGE, weight=1.0)
            self.edge_list[opt.ROOT_EDGE].append((root_list[i], root_list[i + 1]))

    def connect_word_embedding_similarity(self, window_size=1):
        self.edge_list[opt.WORD_EMBEDDING_EDGE] = []

        for i in range(1, window_size + 1):
            pairs = helper.get_word_embedding_connect(self.document, step=i)
            for pair in pairs:
                if not self.graph.has_node(pair[0].get_node()) or not self.graph.has_node(pair[1].get_node()):
                    continue
                else:
                    self.graph.add_edge(pair[0].get_node(), pair[1].get_node(), type=opt.WORD_EMBEDDING_EDGE, relation=opt.WORD_EMBEDDING_EDGE, weight=1.0)
                    self.edge_list[opt.WORD_EMBEDDING_EDGE].append((pair[0].get_node(), pair[1].get_node()))

    def connect_wordnet_synonym(self, window_size=1):
        self.edge_list[opt.WORDNET_SYNONYM_EDGE] = []

        for i in range(1, window_size + 1):
            pairs = helper.get_wordnet_synonym_connect(self.document, step=i)
            for pair in pairs:
                if not self.graph.has_node(pair[0].get_node()) or not self.graph.has_node(pair[1].get_node()):
                    continue
                else:
                    self.graph.add_edge(pair[0].get_node(), pair[1].get_node(), type=opt.WORDNET_SYNONYM_EDGE, relation=opt.WORDNET_SYNONYM_EDGE, weight=1.0)
                    self.edge_list[opt.WORDNET_SYNONYM_EDGE].append((pair[0].get_node(), pair[1].get_node()))

    def connect_co_reference(self, window_size=1):
        self.edge_list[opt.CO_REFERENCE_EDGE] = []

        matching_id_pairs = helper.get_matched_id_entity_connect(self.entities)
        for pair in matching_id_pairs:
            if not self.graph.has_node(pair[0].get_node()) or not self.graph.has_node(pair[1].get_node()):
                continue
            else:
                self.graph.add_edge(pair[0].get_node(), pair[1].get_node(), type=opt.CO_REFERENCE_EDGE, relation=opt.CO_REFERENCE_EDGE, weight=1.0)
                self.edge_list[opt.CO_REFERENCE_EDGE].append((pair[0].get_node(), pair[1].get_node()))

        for i in range(1, window_size + 1):
            pairs = helper.get_next_sentence_reference_connect(self.document, self.entities, step=i)
            for pair in pairs:
                if not self.graph.has_node(pair[0].get_node()) or not self.graph.has_node(pair[1].get_node()):
                    continue
                else:
                    self.graph.add_edge(pair[0].get_node(), pair[1].get_node(), type=opt.CO_REFERENCE_EDGE, relation=opt.CO_REFERENCE_EDGE, weight=1.0)
                    self.edge_list[opt.CO_REFERENCE_EDGE].append((pair[0].get_node(), pair[1].get_node()))

    def get_all_shortest_path(self, step=0):
        ret = defaultdict(list)

        for i in range(len(self.document.sentences) - step):
            sent1_entities = self.sentence_entities[i]
            sent2_entities = self.sentence_entities[i + step]

            for e1, e2 in itertools.product(sent1_entities, sent2_entities):
                if e1.type == constants.ENTITY_TYPE_CHEMICAL and e2.type == constants.ENTITY_TYPE_DISEASE:
                    key = '{}_{}'.format(e1.ids[constants.MESH_KEY], e2.ids[constants.MESH_KEY])
                    ret[key].extend(self.get_shortest_path(e1.tokens[-1].get_node(), e2.tokens[-1].get_node()))

                # reverse the direction from sent2 to sent1
                # if step == 0 mean intra-sentence => skip this step, cuz sent1 == sent2
                if step != 0 and e2.type == constants.ENTITY_TYPE_CHEMICAL and e1.type == constants.ENTITY_TYPE_DISEASE:
                    key = '{}_{}'.format(e2.ids[constants.MESH_KEY], e1.ids[constants.MESH_KEY])
                    ret[key].extend(self.get_shortest_path(e2.tokens[-1].get_node(), e1.tokens[-1].get_node()))

        return ret

    def get_shortest_path(self, source, target):
        raw_paths = nx.all_shortest_paths(self.graph.to_undirected(), source, target)
        ret = []
        for raw_path in raw_paths:
            sdp = []
            for i in range(len(raw_path) - 1):
                sdp.append(self.graph.node[raw_path[i]]['token'])
                sdp.append(self.get_min_edge_data(raw_path[i], raw_path[i + 1]))

            sdp.append(self.graph.node[raw_path[-1]]['token'])

            ret.append(sdp)

        return ret

    def get_min_edge_data(self, source, target):
        ret = ('', '', float('inf'))

        data_fw = self.graph.get_edge_data(source, target)
        if data_fw:
            for k in data_fw:
                data = data_fw[k]
                if data['weight'] < ret[2]:
                    ret = (data['relation'], 'r', data['weight'])

        data_bw = self.graph.get_edge_data(target, source)
        if data_bw:
            for k in data_bw:
                data = data_bw[k]
                if data['weight'] < ret[2]:
                    ret = (data['relation'], 'l', data['weight'])

        return ret

    def visualize(self):
        """
        :return:
        """
        plotlib.next_plot()

        node_labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx(
            self.graph,
            labels=node_labels,
            edgelist=[],
            pos=graphviz_layout(self.graph, prog='dot'),
            node_size=300,
            font_weight='bold',
            font_color='xkcd:red',
            node_color='xkcd:light grey'
        )
        for edge_key in self.edge_list:
            edge_list = self.edge_list[edge_key]
            edge_color = opt.EDGE_COLOR[edge_key]
            style = 'solid' if edge_key != opt.DEPENDENCY_EDGE else 'dotted'
            nx.draw_networkx_edges(
                self.graph,
                pos=graphviz_layout(self.graph, prog='dot'),
                edgelist=edge_list,
                style=style,
                edge_color=edge_color
            )
