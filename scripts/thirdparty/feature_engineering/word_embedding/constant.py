DEFAULT_SENTENCE_LENGTH = 60

WORD_DEFAULT_EMBEDDING_SIZE = 200
REL_DEFAULT_EMBEDDING_SIZE = 200
TAG_DEFAULT_EMBEDDING_SIZE = 0

# GLOROT_RANDOM_SEED = 5
GLOROT_UNIFORM_LIM = 0.224
GLOROT_NORMAL_MEAN = 0
GLOROT_NORMAL_SIGMA = 0.15

# GLOROT_RANDOM_SEED = np.random.randint(-10 * 9, 10 ** 9)
GLOROT_RANDOM_SEED = 466625930
print('GLOROT_RANDOM_SEED: {}'.format(GLOROT_RANDOM_SEED))

RELATION_2_VEC_METHOD_ONE_HOT = 0
RELATION_2_VEC_METHOD_GLOROT_UNIFORM = 1
RELATION_2_VEC_METHOD_GLOROT_NORMAL = 2
RELATION_2_VEC_METHOD_PRETRAIN = 3

TAG_2_VEC_METHOD_ONE_HOT = 0
TAG_2_VEC_METHOD_GLOROT_UNIFORM = 1
TAG_2_VEC_METHOD_GLOROT_NORMAL = 2
TAG_2_VEC_METHOD_PRETRAIN = 3

WORD_2_VEC_METHOD_NLPlAB = 0

WORD_OPTION_SIZE_KEY = 'size'
WORD_OPTION_METHOD_KEY = 'method'

TAG_OPTION_SIZE_KEY = 'size'
TAG_OPTION_METHOD_KEY = 'method'

REL_OPTION_METHOD_KEY = 'method'

PRETRAIN_RELATION_EMBEDDING_FILE = ''

REL_POS = {
    '(acomp)': 0, '(l_acomp)': 1, '(r_acomp)': 2, '(advcl)': 3, '(l_advcl)': 4, '(r_advcl)': 5, '(advmod)': 6,
    '(l_advmod)': 7, '(r_advmod)': 8, '(agent)': 9, '(l_agent)': 10, '(r_agent)': 11, '(amod)': 12, '(l_amod)': 13,
    '(r_amod)': 14, '(appos)': 15, '(l_appos)': 16, '(r_appos)': 17, '(attr)': 18, '(l_attr)': 19, '(r_attr)': 20,
    '(aux)': 21, '(l_aux)': 22, '(r_aux)': 23, '(auxpass)': 24, '(l_auxpass)': 25, '(r_auxpass)': 26, '(cc)': 27,
    '(l_cc)': 28, '(r_cc)': 29, '(ccomp)': 30, '(l_ccomp)': 31, '(r_ccomp)': 32, '(complm)': 33, '(l_complm)': 34,
    '(r_complm)': 35, '(conj)': 36, '(l_conj)': 37, '(r_conj)': 38, '(cop)': 39, '(l_cop)': 40, '(r_cop)': 41,
    '(csubj)': 42, '(l_csubj)': 43, '(r_csubj)': 44, '(csubjpass)': 45, '(l_csubjpass)': 46, '(r_csubjpass)': 47,
    '(dep)': 48, '(l_dep)': 49, '(r_dep)': 50, '(det)': 51, '(l_det)': 52, '(r_det)': 53, '(dobj)': 54, '(l_dobj)': 55,
    '(r_dobj)': 56, '(expl)': 57, '(l_expl)': 58, '(r_expl)': 59, '(hmod)': 60, '(l_hmod)': 61, '(r_hmod)': 62,
    '(hyph)': 63, '(l_hyph)': 64, '(r_hyph)': 65, '(infmod)': 66, '(l_infmod)': 67, '(r_infmod)': 68, '(intj)': 69,
    '(l_intj)': 70, '(r_intj)': 71, '(iobj)': 72, '(l_iobj)': 73, '(r_iobj)': 74, '(mark)': 75, '(l_mark)': 76,
    '(r_mark)': 77, '(meta)': 78, '(l_meta)': 79, '(r_meta)': 80, '(neg)': 81, '(l_neg)': 82, '(r_neg)': 83,
    '(nmod)': 84, '(l_nmod)': 85, '(r_nmod)': 86, '(nn)': 87, '(l_nn)': 88, '(r_nn)': 89, '(npadvmod)': 90,
    '(l_npadvmod)': 91, '(r_npadvmod)': 92, '(nsubj)': 93, '(l_nsubj)': 94, '(r_nsubj)': 95, '(nsubjpass)': 96,
    '(l_nsubjpass)': 97, '(r_nsubjpass)': 98, '(num)': 99, '(l_num)': 100, '(r_num)': 101, '(number)': 102,
    '(l_number)': 103, '(r_number)': 104, '(oprd)': 105, '(l_oprd)': 106, '(r_oprd)': 107, '(obj)': 108, '(l_obj)': 109,
    '(r_obj)': 110, '(obl)': 111, '(l_obl)': 112, '(r_obl)': 113, '(parataxis)': 114, '(l_parataxis)': 115,
    '(r_parataxis)': 116, '(partmod)': 117, '(l_partmod)': 118, '(r_partmod)': 119, '(pcomp)': 120, '(l_pcomp)': 121,
    '(r_pcomp)': 122, '(pobj)': 123, '(l_pobj)': 124, '(r_pobj)': 125, '(poss)': 126, '(l_poss)': 127, '(r_poss)': 128,
    '(possessive)': 129, '(l_possessive)': 130, '(r_possessive)': 131, '(preconj)': 132, '(l_preconj)': 133,
    '(r_preconj)': 134, '(prep)': 135, '(l_prep)': 136, '(r_prep)': 137, '(prt)': 138, '(l_prt)': 139, '(r_prt)': 140,
    '(punct)': 141, '(l_punct)': 142, '(r_punct)': 143, '(quantmod)': 144, '(l_quantmod)': 145, '(r_quantmod)': 146,
    '(rcmod)': 147, '(l_rcmod)': 148, '(r_rcmod)': 149, '(root)': 150, '(l_root)': 151, '(r_root)': 152, '(xcomp)': 153,
    '(l_xcomp)': 154, '(r_xcomp)': 155, '()': 156, '(l_)': 157, '(r_)': 158, '(compound)': 159, '(l_compound)': 160,
    '(r_compound)': 161, '(acl)': 162, '(l_acl)': 163, '(r_acl)': 164, '(relcl)': 165, '(l_relcl)': 166, '(r_relcl)': 167,
    '(nummod)': 168, '(l_nummod)': 169, '(r_nummod)': 170, '(case)': 171, '(l_case)': 172, '(r_case)': 173, '(dative)': 174,
    '(l_dative)': 175, '(r_dative)': 176, '(predet)': 177, '(l_predet)': 178, '(r_predet)': 179,
    '(word_embedding)': 180, '(l_word_embedding)': 181, '(r_word_embedding)': 182,
    '(next_root)': 183, '(l_next_root)': 184, '(r_next_root)': 185,
    '(wordnet_synonym)': 186, '(l_wordnet_synonym)': 187, '(r_wordnet_synonym)': 188,
    '(co-reference)': 189, '(l_co-reference)': 190, '(r_co-reference)': 191, '(prep:at)': 192, '(l_prep:at)': 193,
    '(r_prep:at)': 194, '(prep:given)': 195, '(l_prep:given)': 196, '(r_prep:given)': 197, '(dative:for)': 198,
    '(l_dative:for)': 199, '(r_dative:for)': 200, '(advmod:during)': 201, '(l_advmod:during)': 202,
    '(r_advmod:during)': 203, '(relcl:for)': 204, '(l_relcl:for)': 205, '(r_relcl:for)': 206, '(prep:following)': 207,
    '(l_prep:following)': 208, '(r_prep:following)': 209, '(prep:although)': 210, '(l_prep:although)': 211,
    '(r_prep:although)': 212, '(prep:despite)': 213, '(l_prep:despite)': 214, '(r_prep:despite)': 215,
    '(prep:considering)': 216, '(l_prep:considering)': 217, '(r_prep:considering)': 218, '(conj:on)': 219,
    '(l_conj:on)': 220, '(r_conj:on)': 221, '(prep:trough)': 222, '(l_prep:trough)': 223, '(r_prep:trough)': 224,
    '(prep:wtih)': 225, '(l_prep:wtih)': 226, '(r_prep:wtih)': 227, '(prep:for)': 228, '(l_prep:for)': 229,
    '(r_prep:for)': 230, '(prep:through)': 231, '(l_prep:through)': 232, '(r_prep:through)': 233, '(prep:af)': 234,
    '(l_prep:af)': 235, '(r_prep:af)': 236, '(prep:about)': 237, '(l_prep:about)': 238, '(r_prep:about)': 239,
    '(prep:as)': 240, '(l_prep:as)': 241, '(r_prep:as)': 242, '(prep:after)': 243, '(l_prep:after)': 244,
    '(r_prep:after)': 245, '(prep:past)': 246, '(l_prep:past)': 247, '(r_prep:past)': 248, '(prep:regarding)': 249,
    '(l_prep:regarding)': 250, '(r_prep:regarding)': 251, '(prep:vs)': 252, '(l_prep:vs)': 253, '(r_prep:vs)': 254,
    '(dep:with)': 255, '(l_dep:with)': 256, '(r_dep:with)': 257, '(prep:onto)': 258, '(l_prep:onto)': 259,
    '(r_prep:onto)': 260, '(prep:than)': 261, '(l_prep:than)': 262, '(r_prep:than)': 263, '(prep:via)': 264,
    '(l_prep:via)': 265, '(r_prep:via)': 266, '(prep:toward)': 267, '(l_prep:toward)': 268, '(r_prep:toward)': 269,
    '(conj:with)': 270, '(l_conj:with)': 271, '(r_conj:with)': 272, '(acl:regarding)': 273, '(l_acl:regarding)': 274,
    '(r_acl:regarding)': 275, '(prep:amongst)': 276, '(l_prep:amongst)': 277, '(r_prep:amongst)': 278,
    '(prep:like)': 279, '(l_prep:like)': 280, '(r_prep:like)': 281, '(advcl:in)': 282, '(l_advcl:in)': 283,
    '(r_advcl:in)': 284, '(pcomp:following)': 285, '(l_pcomp:following)': 286, '(r_pcomp:following)': 287,
    '(amod:dose)': 288, '(l_amod:dose)': 289, '(r_amod:dose)': 290, '(preconj:dose)': 291, '(l_preconj:dose)': 292,
    '(r_preconj:dose)': 293, '(prep:among)': 294, '(l_prep:among)': 295, '(r_prep:among)': 296, '(prep:of)': 297,
    '(l_prep:of)': 298, '(r_prep:of)': 299, '(ccomp:to)': 300, '(l_ccomp:to)': 301, '(r_ccomp:to)': 302,
    '(cc:than)': 303, '(l_cc:than)': 304, '(r_cc:than)': 305, '(prep:due)': 306, '(l_prep:due)': 307,
    '(r_prep:due)': 308, '(xcomp:with)': 309, '(l_xcomp:with)': 310, '(r_xcomp:with)': 311, '(conj:at)': 312,
    '(l_conj:at)': 313, '(r_conj:at)': 314, '(acomp:due)': 315, '(l_acomp:due)': 316, '(r_acomp:due)': 317,
    '(prep:over)': 318, '(l_prep:over)': 319, '(r_prep:over)': 320, '(prep:with)': 321, '(l_prep:with)': 322,
    '(r_prep:with)': 323, '(conj:after)': 324, '(l_conj:after)': 325, '(r_conj:after)': 326, '(ccomp:compared)': 327,
    '(l_ccomp:compared)': 328, '(r_ccomp:compared)': 329, '(conj:following)': 330, '(l_conj:following)': 331,
    '(r_conj:following)': 332, '(prep:during)': 333, '(l_prep:during)': 334, '(r_prep:during)': 335, '(conj:in)': 336,
    '(l_conj:in)': 337, '(r_conj:in)': 338, '(prep:below)': 339, '(l_prep:below)': 340, '(r_prep:below)': 341,
    '(conj:by)': 342, '(l_conj:by)': 343, '(r_conj:by)': 344, '(prep:within)': 345, '(l_prep:within)': 346,
    '(r_prep:within)': 347, '(prep:under)': 348, '(l_prep:under)': 349, '(r_prep:under)': 350, '(prep:into)': 351,
    '(l_prep:into)': 352, '(r_prep:into)': 353, '(agent:by)': 354, '(l_agent:by)': 355, '(r_agent:by)': 356,
    '(prep:until)': 357, '(l_prep:until)': 358, '(r_prep:until)': 359, '(acl:following)': 360, '(l_acl:following)': 361,
    '(r_acl:following)': 362, '(npadvmod:times)': 363, '(l_npadvmod:times)': 364, '(r_npadvmod:times)': 365,
    '(prep:across)': 366, '(l_prep:across)': 367, '(r_prep:across)': 368, '(prep:from)': 369, '(l_prep:from)': 370,
    '(r_prep:from)': 371, '(acl:with)': 372, '(l_acl:with)': 373, '(r_acl:with)': 374, '(prep:in)': 375,
    '(l_prep:in)': 376, '(r_prep:in)': 377, '(prep:per)': 378, '(l_prep:per)': 379, '(r_prep:per)': 380,
    '(advcl:to)': 381, '(l_advcl:to)': 382, '(r_advcl:to)': 383, '(conj:between)': 384, '(l_conj:between)': 385,
    '(r_conj:between)': 386, '(prep:by)': 387, '(l_prep:by)': 388, '(r_prep:by)': 389, '(conj:for)': 390,
    '(l_conj:for)': 391, '(r_conj:for)': 392, '(acl:in)': 393, '(l_acl:in)': 394, '(r_acl:in)': 395,
    '(prep:because)': 396, '(l_prep:because)': 397, '(r_prep:because)': 398, '(amod:due)': 399, '(l_amod:due)': 400,
    '(r_amod:due)': 401, '(acl:given)': 402, '(l_acl:given)': 403, '(r_acl:given)': 404, '(dobj:in)': 405,
    '(l_dobj:in)': 406, '(r_dobj:in)': 407, '(acl:for)': 408, '(l_acl:for)': 409, '(r_acl:for)': 410,
    '(prep:between)': 411, '(l_prep:between)': 412, '(r_prep:between)': 413, '(prep:upon)': 414, '(l_prep:upon)': 415,
    '(r_prep:upon)': 416, '(advmod:after)': 417, '(l_advmod:after)': 418, '(r_advmod:after)': 419, '(mark:after)': 420,
    '(l_mark:after)': 421, '(r_mark:after)': 422, '(prep:unlike)': 423, '(l_prep:unlike)': 424, '(r_prep:unlike)': 425,
    '(prep:around)': 426, '(l_prep:around)': 427, '(r_prep:around)': 428, '(prep:since)': 429, '(l_prep:since)': 430,
    '(r_prep:since)': 431, '(prep:on)': 432, '(l_prep:on)': 433, '(r_prep:on)': 434, '(prep:excluding)': 435,
    '(l_prep:excluding)': 436, '(r_prep:excluding)': 437, '(prep:except)': 438, '(l_prep:except)': 439,
    '(r_prep:except)': 440, '(prep:including)': 441, '(l_prep:including)': 442, '(r_prep:including)': 443,
    '(advcl:by)': 444, '(l_advcl:by)': 445, '(r_advcl:by)': 446, '(advcl:on)': 447, '(l_advcl:on)': 448,
    '(r_advcl:on)': 449, '(prep:versus)': 450, '(l_prep:versus)': 451, '(r_prep:versus)': 452, '(advcl:with)': 453,
    '(l_advcl:with)': 454, '(r_advcl:with)': 455, '(prep:before)': 456, '(l_prep:before)': 457, '(r_prep:before)': 458,
    '(advcl:following)': 459, '(l_advcl:following)': 460, '(r_advcl:following)': 461, '(prep:while)': 462,
    '(l_prep:while)': 463, '(r_prep:while)': 464, '(prep:to)': 465, '(l_prep:to)': 466, '(r_prep:to)': 467,
    '(prep:above)': 468, '(l_prep:above)': 469, '(r_prep:above)': 470, '(pobj:in)': 471, '(l_pobj:in)': 472,
    '(r_pobj:in)': 473, '(prep:without)': 474, '(l_prep:without)': 475, '(r_prep:without)': 476, '(prep:towards)': 477,
    '(l_prep:towards)': 478, '(r_prep:towards)': 479, '(pcomp:to)': 480, '(l_pcomp:to)': 481, '(r_pcomp:to)': 482,
    '(prep:against)': 483, '(l_prep:against)': 484, '(r_prep:against)': 485, '(dative:to)': 486, '(l_dative:to)': 487,
    '(r_dative:to)': 488
}

TAG_POS = {
    '-LRB-': 0, '-PRB-': 1, ',': 2, ':': 3, '.': 4, "''": 5, '""': 6, '#': 7, '``': 8, '$': 9, 'ADD': 10, 'AFX': 11,
    'BES': 12, 'CC': 13, 'CD': 14, 'DT': 15, 'EX': 16, 'FW': 17, 'GW': 18, 'HVS': 19, 'HYPH': 20, 'IN': 21, 'JJ': 22,
    'JJR': 23, 'JJS': 24, 'LS': 25, 'MD': 26, 'NFP': 27, 'NIL': 28, 'NN': 29, 'NNP': 30, 'NNPS': 31, 'NNS': 32,
    'PDT': 33, 'POS': 34, 'PRP': 35, 'PRP$': 36, 'RB': 37, 'RBR': 38, 'RBS': 39, 'RP': 40, 'SP': 41, 'SYM': 42,
    'TO': 43, 'UH': 44, 'VB': 45, 'VBD': 46, 'VBG': 47, 'VBN': 48, 'VBP': 49, 'VBZ': 50, 'WDT': 51, 'WP': 52, 'WP$': 53,
    'WRB': 54, 'XX': 55
}
