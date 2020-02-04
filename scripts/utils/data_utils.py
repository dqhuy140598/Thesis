from scripts.config import args


def parse_raw_dataset(lines):
    all_words = []
    all_positions = []
    all_relations = []
    all_directions = []
    all_poses = []
    all_labels = []
    all_lens = []
    all_identifier = []
    pmid = ''
    for line in lines:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    nodes = sdp.split()
                    words = []
                    positions = []
                    poses = []
                    relations = []
                    directions = []

                    for idx, node in enumerate(nodes):
                        if idx % 2 == 0:
                            word_pos = args.UNK if node == '' else node
                            w, p = word_pos.rsplit('/', 1)

                            p = 'NN' if p == '' else p
                            w, position = w.rsplit('_', 1)
                            words.append(w)
                            positions.append(min(int(position), args.MAX_LENGTH))
                            poses.append(p)
                        else:
                            dependency = node
                            r = '(' + dependency[3:]
                            d = dependency[1]
                            r = r.split(':', 1)[0] + ')' if ':' in r else r
                            relations.append(r)
                            directions.append(d)
                    all_words.append(words)
                    all_positions.append(positions)
                    all_relations.append(relations)
                    all_directions.append(directions)
                    all_labels.append([label])
                    all_poses.append(poses)
                    all_identifier.append((pmid, pair))
            else:
                print(l)

    assert len(all_words) == len(all_labels)
    # print(len(all_labels))

    return all_words, all_positions, all_labels, all_poses, \
           all_relations, all_directions, all_identifier


def process_dataset(lines, words_vocab, poses_vocab, relations_vocab):
    all_words, all_positions, all_labels, all_poses, all_relations, all_directions, all_identifier = parse_raw_dataset(
        lines)

    words = []
    positions_1 = []
    positions_2 = []
    labels = []
    poses = []
    relations = []
    directions = []

    for i in range(len(all_positions)):
        position_1, position_2 = [], []
        e1 = all_positions[i][0]
        e2 = all_positions[i][-1]

        for po in all_positions[i]:
            position_1.append((po - e1 + args.MAX_LENGTH) // 5 + 1)
            position_2.append((po - e2 + args.MAX_LENGTH) // 5 + 1)

        positions_1.append(position_1)
        positions_2.append(position_2)

    for i in range(len(all_words)):
        rs = []
        for r in all_relations[i]:
            rid = relations_vocab[r]
            rs += [rid]
        relations.append(rs)

        ds = []
        for d in all_directions[i]:
            did = 1 if d == 'l' else 2

            ds += [did]

        directions.append(ds)

        ws, ps = [], []

        for w, p in zip(all_words[i], all_poses[i]):

            if w in words_vocab.keys():
                word_id = words_vocab[w]

            else:
                word_id = words_vocab[args.UNK]

            ws.append(word_id)
            p_id = poses_vocab[p]
            ps += [p_id]
        words.append(ws)
        poses.append(ps)

        lb = args.ALL_LABELS.index(all_labels[i][0])
        labels.append(lb)

    return words, positions_1, positions_2, \
           labels, poses, relations, directions


def pad_to_same(words, pad_token, max_length):
    pad_length = (max_length - len(words)) // 2
    left_padded = [pad_token] * pad_length
    right_padded = [pad_token] * (max_length - pad_length - len(words))
    padded = []
    padded.extend(left_padded)
    padded.extend(words)
    padded.extend(right_padded)

    return padded
