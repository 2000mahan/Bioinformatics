import numpy as np
def global_align(x, y, s_match, s_mismatch, s_gap):
    A = []

    for i in range(len(y) + 1):
        A.append([0] * (len(x) + 1))

    for i in range(len(y) + 1):
        A[i][0] = s_gap * i

    for i in range(len(x) + 1):
        A[0][i] = s_gap * i

    for i in range(1, len(y) + 1):

        for j in range(1, len(x) + 1):
            A[i][j] = max(

                A[i][j - 1] + s_gap,

                A[i - 1][j] + s_gap,

                A[i - 1][j - 1] + (s_match if (y[i - 1] == x[j - 1] and y[i - 1] != '-') else 0) + (

                    s_mismatch if (y[i - 1] != x[j - 1] and y[i - 1] != '-' and x[j - 1] != '-') else 0) + (

                    s_gap if (y[i - 1] == '-' or x[j - 1] == '-') else 0)

            )

    align_X = ""

    align_Y = ""

    i = len(x)

    j = len(y)

    while i > 0 or j > 0:

        current_score = A[j][i]

        if i > 0 and j > 0 and (

                ((x[i - 1] == y[j - 1] and y[j - 1] != '-') and current_score == A[j - 1][i - 1] + s_match) or

                ((y[j - 1] != x[i - 1] and y[j - 1] != '-' and x[i - 1] != '-') and current_score == A[j - 1][

                    i - 1] + s_mismatch) or

                ((y[j - 1] == '-' or x[i - 1] == '-') and current_score == A[j - 1][i - 1] + s_gap)

        ):

            align_X = x[i - 1] + align_X

            align_Y = y[j - 1] + align_Y

            i = i - 1

            j = j - 1

        elif i > 0 and (current_score == A[j][i - 1] + s_gap):

            align_X = x[i - 1] + align_X

            align_Y = "-" + align_Y

            i = i - 1

        else:

            align_X = "-" + align_X

            align_Y = y[j - 1] + align_Y

            j = j - 1

    return (align_X, align_Y, A[len(y)][len(x)])


def score_calculator(sequences):
    score = 0
    seq = list()
    for i in sequences:
        seq.append(i)

    for i in range(0, len(seq[0])):
        for p in range(0, len(seq) - 1):
            for q in range(p + 1, len(seq)):
                if (seq[p][i] == seq[q][i]) and (seq[p][i] != "-"):
                    score = score + match
                else:
                    if (seq[p][i] == "-") and (seq[q][i] == "-"):
                        score = score + 0
                    elif (seq[p][i] != "-") and (seq[q][i] != "-"):
                        score = score + mismatch
                    elif ((seq[p][i] == "-") and (seq[q][i] != "-")) or ((seq[p][i] != "-") and (seq[q][i] == "-")):
                        score = score + gap
    return score


def star_alignment(input_sequences, n):
    score_list = list()
    score_dict = dict()
    for i in range(0, n):
        for j in range(i + 1, n):
            str_i = str(i)
            str_j = str(j)
            i_j = str_i + str_j
            score_dict[i_j] = global_align(input_sequences[i], input_sequences[j], match, mismatch, gap)

    for i in range(0, n):
        score = 0
        for j in score_dict:
            str_i = str(i)
            if str_i in j:
                score = score + score_dict[j][2]
        score_list.append(score)

    score_per_sequence = dict()

    seq_count = 0
    for i in score_list:
        score_per_sequence[seq_count] = i
        seq_count = seq_count + 1

    score_per_sequence = dict(sorted(score_per_sequence.items(), key=lambda item: item[1], reverse=True))
    k = list(score_per_sequence.keys())
    max_value = score_per_sequence[k[0]]
    max_value_index = k[0]
    new_sequences = list()

    for i in range(0, n):
        new_sequences.append(" ")

    pre_center = input_sequences[max_value_index]
    center = input_sequences[max_value_index]

    align_score = dict()
    c = 0
    for i in input_sequences:
        if c != max_value_index:
            _, _, s = global_align(i, center, match, mismatch, gap)
            align_score[c] = s
        c = c + 1
    align_score = dict(sorted(align_score.items(), key=lambda item: item[1], reverse=True))
    align_score_keys = list(align_score.keys())
    iteration = 0
    for i in align_score_keys:
        center_gaps = list()
        if i != max_value_index:
            iteration = iteration + 1
            pre_center = center
            res = global_align(input_sequences[i], center, match, mismatch, gap)
            center = res[1]
            new_sequences[i] = res[0]

            p1 = 0
            p2 = 0
            # print(new_sequences)
            while True:
                # pre_center[p1] != "-" or center[p2] != "-" both mean the same
                if p1 >= len(pre_center) and p2 >= len(center):
                    break
                if p1 >= len(pre_center) and center[p2] == "-":
                    center_gaps.append(p2)
                    p2 = p2 + 1
                    continue
                if p1 < len(pre_center):
                    if center[p2] == "-" and pre_center[p1] != "-":
                        center_gaps.append(p2)
                        p2 = p2 + 1
                        continue

                if p2 == len(center):
                    break

                if pre_center[p1] == center[p2]:
                    p1 = p1 + 1
                    p2 = p2 + 1
                    continue

            count = 0
            for s in new_sequences:
                if s != " " and s != new_sequences[i]:
                    for gap_position in center_gaps:
                        sequence = new_sequences[count]
                        new_sequences[count] = sequence[:gap_position] + "-" + sequence[gap_position:]
                count = count + 1
            # score = score + score_calculator(new_sequences, center)
        else:
            continue

    count = 0

    for s in new_sequences:
        if s == " ":
            new_sequences[count] = center
            break
        count = count + 1

    return new_sequences, count


def block_detection(sequences):
    candidates = list()
    blocks = list()
    start = -1
    end = -1
    block_size = -1
    # count1 checks that if all the entries in one column are the same
    count1 = 0
    for i in range(0, len(sequences[0])):
        for p in range(0, len(sequences) - 1):
            if sequences[p][i] == sequences[p + 1][i]:
                count1 = count1 + 1

        if count1 != (len(sequences) - 1):
            candidates.append(i)
        count1 = 0
    # print(candidates)
    for i in range(0, len(candidates) - 1):
        if candidates[i + 1] - candidates[i] == 1:
            blocks.append(candidates[i])
            blocks.append(candidates[i + 1])

    blocks = set(blocks)
    blocks = list(blocks)

    blocks_details = list()
    if len(blocks) == 0:
        return blocks
    start = blocks[0]
    for i in range(0, len(blocks) - 1):
        if blocks[i + 1] - blocks[i] == 1:
            if i == (len(blocks) - 2):
                end = blocks[i + 1]
                blocks_details.append(start)
                blocks_details.append(end)
                block_size = end - start
                blocks_details.append(block_size)
        else:
            end = blocks[i]
            blocks_details.append(start)
            blocks_details.append(end)
            block_size = end - start
            blocks_details.append(block_size)
            start = blocks[i + 1]
    return blocks_details


def create_block(start, end, size, sequences):
    new_seqs = list()
    for i in range(0, len(sequences)):
        new_seq = sequences[i][start:end + 1]
        new_seq = new_seq.replace("-", "")
        new_seqs.append(new_seq)
    return new_seqs

#while True:
#    n = np.random.randint(low=6, high=12)
#    random_seq = list()
#    random_len = 0
#    for j in range(n):
#        random_len = np.random.randint(low=8, high=14)
#        sequence = ""
#        for i in range(random_len):
#            new_c = str(np.random.randint(low=0, high=9))
#            sequence = sequence + new_c
#        random_seq.append(sequence)

match = 3
mismatch = -1
gap = -2
center = ""
n = input()
input_sequences = list()
n = int(n)
new_sequences = list()
ns = list()
for i in range(0, n):
    ns.append(" ")
score = 0

for i in range(0, n):
    seq = input()
    input_sequences.append(seq)
    #input_sequences = random_seq

new_sequences, center_index = star_alignment(input_sequences, n)


score = score_calculator(new_sequences)

b = block_detection(new_sequences)
for i in range(0, len(b), 3):
    new_seqs = create_block(b[i], b[i + 1], b[i + 2], new_sequences)
    new_seqs, _ = star_alignment(new_seqs, n)
    for j in range(0, n):
        s = new_sequences[j]
        ns[j] = s[:b[i]] + new_seqs[j] + s[b[i + 1] + 1:]
    new_score = score_calculator(ns)
    if new_score > score:
        new_sequences = ns
        score = new_score

print(score)
for i in new_sequences:
    print(i)

