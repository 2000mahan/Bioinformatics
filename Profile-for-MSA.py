import numpy as np
import math
import itertools

pseudocount = 2
N = input()
N = int(N)
input_sequences = list()

for i in range(0, N):
    input_sequences.append(input())
seq = input()
chars_in_input_sequences = list()
position_len = len(input_sequences[0])
for i in range(0, N):
    for j in input_sequences[i]:
        if j not in chars_in_input_sequences:
            chars_in_input_sequences.append(j)

profile = np.zeros((len(chars_in_input_sequences), position_len))

B = len(chars_in_input_sequences)
# step 1, 2
counter = 0
for c in chars_in_input_sequences:
    for p in range(0, position_len):
        count = 0
        for i in range(0, N):
            if input_sequences[i][p] == c:
                count = count + 1
        profile[counter][p] = (count + pseudocount)/(N + (B * pseudocount))
        profile[counter][p] = format(profile[counter][p], ".3f")

    counter = counter + 1

#print(profile)

# step 3
counter = 0
overall = list()
for c in chars_in_input_sequences:
    sum = 0
    for p in range(0, position_len):
        sum += profile[counter][p]
    overall.append(sum/position_len)
    counter = counter + 1

#print(overall)

# step 4, 5
counter = 0
for c in chars_in_input_sequences:
    for p in range(0, position_len):
        profile[counter][p] = profile[counter][p]/overall[counter]
        profile[counter][p] = math.log(profile[counter][p], 2)
        profile[counter][p] = format(profile[counter][p], ".3f")
    counter = counter + 1

#print(profile)

states = dict()
counter = position_len
count = 0
for i in range(counter, 0, -1):
    for j in range(0, len(seq)):
        res = seq[j:i + j]
        score = 0
        if len(res) < position_len:
            res_list = list(res)
            for k in range(0, position_len - len(res)):
                res_list.append('-')
            permutations = list(itertools.permutations(res_list))
            for b in permutations:
                new_res = ''
                new_res = ''.join(b)
                indexes = list()
                for l in range(0, len(res)):
                    indexes.append(new_res.index(res[l]))
                #if res == 'HLP' and new_res == 'H-L-P':
                #print(indexes)
                if not (all(indexes[i] <= indexes[i+1] for i in range(len(indexes) - 1))):
                    continue
                score = 0
                for p in range(0, len(new_res)):
                    index = chars_in_input_sequences.index(new_res[p])
                    score += profile[index][p]

                states[new_res] = score
        else:
            score = 0
            for p in range(0, len(res)):
                index = chars_in_input_sequences.index(res[p])
                score += profile[index][p]
            states[res] = score


key_with_max_value = max(states, key=states.get)
print(key_with_max_value)











