import numpy as np

PAM250 = {
    'A': {'A':  2, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N':  0, 'P':  1, 'Q':  0, 'R': -2, 'S':  1, 'T':  1, 'V':  0, 'W': -6, 'Y': -3},
    'C': {'A': -2, 'C': 12, 'D': -5, 'E':-5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4, 'P': -3, 'Q': -5, 'R': -4, 'S':  0, 'T': -2, 'V': -2, 'W': -8, 'Y':  0},
    'D': {'A':  0, 'C': -5, 'D':  4, 'E': 3, 'F': -6, 'G':  1, 'H':  1, 'I': -2, 'K':  0, 'L': -4, 'M': -3, 'N':  2, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
    'E': {'A':  0, 'C': -5, 'D':  3, 'E': 4, 'F': -5, 'G':  0, 'H':  1, 'I': -2, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
    'F': {'A': -3, 'C': -4, 'D': -6, 'E':-5, 'F':  9, 'G': -5, 'H': -2, 'I':  1, 'K': -5, 'L':  2, 'M':  0, 'N': -3, 'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W':  0, 'Y':  7},
    'G': {'A':  1, 'C': -3, 'D':  1, 'E': 0, 'F': -5, 'G':  5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P':  0, 'Q': -1, 'R': -3, 'S':  1, 'T':  0, 'V': -1, 'W': -7, 'Y': -5},
    'H': {'A': -1, 'C': -3, 'D':  1, 'E': 1, 'F': -2, 'G': -2, 'H':  6, 'I': -2, 'K':  0, 'L': -2, 'M': -2, 'N':  2, 'P':  0, 'Q':  3, 'R':  2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y':  0},
    'I': {'A': -1, 'C': -2, 'D': -2, 'E':-2, 'F':  1, 'G': -3, 'H': -2, 'I':  5, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -5, 'Y': -1},
    'K': {'A': -1, 'C': -5, 'D':  0, 'E': 0, 'F': -5, 'G': -2, 'H':  0, 'I': -2, 'K':  5, 'L': -3, 'M':  0, 'N':  1, 'P': -1, 'Q':  1, 'R':  3, 'S':  0, 'T':  0, 'V': -2, 'W': -3, 'Y': -4},
    'L': {'A': -2, 'C': -6, 'D': -4, 'E':-3, 'F':  2, 'G': -4, 'H': -2, 'I':  2, 'K': -3, 'L':  6, 'M':  4, 'N': -3, 'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V':  2, 'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -5, 'D': -3, 'E':-2, 'F':  0, 'G': -3, 'H': -2, 'I':  2, 'K':  0, 'L':  4, 'M':  6, 'N': -2, 'P': -2, 'Q': -1, 'R':  0, 'S': -2, 'T': -1, 'V':  2, 'W': -4, 'Y': -2},
    'N': {'A':  0, 'C': -4, 'D':  2, 'E': 1, 'F': -3, 'G':  0, 'H':  2, 'I': -2, 'K':  1, 'L': -3, 'M': -2, 'N':  2, 'P':  0, 'Q':  1, 'R':  0, 'S':  1, 'T':  0, 'V': -2, 'W': -4, 'Y': -2},
    'P': {'A':  1, 'C': -3, 'D': -1, 'E':-1, 'F': -5, 'G':  0, 'H':  0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N':  0, 'P':  6, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -1, 'W': -6, 'Y': -5},
    'Q': {'A':  0, 'C': -5, 'D':  2, 'E': 2, 'F': -5, 'G': -1, 'H':  3, 'I': -2, 'K':  1, 'L': -2, 'M': -1, 'N':  1, 'P':  0, 'Q':  4, 'R':  1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
    'R': {'A': -2, 'C': -4, 'D': -1, 'E':-1, 'F': -4, 'G': -3, 'H':  2, 'I': -2, 'K':  3, 'L': -3, 'M':  0, 'N':  0, 'P':  0, 'Q':  1, 'R':  6, 'S':  0, 'T': -1, 'V': -2, 'W':  2, 'Y': -4},
    'S': {'A':  1, 'C':  0, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P':  1, 'Q': -1, 'R':  0, 'S':  2, 'T':  1, 'V': -1, 'W': -2, 'Y': -3},
    'T': {'A':  1, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  0, 'H': -1, 'I':  0, 'K':  0, 'L': -2, 'M': -1, 'N':  0, 'P':  0, 'Q': -1, 'R': -1, 'S':  1, 'T':  3, 'V':  0, 'W': -5, 'Y': -3},
    'V': {'A':  0, 'C': -2, 'D': -2, 'E':-2, 'F': -1, 'G': -1, 'H': -2, 'I':  4, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -6, 'Y': -2},
    'W': {'A': -6, 'C': -8, 'D': -7, 'E':-7, 'F':  0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4, 'P': -6, 'Q': -5, 'R':  2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y':  0},
    'Y': {'A': -3, 'C':  0, 'D': -4, 'E':-4, 'F':  7, 'G': -5, 'H':  0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2, 'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W':  0, 'Y': 10}
}


def backtrack(r, c):
    if (r == 0) or (c == 0):
        #print(":/")
        new_seq1.reverse()
        new_seq2.reverse()
        str1 = ""
        seq1_string = str1.join(new_seq1)
        seq2_string = str1.join(new_seq2)
        new_seq1.reverse()
        new_seq2.reverse()
        seq_begin = list()
        seq_end = list()
        gaps = list()
        # fixing the very first part of sequences
        if (c == 0) and (r != 0):
            count = r
            while count != 0:
                seq_begin.append(seq2[count - 1])
                count = count - 1
                gaps.append("-")
            seq_begin.reverse()
            seq2_string = str1.join(seq_begin) + "" + seq2_string
            seq1_string = str1.join(gaps) + "" + seq1_string

        elif (r == 0) and (c != 0):
            count = c
            while count != 0:
                seq_begin.append(seq1[count - 1])
                count = count - 1
                gaps.append("-")
            seq_begin.reverse()
            seq1_string = str1.join(seq_begin) + "" + seq1_string
            seq2_string = str1.join(gaps) + "" + seq2_string
        # fixing the very last part of sequences
        gaps = list()
        if (start_backtrack_coordinate[1] == columns - 1) and (start_backtrack_coordinate[0] != rows - 1):
            count = start_backtrack_coordinate[0] + 1
            while count < (len(seq2) + 1):
                seq_end.append(seq2[count - 1])
                count = count + 1
                gaps.append("-")
            seq2_string = seq2_string + "" + str1.join(seq_end)
            seq1_string = seq1_string + "" + str1.join(gaps)

        elif (start_backtrack_coordinate[0] == rows - 1) and (start_backtrack_coordinate[1] != columns - 1):
            count = start_backtrack_coordinate[1] + 1
            while count < (len(seq1) + 1):
                seq_end.append(seq1[count - 1])
                count = count + 1
                gaps.append("-")
            seq1_string = seq1_string + "" + str1.join(seq_end)
            seq2_string = seq2_string + "" + str1.join(gaps)

        seq.append(seq1_string + "" + seq2_string)

    else:
        new_new_arrow = dict()
        for k in new_arrow:
            new_new_arrow[k] = list()
            for e in arrow[k]:
                new_new_arrow[k].append(e)
        r_string = str(r)
        c_string = str(c)
        new_new_arrow_key_generate = r_string + " " + c_string
        while len(new_new_arrow[new_new_arrow_key_generate]) != 0:
            #print("---")
            #print(len(new_new_arrow[new_new_arrow_key_generate]))
            #print(r)
            #print(c)
            #print(new_new_arrow[new_new_arrow_key_generate][0])
            #print("---")
            direction = new_new_arrow[new_new_arrow_key_generate][0]
            if direction == "l":
                new_seq1.append(seq1[c - 1])
                new_seq2.append("-")
                backtrack(r, c - 1)
                new_seq1.remove(seq1[c - 1])
                new_seq2.remove("-")
                new_new_arrow[new_new_arrow_key_generate].remove("l")
            elif direction == "u":
                new_seq1.append("-")
                new_seq2.append(seq2[r - 1])
                backtrack(r - 1, c)
                new_seq1.remove("-")
                new_seq2.remove(seq2[r - 1])
                new_new_arrow[new_new_arrow_key_generate].remove("u")
            elif direction == "d":
                new_seq1.append(seq1[c - 1])
                new_seq2.append(seq2[r - 1])
                backtrack(r - 1, c - 1)
                new_seq1.remove(seq1[c - 1])
                new_seq2.remove(seq2[r - 1])
                new_new_arrow[new_new_arrow_key_generate].remove("d")


seq1 = input()
seq2 = input()
rows = len(seq2) + 1
columns = len(seq1) + 1
matrix = np.zeros((rows, columns))
arrow = dict()
gp = 9
left_value = 0
up_value = 0
diagonal_value = 0
last_row_column_elements = dict()
score = 0
seq = list()
new_seq1 = list()
new_seq2 = list()
for i in range(1, rows):
    for j in range(1, columns):
        left_value = matrix[i][j - 1] - gp
        up_value = matrix[i - 1][j] - gp
        diagonal_value = matrix[i - 1][j - 1] + PAM250[seq1[j - 1]][seq2[i - 1]]
        values = [left_value, up_value, diagonal_value]
        maximum = max(values)
        matrix[i][j] = maximum
        i_string = str(i)
        j_string = str(j)
        arrow_key = i_string + " " + j_string
        arrow[arrow_key] = list()
        if maximum == left_value:
            arrow[arrow_key].append("l")

        if maximum == up_value:
            arrow[arrow_key].append("u")

        if maximum == diagonal_value:
            arrow[arrow_key].append("d")

        # are being used in backtrack part for starting from the maximum element
        if ((i == rows - 1) and (j != 0)) or ((i != 0) and (j == columns - 1)):
            dict_key = str(i) + " " + str(j)
            last_row_column_elements[dict_key] = matrix[i][j]

last_row_column_elements_key_max = max(last_row_column_elements, key=last_row_column_elements.get)

start_backtrack_coordinate = list()
for element in last_row_column_elements:
    new_arrow = dict()
    if last_row_column_elements[element] == last_row_column_elements[last_row_column_elements_key_max]:
        for k in arrow:
            new_arrow[k] = list()
            for e in arrow[k]:
                new_arrow[k].append(e)

        coordinate = element.split()
        start_backtrack_row = int(coordinate[0])
        start_backtrack_column = int(coordinate[1])
        start_backtrack_coordinate.append(start_backtrack_row)
        start_backtrack_coordinate.append(start_backtrack_column)
        score = matrix[start_backtrack_row][start_backtrack_column]
        backtrack(start_backtrack_row, start_backtrack_column)
        start_backtrack_coordinate = list()

score = int(score)
print(score)
seq.sort()
for i in seq:
    print(i[0:int(len(i)/2)])
    print(i[int(len(i)/2):])

#print(matrix)
#print(arrow)