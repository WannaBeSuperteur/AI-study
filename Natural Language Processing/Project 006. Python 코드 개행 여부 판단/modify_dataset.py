import os


# find the number of spaces (' ') before any non-space character
def get_starting_spaces(line):
    result = 0
    while result < len(line):
        if line[result] == ' ':
            result += 1
        else:
            break

    return result


# modify code using rules
def modify_code(lines):
    result = []
    starting_spaces_count = []

    # get starting spaces count
    for line in lines:
        starting_spaces_count.append(get_starting_spaces(line))

    # modify code
    for idx, line in enumerate(lines):
        if idx >= 1:
            previous_line = lines[idx - 1]
        else:
            previous_line = None
            
        is_current_line_empty = (len(line.replace('\t', '').replace(' ', '').replace('\n', '')) == 0)

        if not is_current_line_empty:
            
            # indent count decrease -> add an empty line
            if previous_line is not None and (previous_line.count('\t') > line.count('\t') or starting_spaces_count[idx - 1] > starting_spaces_count[idx]):
                result.append('\n')

            # current line starts with 'def' -> add an empty line
            elif line.replace('\t', '').replace(' ', '').startswith('def'):
                result.append('\n')

            # current line starts with 'if' -> add an empty line
            elif line.replace('\t', '').replace(' ', '').startswith('if'):
                result.append('\n')

        # add current line
        result.append(line)

    return result


if __name__ == '__main__':
    code_file = open('Python_code_data.txt', 'r', encoding='utf-8')
    lines = code_file.readlines()
    lines_modified = modify_code(lines)

    new_code_data = ''.join(lines_modified)
    
    new_code_file = open('Python_code_data_modified.txt', 'w', encoding='utf-8')
    new_code_file.write(new_code_data)
    new_code_file.close()
