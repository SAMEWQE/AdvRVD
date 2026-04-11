# coding=utf-8
import os
import re
import shutil
import argparse
from clean_gadget import clean_gadget

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', default='/home/pod/shared-nvme/pxyang/AdvRVD/datasets/reveal/valid',
 help='The dir path of input dataset', type=str, required=False)
    args = parser.parse_args()
    return args

def normalize(path):
    if os.path.isfile(path):
        if path.endswith('.c'):
            normalization(path)
        return

    for root, dirs, files in os.walk(path):
        for file in files:
            # 只处理 .c 文件，忽略其他所有文件
            if not file.endswith('.c'):
                continue
            
            filepath = os.path.join(root, file)
            # print(filepath)
            try:
                normalization(filepath)
            except Exception as e:
                print('Error processing ' + filepath + ': ' + str(e))

def normalization(filepath):
    # print(filepath)
    try:
        with open(filepath, 'r') as file:
            fun = file.read()
    except IsADirectoryError:
        return
    except Exception:
        return

    lines = fun.split('\n')
    # if len(lines) < 30 or len(lines) > 200:
    #     os.system('rm ' + filepath)
    #     return
    code = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        line = re.sub('#.*', '', line)
        code += line + ' '
    # fun_name = code.split('(')[0]
    # if re.match('.*main.*', fun_name):
    #     os.system('rm ' + filepath)
    #     return
    code = re.sub('/\*.*?\*/', '', code)
    code = clean_gadget([code])
    with open(filepath, 'w') as file:
        file.writelines(code[0])
    file.close()
    print(code[0])
def pro_one_file(filepath):
    with open(filepath, 'r') as file:
        code = file.read()
    file.close()
    if re.search('.*::.*', code) or re.search('.*:.*', code):
        os.system('rm ' + filepath)
        return
    lines = code.split('\n')
    if len(lines) < 480:
        code = re.sub('(?<!:)\/\/.*|\/\*(\s|.)*?\*\/', '', code)
    else:
        os.system('rm ' + filepath)
        return
    # code = re.sub('(?<!:)\/\/.*|\/\*(\s|.)*?\*\/', '', code)
    #print(code)
    with open(filepath, 'w') as file:
        file.write(code.strip())
    file.close()
    with open(filepath, 'r') as file:
        org_code = file.readlines()
        # print(org_code)
        nor_code = clean_gadget(org_code)
    file.close()
    with open(filepath, 'w') as file:
        file.writelines(nor_code)
    file.close()


def main():
    args = parse_options()
    normalize(args.input)
    

if __name__ == '__main__':
    # pro_one_file('/home/pod/shared-nvme/pxyang/win_linux_mapping/Adversarial_Reprogramming-master/datasets/vuldeepecker/train/No-Vul/Good59369.c')
    main()
