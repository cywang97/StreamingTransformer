#!/usr/bin/env python2
# encoding: utf-8

import argparse
import json
import os

def write_json(js, json_file):
    with open(json_file, 'wb') as f:
        f.write(json.dumps({'utts': js}, indent=4, sort_keys=True).encode('utf_8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', '-p', type=int, required=True,
                        help='Number of subparts to be prepared', default=0)
    parser.add_argument('--json', type=str, required=True,
                        help='json file')
    parser.add_argument('--datadir', type=str, required=True,
                        help='path to save json file')
    parser.add_argument('--offset', type=int, default=0,
                        help='offset of file name')
    args = parser.parse_args()

    js = json.load(open(args.json, 'rb'))['utts']
    js = list(sorted(js.items(), key=lambda x: -x[1]['input'][0]['shape'][0]))

    new_js = [[] for _ in range(args.parts)]
    for i, j in enumerate(js):
        new_js[i % args.parts].append(j)

    file_name_prefix = args.json.split('/')[-1].split('.')[0]
    for i, new_j in enumerate(new_js):
        print('Part {}: {}'.format(i, len(new_j)))
        file_name = file_name_prefix + '_{}.json'.format(i + args.offset)
        write_json(dict(new_j), os.path.join(args.datadir, file_name))