#!/usr/bin/env python2
# encoding: utf-8

import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', '-p', type=int, required=True,
                        help='Number of subparts to be merged')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='json file dir')
    parser.add_argument('--result-label', type=str, required=True,
                        help='path to save json file')
    parser.add_argument('--offset', type=int, default=0,
                        help='offset of file name')
    args = parser.parse_args()

    new_js = {}
    for i in range(args.parts):
        file_name = os.path.join(args.result_dir, 'data.{}.json'.format(i + args.offset))
        if not os.path.exists(file_name):
            continue
        js = json.load(open(file_name, 'rb'))['utts']
        new_js.update(js)

    print('After merged:', len(new_js))
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
