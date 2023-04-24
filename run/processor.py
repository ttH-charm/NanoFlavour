#!/usr/bin/env python3
from __future__ import print_function

import os
import time
import json
import argparse
import subprocess


def xrd_prefix(filepaths):
    prefix = ''
    allow_prefetch = False
    if not isinstance(filepaths, (list, tuple)):
        filepaths = [filepaths]
    filepath = filepaths[0]
    if filepath.startswith('/eos/cms'):
        prefix = 'root://eoscms.cern.ch/'
    elif filepath.startswith('/eos/user'):
        prefix = 'root://eosuser.cern.ch/'
    elif filepath.startswith('/eos/uscms'):
        prefix = 'root://cmseos.fnal.gov/'
    elif filepath.startswith('/store/'):
        # remote file
        import socket
        host = socket.getfqdn()
        if 'cern.ch' in host:
            prefix = 'root://xrootd-cms.infn.it//'
        else:
            prefix = 'root://cmsxrootd.fnal.gov//'
        allow_prefetch = True
    expanded_paths = [(prefix + '/' + f if prefix else f) for f in filepaths]
    return expanded_paths, allow_prefetch


def outputName(md, jobid):
    info = md['jobs'][jobid]
    return '{samp}_{idx}_tree.root'.format(samp=info['samp'], idx=info['idx'])


def main(args):

    # load job metadata
    with open(args.metadata) as fp:
        md = json.load(fp)

    # load modules
    modules = []
    for mod, names in md['imports']:
        print(mod, names)
        modules.append(f'-I {mod} {names}')

    # remove any existing root files
    for f in os.listdir('.'):
        if f.endswith('.root'):
            print('Removing file %s' % f)
            os.remove(f)

    # run postprocessor
    inputfiles = args.files if len(args.files) else md['jobs'][args.jobid]['inputfiles']
    filepaths, allow_prefetch = xrd_prefix(inputfiles)
    print(filepaths)
    outputname = outputName(md, args.jobid)

    for idx, fpath in enumerate(filepaths):
        cmd = "nano_postproc.py -c '{cut}' --bi {branchsel} --bo {outputbranchsel} -z {compression} {jsonInput} --first-entry {firstEntry} {postfix} {modules} {friend} {prefetch} {maxEntries} {outputDir} {inputFiles}".format(
            outputDir='.',
            inputFiles=fpath,
            cut=md.get('cut', '1==1'),
            branchsel=os.path.basename(md['branchsel_in']),
            outputbranchsel=os.path.basename(md['branchsel_out']),
            modules=' '.join(modules),
            compression=md.get('compression', 'LZMA:9'),
            friend='--friend' if md.get('friend', False) else '',
            postfix='-s %s' % md['postfix'] if md.get('postfix') else '',
            jsonInput='-J %s' % md['json'] if md.get('json') else '',
            prefetch='--prefetch' if (allow_prefetch and md.get('prefetch', False)) else '',
            maxEntries='-N %s' % md['maxEntries'] if md.get('maxEntries', None) else '',
            firstEntry=md.get('firstEntry', 0),
        )
        print(f'====== Start processing {idx}/{len(filepaths)} files ======')
        print('    ' + cmd)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f'Failed to process file {fpath}!')

    # hadd files
    p = subprocess.Popen('haddnano.py %s *.root' % outputname, shell=True)
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Hadd failed!')

    # keep only the hadd file
    for f in os.listdir('.'):
        if f.endswith('.root') and f != outputname:
            os.remove(f)

    # stage out
    if md['outputdir'].startswith('/eos'):
        cmd = 'xrdcp --silent -f {outputname} {outputdir}/{outputname}'.format(
            outputname=outputname, outputdir=xrd_prefix(md['joboutputdir'])[0][0])
        print(cmd)
        success = False
        for count in range(args.max_retry):
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            if p.returncode == 0:
                success = True
                break
            else:
                time.sleep(args.sleep)
        if not success:
            raise RuntimeError("Stage out FAILED!")

        # clean up
        os.remove(outputname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NanoAOD postprocessing.')
    parser.add_argument('-m', '--metadata',
                        default='metadata.json',
                        help='Path to the metadata file. Default:%(default)s')
    parser.add_argument('--max-retry',
                        type=int, default=3,
                        help='Max number of retry for stageout. Default: %(default)s'
                        )
    parser.add_argument('--sleep',
                        type=int, default=120,
                        help='Seconds to wait before retry stageout. Default: %(default)s'
                        )
    parser.add_argument('--files',
                        nargs='*', default=[],
                        help='Run over the specified input file. Default:%(default)s')
    parser.add_argument('jobid', type=int, help='Index of the output job.')

    args = parser.parse_args()
    main(args)
