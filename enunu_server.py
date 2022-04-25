#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import json
import os
import subprocess
import sys
import traceback

sys.path.append(os.path.dirname(__file__))
import enunu
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.8.10-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on


def timing(path_ust: str):
    path_full_timing, path_mono_timing = enunu.run_timing(path_ust)
    return {
        'path_full_timing': path_full_timing,
        'path_mono_timing': path_mono_timing,
    }


def acoustic(path_ust: str):
    path_acoustic, path_f0, path_spectrogram, \
        path_aperiodicity = enunu.run_acoustic(path_ust)
    return {
        'path_acoustic': path_acoustic,
        'path_f0': path_f0,
        'path_spectrogram': path_spectrogram,
        'path_aperiodicity': path_aperiodicity,
    }


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:15555')
    print('Started enunu server')

    while True:
        message = socket.recv()
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        try:
            if request[0] == 'timing':
                response['result'] = timing(request[1])
            elif request[0] == 'acoustic':
                response['result'] = acoustic(request[1])
            else:
                raise NotImplementedError('unexpected command %s' % request[0])
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()

        print('Sending response: %s' % response)
        socket.send_string(response)


if __name__ == '__main__':
    main()
