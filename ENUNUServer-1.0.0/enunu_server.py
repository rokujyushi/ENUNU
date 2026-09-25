#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
import numpy as np
import enunu
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.12.10-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on

def check():
    return{
        'name': 'SimpleENUNUServer',
        'version': '1.0.0',
        'author': 'roku10shi',
    }

def timing(engine: enunu.ENUNU):
    print('timing: start')
    enunu.run_timing(engine=engine,)
    
    for path in (engine.path_full_timing, engine.path_mono_timing):
        if path is None or not os.path.isfile(path):
            raise Exception(f'{datetime.now()} :`{os.path.basename(path) if path else "None"}` does not exist.')
    print('timing: end')
    return {
        'path_full_timing': engine.path_full_timing,
        'path_mono_timing': engine.path_mono_timing,
    }


def acoustic(engine: enunu.ENUNU):
    print('acoustic: start')
    if not set_features(engine):
        enunu.run_timing(engine=engine,step='acoustic')
        enunu.run_acoustic(engine=engine)
        enunu.run_npy(engine=engine)
    print('acoustic: end')
    return {
        'path_f0': engine.path_f0_npy,
        'path_spectrogram': engine.path_spectrogram_npy,
        'path_aperiodicity': engine.path_aperiodicity_npy,
        'path_mel': engine.path_mel_npy,
        'path_vuv': engine.path_vuv_npy,
    }

def pitch(engine: enunu.ENUNU):
    """ピッチ (F0) だけを推定する。

    lf0_model を持つモデルは声色を計算しないので acoustic より大幅に速い。
    結果は pitch_f0.npy (f0.npy とは別) に保存する。
    """
    print('pitch: start')
    enunu.run_timing(engine=engine,step='acoustic')
    enunu.run_pitch(engine=engine)
    print('pitch: end')
    return {
        'path_f0': engine.path_pitch_npy,
        'lf0_conditioning': engine.supports_lf0_conditioning(),
    }

def acoustic_f0(engine: enunu.ENUNU, editor_f0: np.ndarray):
    """エディタのピッチを条件にして音響特徴量を作り直す。

    editor_f0: float64 配列 (Hz)。0 のフレームはモデル自身のピッチを使う。
    lf0_model を持たないモデルでは通常の acoustic と同じ結果になる。
    出力は acoustic と同じ (f0.npy は editor_f0 のピッチになる)。
    """
    print('acoustic_f0: start')
    enunu.run_timing(engine=engine,step='acoustic')
    enunu.run_acoustic(engine=engine,editor_f0=editor_f0)
    enunu.run_npy(engine=engine)
    print('acoustic_f0: end')
    return {
        'path_f0': engine.path_f0_npy,
        'path_spectrogram': engine.path_spectrogram_npy,
        'path_aperiodicity': engine.path_aperiodicity_npy,
        'path_mel': engine.path_mel_npy,
        'path_vuv': engine.path_vuv_npy,
        'lf0_conditioning': engine.supports_lf0_conditioning(),
    }

def synthe(out_wav_path: str,engine: enunu.ENUNU):
    print('synthe: start')
    set_features(engine)
    enunu.run_synthesizer(out_wav_path=out_wav_path,engine=engine)
    print('synthe: end')
    return {
        'path_wav': out_wav_path,
    }

def set_features(engine: enunu.ENUNU):
    return engine.multistream_features is not None

def poll_socket(socket, timetick = 100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass
    # Escape while loop if there's a keyboard interrupt.


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:15556')
    print('Started enunu server')


    support = False
    engine_dict = {}

    for message in poll_socket(socket):
        """
        request body
        request[0]:step,
        request[1]:ust_path,
        request[2]:wav_path,
        request[3]:singer_name,
        request[4]:duration,
        request[5]:style_shift
        """
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        engine,duration,request_time = None,600,None
        try:
            if request[0] == 'ver_check':
                support = True
                response['result'] = check()
            elif support:
                if request[3] in engine_dict:
                    engine,duration,request_time = engine_dict[request[3]]
                    enunu.updete_path(request[1],engine)
                else:
                    duration = int(request[4])
                    engine = enunu.setup(request[1])
                    request_time = time.time()
                    engine_dict[request[3]] = engine,duration,request_time


                if request[0] == 'timing':
                    response['result'] = timing(engine)
                elif request[0] == 'acoustic':
                    response['result'] = acoustic(engine)
                elif request[0] == 'pitch':
                    response['result'] = pitch(engine)
                elif request[0] == 'acoustic_f0':
                    editor_f0 = np.asarray(request[5], dtype=np.float64)
                    response['result'] = acoustic_f0(engine, editor_f0)
                elif request[0] == 'synthe':
                    response['result'] = synthe(request[2],engine)
                else:
                    raise NotImplementedError('unexpected command %s' % request[1])
            else:
                response['error'] = 'run ver_check.'
            
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()
        finally:
            keys_to_delete = [
                key for key, (_, duration, timestamp) in engine_dict.items()
                    if time.time() - timestamp > duration
            ]
            for key in keys_to_delete:
                print(key)
                del engine_dict[key]

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))


if __name__ == '__main__':
    main()
