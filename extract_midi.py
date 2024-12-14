import pretty_midi
import numpy as np
import os

# If multiple instruments are in MIDI file and music21 cannot parse effectively, this function uses pretty_midi library to extract one piano track from each file and save it in a different directory with the same file name.

def convert(inp_dir, out_dir, interval=100):
    paths = []
    for dirpath, dirnames, filenames in os.walk(inp_dir):
        if not os.path.isdir(os.path.join(out_dir, dirpath)):
            os.makedirs(os.path.join(out_dir, dirpath))
        for filename in filenames:
            if filename[-4:].lower() == '.mid':
                paths.append(os.path.join(dirpath, filename))
    if inp_dir[-1] != '/':
        inp_dir += '/'
    
    f = 0
    for i,p in enumerate(paths):
        if i % interval == 0:
            print(i, p)
        try:
            midi = pretty_midi.PrettyMIDI(p)
            piano_instruments = [inst for inst in midi.instruments if inst.is_drum == False and inst.program in range(0, 6)]
            if len(piano_instruments) > 0:
                counts = [len(instrument.notes) for instrument in piano_instruments]
                best_idx = np.argmax(counts)
                best_piano = piano_instruments[best_idx]
                piano_midi = pretty_midi.PrettyMIDI()
                piano_midi.instruments.append(best_piano)
                piano_midi.write(os.path.join(out_dir, p))
            else:
                f += 1
        except:
            f += 1

    print(f'FAILED: {f}')
    return paths

