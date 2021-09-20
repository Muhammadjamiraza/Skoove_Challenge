import os
import sys

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colors
import pretty_midi
import pandas as pd
import IPython.display as ipd
import mir_eval
import madmom
from madmom.features import notes
from madmom.features.notes import CNNPianoNoteProcessor
from madmom.features.notes import ADSRNoteTrackingProcessor
import numpy as np
import pickle
import glob
 
sys.path.append('..')
data_path = os.path.join('/home/mura4197/Downloads/datasets/')
file_path = os.path.join(data_path, 'pitch_ranges')







proc = CNNPianoNoteProcessor()
adsr = ADSRNoteTrackingProcessor()
midi_list = []
audio_list = []
results = []
results_all = []
file_list = []
precision = []
recall = []
F_measure = []
Average_Overlap_Ratio = []
Precision_no_offset = []
recall_no_offset = []
F_measure_no_offset = []
Average_Overlap_Ratio_no_offset  = []
Onset_Precision = []
Onset_Recall = []
Onset_F_measure = []
Offset_Precision = []
Offset_Recall = []
Offset_F_measure = []





csv_files = glob.glob(os.path.join(data_path, "*.csv"))

for f in csv_files:
    metadata = pd.read_csv(f, sep = '\t|,' , engine='python')
    print(f)
      
    for index, row in metadata.iterrows():
        file_0 = data_path + "/" + str(row["midi_filename"])
        file_1 = data_path + "/" + str(row["audio_filename"])
        file_list.append(file_1)
        midi_data = pretty_midi.PrettyMIDI(file_0)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                pitch = note.pitch
                end = note.end
                midi_list.append([start, end, pitch])
        df = pd.DataFrame(midi_list)
        df.to_csv('reference.txt', sep='\t', index=False)
        with open('reference.txt', "r") as g:
            rows = g.readlines()[1:]
            ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals('reference.txt')
        act = proc(file_1)
        adsr_1 = adsr(act)
        
        for audio in adsr_1:
            start = audio[0]
            
            end = audio[2] + audio[0]
            pitch = audio[1]
            
            audio_list.append([start, end, pitch])
        df_1 = pd.DataFrame(audio_list)           
        df_1.to_csv('estimate.txt', sep='\t', index=False)
        with open('estimate.txt', "r") as h:
            rows = h.readlines()[1:]
            est_intervals, est_pitches = mir_eval.io.load_valued_intervals('estimate.txt')
        scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches,est_intervals, est_pitches)
        values = scores.values()
        values_list = list(values)
        results.append(values_list)
        open('reference.txt', 'w').close()
        open('estimate.txt', 'w').close()
    results = np.array(results, dtype = 'object').T.tolist() 
    def extractDigits(lst):
        return [[el] for el in lst]
    result = extractDigits(results)
    #print(type(result))
    precision.append(result[0])
    recall.append(result[1])
    F_measure.append(result[2])
    Average_Overlap_Ratio.append(result[3])
    Precision_no_offset.append(result[4])
    recall_no_offset.append(result[5])
    F_measure_no_offset.append(result[6])
    Average_Overlap_Ratio_no_offset.append(result[7])
    Onset_Precision.append(result[8])
    Onset_Recall.append(result[9])
    Onset_F_measure.append(result[10])
    Offset_Precision.append(result[11])
    Offset_Recall.append(result[12])
    Offset_F_measure.append(result[13])

precision = np.array(precision)
precision = precision.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Precision')
ax1.boxplot(precision)
plt.show()

recall = np.array(recall)
recall = recall.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('recall')
ax1.boxplot(recall)
plt.show()

F_measure = np.array(F_measure)
F_measure = F_measure.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('F_measure')
ax1.boxplot(F_measure)
plt.show()

Average_Overlap_Ratio = np.array(Average_Overlap_Ratio)
Average_Overlap_Ratio = Average_Overlap_Ratio.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Average_Overlap_Ratio')
ax1.boxplot(Average_Overlap_Ratio)

Precision_no_offset = np.array(Precision_no_offset)
Precision_no_offset = Precision_no_offset.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Precision_no_offset')
ax1.boxplot(Precision_no_offset)
plt.show()

recall_no_offset = np.array(recall_no_offset)
recall_no_offset = recall_no_offset.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('recall_no_offset')
ax1.boxplot(recall_no_offset)
plt.show()

F_measure_no_offset = np.array(F_measure_no_offset)
F_measure_no_offset = F_measure_no_offset.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('F_measure_no_offset')
ax1.boxplot(F_measure_no_offset)
plt.show()

Average_Overlap_Ratio_no_offset = np.array(Average_Overlap_Ratio_no_offset)
Average_Overlap_Ratio_no_offset = Average_Overlap_Ratio_no_offset.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Average_Overlap_Ratio_no_offset')
ax1.boxplot(Average_Overlap_Ratio_no_offset)
plt.show()


Onset_Precision = np.array(Onset_Precision)
Onset_Precision = Onset_Precision.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Onset_Precision')
ax1.boxplot(Onset_Precision)
plt.show()


Onset_Recall = np.array(Onset_Recall)
Onset_Recall = Onset_Recall.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Onset_Recall')
ax1.boxplot(Onset_Recall)
plt.show()

Onset_F_measure = np.array(Onset_F_measure)
Onset_F_measure = Onset_F_measure.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Onset_F_measure')
ax1.boxplot(Onset_F_measure)
plt.show()

Offset_Precision = np.array(Offset_Precision)
Offset_Precision = Offset_Precision.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Offset_Precision')
ax1.boxplot(Offset_Precision)
plt.show()


Offset_Recall = np.array(Offset_Recall)
Offset_Recall = Offset_Recall.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Offset_Recall')
ax1.boxplot(Offset_Recall)
plt.show()         


Offset_F_measure = np.array(Offset_F_measure)
Offset_F_measure = Offset_F_measure.reshape(20,6)
fig1, ax1 = plt.subplots()
ax1.set_title('Offset_F_measure')
ax1.boxplot(Offset_F_measure)
plt.show() 
        
        

















