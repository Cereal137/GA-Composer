from typing import Any
import numpy as np
from music21.stream import Stream
from music21.note import Note, Rest
from music21.chord import Chord
from music21.duration import Duration
from src.utils.utils import _get_available_pitches
import abc


class Translator(metaclass=abc.ABCMeta):
     
    # These abstract methods and attributes must be implemented cuz they are used in Composer.GAComposer
    @abc.abstractmethod
    def decode(self):
        pass

    

class SingleChromosome_StreamTranslator(Translator):
    '''
        Encode a stream to a numpy array.
        The SimpleEncoder ony support single track without any chords.
        A shortest duration is encode follow the rules below:
            Rest: 0
            Note: pitch_lowest ~ pitch_highest
            Fermata: pitch_highest + 1
        The quarter duration is treated as 1.0.
    '''
    def __init__(self, pitch_lowest: str, pitch_highest: str, bar_num: int, signature: str, min_quarter_duration: float = 0.25) -> None:
        '''Initialize the SimpleEncoder.

        Args:
            pitch_lowest (str): The lowest pitch in the note set.
            pitch_highest (str): The lowest pitch in the note set.
            bar_num (int): The bar num that once the encoder could encode.
            signature(str): The time signature of the encoder.
            min_quarter_duration (float, optional): The shortest duration of the encoder, where a signature dominator's duration is 1. Defaults to 0.25 (16th in x/4 signature).
        '''
        avl_pitches = _get_available_pitches(pitch_lowest, pitch_highest)
        self.bar_num = bar_num
        self.signature = signature
        self.bar_beat, self.beat_duration = map(int, signature.split('/'))
        # Convert beat duration in the unit of quarter. 
        self.beat_duration = 4 / self.beat_duration
        self.min_quarter_duration = min_quarter_duration
        # Encode note and rest.
        self.pitch2idx = {pitch: idx+1 for idx, pitch in enumerate(avl_pitches)}
        self.idx2pitch = {idx+1: pitch for idx, pitch in enumerate(avl_pitches)}
        self.fermata_idx = len(self.pitch2idx) + 1

        # Calculate the length of the encoded array.
        self.encode_dim = self.bar_num * round(self.bar_beat * self.beat_duration) * round(1/min_quarter_duration)
        # Upper bound of the elementin encoded array.
        self.max_encode = self.fermata_idx
        self.max_encode = self.fermata_idx**3 #for chords
        self.obj_arg_info = {'fermata_code': self.max_encode, 'rest_code': 0, 'translator': self}

    def encode(self, stream: Stream, *args: Any, **kwds: Any) -> np.ndarray:
        stream_arr = np.zeros(self.length, dtype=np.int8)
        from_idx = 0
        for note in stream:
            length = round(note.duration.quarterLength / self.min_quarter_duration)
            if note.isRest:
                continue
            stream_arr[from_idx:from_idx+length] = self.fermata_idx
            stream_arr[from_idx] = self.pitch2idx[note.pitch.nameWithOctave]
            
            from_idx += length

        return stream_arr
    
    def decode(self, stream_arr: np.ndarray) -> Stream:
        ret_stream = Stream()
        for code in stream_arr:
            if code == 0:
                ret_stream.append(Rest(length=self.min_quarter_duration))
            elif code == self.fermata_idx:
                if len(ret_stream) == 0:
                    ret_stream.append(Rest(length=self.min_quarter_duration))
                ret_stream[-1].quarterLength += self.min_quarter_duration
            else:
                new_note = Note(pitch=self.idx2pitch[code], quarterLength=self.min_quarter_duration)
                ret_stream.append(new_note)

        return ret_stream



class PitchOnlyTranslator(Translator):
    '''
        Encode a stream to a numpy array.
        Single track without any chords.
        A shortest duration is encode follow the rules below:
            Rest: 0
            Note: pitch_lowest ~ pitch_highest
    '''
    def __init__(self, pitch_lowest: str, pitch_highest: str, bar_num: int, signature: str, quarter_duration: float = 0.25) -> None:
        '''Initialization.

        Args:
            pitch_lowest (str): The lowest pitch in the note set.
            pitch_highest (str): The lowest pitch in the note set.
            bar_num (int): The bar num that once the encoder could encode.
            signature(str): The time signature of the encoder.
            quarter_duration (float, optional): The duration of the encoder, where a signature dominator's duration is 1. Defaults to 0.25 (16th in x/4 signature).
        '''
        avl_pitches = _get_available_pitches(pitch_lowest, pitch_highest)
        self.bar_num = bar_num
        self.signature = signature
        self.bar_beat, self.beat_duration = map(int, signature.split('/'))
        # Convert beat duration in the unit of quarter. 
        self.beat_duration = 4 / self.beat_duration
        self.quarter_duration = quarter_duration
        # Encode note and rest.
        self.pitch2idx = {pitch: idx+1 for idx, pitch in enumerate(avl_pitches)}
        self.idx2pitch = {idx+1: pitch for idx, pitch in enumerate(avl_pitches)}

        # Calculate the length of the encoded array.
        self.encode_dim = self.bar_num * round(self.bar_beat * self.beat_duration) * round(1/quarter_duration)
        # Upper bound of the elementin encoded array.
        self.max_encode = len(self.pitch2idx)
        self.obj_arg_info = {'rest_code': 0, 'translator': self}
        
    def decode(self, stream_arr: np.ndarray) -> Stream:
        ret_stream = Stream()
        for code in stream_arr:
            if code == 0:
                ret_stream.append(Rest(length=self.quarter_duration))
            else:
                new_note = Note(pitch=self.idx2pitch[code], quarterLength=self.quarter_duration)
                ret_stream.append(new_note)

        return ret_stream

class ChordOnlyTranslator(Translator):
    """
        Encode a stream to a numpy array.
        only chords.
        A shortest duration is encode follow the rules below:
            Rest: 0
            Note: pitch_lowest ~ pitch_highest
    """
    def __init__(self, pitch_lowest: str, pitch_highest: str, bar_num: int, signature: str, quarter_duration: float = 0.25) -> None:
        '''Initialization.

        Args:
            pitch_lowest (str): The lowest pitch in the note set.
            pitch_highest (str): The lowest pitch in the note set.
            bar_num (int): The bar num that once the encoder could encode.
            signature(str): The time signature of the encoder.
            quarter_duration (float, optional): The duration of the encoder, where a signature dominator's duration is 1. Defaults to 0.25 (16th in x/4 signature).
        '''
        avl_pitches = _get_available_pitches(pitch_lowest, pitch_highest)
        self.bar_num = bar_num
        self.signature = signature
        self.bar_beat, self.beat_duration = map(int, signature.split('/'))
        # Convert beat duration in the unit of quarter. 
        self.beat_duration = 4 / self.beat_duration
        self.quarter_duration = quarter_duration
        # Encode note and rest.
        self.pitch2idx = {pitch: idx+1 for idx, pitch in enumerate(avl_pitches)}
        self.idx2pitch = {idx+1: pitch for idx, pitch in enumerate(avl_pitches)}

        # Calculate the length of the encoded array.
        self.encode_dim = self.bar_num * round(self.bar_beat * self.beat_duration) * round(1/quarter_duration)
        # Upper bound of the elementin encoded array.
        self.max_encode = len(self.pitch2idx)
        self.obj_arg_info = {'rest_code': 0, 'translator': self}
        
    def decode(self, stream_arr: np.ndarray) -> Stream:
        ret_stream = Stream()
        for code in stream_arr:
            
            if code == 0:
                ret_stream.append(Rest(length=self.quarter_duration))
            else:
                code1 = code
                code3 = code + 4
                code5 = code + 7
                if code3 > len(self.pitch2idx):
                    code3 -= 12 
                    code5 -= 12
                elif code5 > len(self.pitch2idx):
                    code5 -= 12
                chord_in = [self.idx2pitch[code1],self.idx2pitch[code3],self.idx2pitch[code5]]
                new_chord = Chord(chord_in, quarterLength=self.quarter_duration)
                ret_stream.append(new_chord)
        """
            for code in stream_arr:
                code1 = code % len(self.pitch2idx)
                code3 = code // len(self.pitch2idx) % len(self.pitch2idx)
                code5 = code // len(self.pitch2idx) // len(self.pitch2idx) % len(self.pitch2idx)
                chord_in = [code1,code3,code5]
                new_chord = Chord(chord_in, quarterLength=self.quarter_duration)
                ret_stream.append(new_chord)
        """
        
        return ret_stream