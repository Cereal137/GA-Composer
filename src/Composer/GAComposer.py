from src.modules.Translator import SingleChromosome_StreamTranslator, PitchOnlyTranslator, ChordOnlyTranslator
from src.modules.GA import SingleChromosome_GARunner

from music21 import note, clef, meter, stream

from typing import Dict, Optional
import random
import os


class GAComposer:
    def __init__(self, code_type: str, translator_args_dict: Dict, ga_args_dict: Dict) -> None:
        if code_type == 'single':
            self.translator = SingleChromosome_StreamTranslator(**translator_args_dict)
            self.ga_runner = SingleChromosome_GARunner(encode_dim=self.translator.encode_dim, max_encode=self.translator.max_encode, obj_arg_info=self.translator.obj_arg_info, **ga_args_dict)
        elif code_type == 'pitchOnly':
            self.translator = PitchOnlyTranslator(**translator_args_dict)
            self.ga_runner = SingleChromosome_GARunner(encode_dim=self.translator.encode_dim, max_encode=self.translator.max_encode, obj_arg_info=self.translator.obj_arg_info, **ga_args_dict)
        elif code_type == 'chordOnly':
            self.translator = ChordOnlyTranslator(**translator_args_dict)
            self.ga_runner = SingleChromosome_GARunner(encode_dim=self.translator.encode_dim, max_encode=self.translator.max_encode, obj_arg_info=self.translator.obj_arg_info, **ga_args_dict)
           
        # TODO: implement other code type for multi-chromosome
        else:
            raise NotImplementedError('Code type {} is not implemented'.format(code_type))
        
    def compose(self, job_name: str, out_path: str, sample_num: Optional[int] = None, **kwargs):
        solution = self.ga_runner.run(**kwargs)
        if sample_num is not None:
            s_pop = solution['lastPop']
            assert sample_num <= s_pop.Phen.shape[0], 'sample_num should be less than the population size'
            samples = random.sample(list(s_pop.Phen), sample_num)
            assert os.path.isdir(out_path), '{} is not a directory or does not exist'.format(out_path)
            out_dir = os.path.join(out_path, job_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for i, sample in enumerate(samples):
                my_stream = self.translator.decode(sample)
                final_stream = stream.Stream([clef.bestClef(my_stream), meter.TimeSignature(self.translator.signature)]) + my_stream
                final_stream.write('midi', fp=os.path.join(out_dir, 'mus_{}.mid'.format(i)))
                final_stream.write('lily.pdf', fp=os.path.join(out_dir, 'mus_{}'.format(i)))
        else:
            s_pop = solution['optPop']
            my_stream = self.translator.decode(s_pop.Phen[0])
            final_stream = stream.Stream([clef.bestClef(my_stream), meter.TimeSignature(self.translator.signature)]) + my_stream
            my_stream.write('midi', fp=out_path)
            my_stream.write('lily.pdf', fp=out_path)
