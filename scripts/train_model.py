import sys
import os
from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.utils import build_llm

llm, llm_tokenizer = build_llm("gpt-3")

modelName = 'speechBaseline4'
args = {}
args['outputDir'] = 'speech_logs/' + modelName
args['datasetPath'] = 'ptDecoder_ctc/dataset.pkl'  # Update the path to match the remote path in the container
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 5000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args, llm, llm_tokenizer)