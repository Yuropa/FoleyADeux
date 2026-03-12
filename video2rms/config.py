import numpy as np
import librosa
from yacs.config import CfgNode as CN

_C = CN()

### Data Config
_C.data = CN()
_C.data.video_samples     = 300
_C.data.audio_samples     = 10
_C.data.audio_sample_rate = 16000
_C.data.rms_nframes       = 512
_C.data.rms_hop           = 128
_C.data.envelope_decay    = 0.9995
_dummy_audio = np.pad(
    np.zeros(_C.data.audio_samples * _C.data.audio_sample_rate),
    (int((_C.data.rms_nframes - _C.data.rms_hop) / 2),
     int((_C.data.rms_nframes - _C.data.rms_hop) / 2)),
    mode="reflect",
)
_C.data.rms_samples   = int(
    librosa.feature.rms(
        y=_dummy_audio,
        frame_length=_C.data.rms_nframes,
        hop_length=_C.data.rms_hop,
        center=False,
        pad_mode="reflect",
    ).shape[1]
)
_C.data.rms_discretize     = True
_C.data.rms_num_bins       = 64
_C.data.rms_mu             = _C.data.rms_num_bins - 1
_C.data.rms_min            = 0.01
_C.data.onset_supervision  = False   # inference: no onset branch

### Train Config — minimal stubs required to instantiate Video2Sound at inference time
_C.train = CN()
_C.train.onset_supervision = False
_C.train.loss              = CN()
_C.train.loss.type         = "CE"   # model was trained with discretised CE loss
_C.train.lr                = 1e-4
_C.train.beta1             = 0.9

### Model Config
_C.model = CN()
_C.model.encoder_embedding_dim  = 2048
_C.model.encoder_kernel_size    = 5
_C.model.encoder_n_convolutions = 3
_C.model.encoder_n_lstm         = 2
