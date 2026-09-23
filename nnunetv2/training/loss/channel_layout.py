"""
Single source of truth for what each channel of the vein-segmentation ``data``
tensor holds, and for the domain index that marks a case as non-QSM.

``data`` is always the 3-channel tensor produced by the dataloader
(primary image / local field / Frangi vesselness prior) — this layout does
not change whether the primary image is a QSM susceptibility map or an R2*
map. The network itself only ever sees ``data[:, CH_PRIMARY:CH_PRIMARY+1]``;
the other two channels exist purely to feed the training-time losses.
"""
from nnunetv2.training.network_architecture.unet_with_attention import DOMAIN_METHODS

CH_PRIMARY = 0     # QSM chi map OR R2* map, depending on the case
CH_LOCALFIELD = 1  # measured local field (ppm); zero-filled for R2* cases
CH_FRANGI = 2       # precomputed vesselness prior, computed from CH_PRIMARY

# The "prior" channels: never fed to the network as extra input channels,
# only used (a) to compute the training-time gate that modulates CH_PRIMARY
# before it enters the encoder (see PriorGatedSingleChannelUNet), and
# (b) directly by the physics/Frangi losses.
PRIOR_CHANNELS = slice(CH_LOCALFIELD, CH_FRANGI + 1)
N_PRIORS = CH_FRANGI - CH_LOCALFIELD + 1

R2STAR_DOMAIN_IDX = DOMAIN_METHODS.index('R2star')
