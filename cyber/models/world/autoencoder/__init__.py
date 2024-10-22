import sys
# get the directoy of this file
# add it to the system path
# so that we can import modules from this directory
# TODO: this is a hack, find a better way to do this
sys.path.append(__file__.rsplit("/", 1)[0])
from cyber.models.world.autoencoder.magvit2.models.lfqgan import VQModel as VQModel
