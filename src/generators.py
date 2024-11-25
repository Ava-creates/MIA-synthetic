### add generators
from abc import ABC
import torch
import sys

# from reprosyn.methods import DS_PRIVBAYES, SYNTHPOP, DS_BAYNET, DS_INDHIST
sys.path.append('dp-ctgans/src')

from ctgan import CTGAN
from dp_cgans import DP_CGAN

class Generator(ABC):
    """Base class for generators"""
    def __init__(self):
        self.trained = False

    @property
    def label(self):
        return "Unnamed Generator"

    def __str__(self):
        return self.label

class identity(Generator):
    """This generator is the identity generator: just return the input dataset."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        return dataset

    @property
    def label(self):
        return "identity"

class baynet(Generator):
    """This generator is based on BAYNET."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        baynet = DS_BAYNET(dataset=dataset, metadata=metadata, size=size, seed = seed)
        baynet.run()
        return baynet.output

    @property
    def label(self):
        return "BAYNET"

class privbayes(Generator):
    """This generator is based on privbayes."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        pbayes = DS_PRIVBAYES(dataset=dataset, metadata=metadata, size=size,
                           epsilon=self.epsilon, seed = seed)
        pbayes.run()
        return pbayes.output

    @property
    def label(self):
        return "privbayes"

class ctgan(Generator):
    """This generator is based on CTGAN."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed, epochs = 50):

        torch.manual_seed(seed)
        ctgan = CTGAN(epochs)
        print(metadata)
        discrete_columns = [entry['name'] for entry in metadata if entry['type'] == 'finite']
        print(discrete_columns)
        #metadata needs to be discrete columns 
        ctgan.fit(dataset, discrete_columns)
        # Create synthetic data
        synthetic_data = ctgan.sample(size)
        # ctgan = CTGAN(dataset=dataset, metadata=metadata, size=size, epochs = epochs)
        # ctgan.run()
        return synthetic_data

    @property
    def label(self):
        return "CTGAN"
    

class dpctgan(Generator):
    """This generator is based on CTGAN."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed, epochs = 50):

        torch.manual_seed(seed)
        model = DP_CGAN(
                epochs=epochs, # number of training epochs
                batch_size=128, # the size of each batch
                log_frequency=True,
                verbose=True,
                generator_dim=(128, 128, 128),
                discriminator_dim=(128, 128, 128),
                generator_lr=2e-4, 
                discriminator_lr=2e-4,
                discriminator_steps=1, 
                private=False,
                )
        model.fit(dataset)

        synthetic_data = model.sample(size)

        return synthetic_data

    @property
    def label(self):
        return "DPCTGAN"


class synthpop(Generator):
    """This generator is based on SYNTHPOP."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        spop = SYNTHPOP(dataset=dataset, metadata=metadata, size=size, seed = seed)
        spop.run()
        return spop.output

    @property
    def label(self):
        return "SYNTHPOP"

class indhist(Generator):
    """This generator is based on INDHIST."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        indhist = DS_INDHIST(dataset=dataset, metadata=metadata, size=size)
        indhist.run()
        return indhist.output

    @property
    def label(self):
        return "INDHIST"

def get_generator(name_generator: str, epsilon: float):
    if name_generator == 'identity':
        return identity()
    elif name_generator == 'BAYNET':
        return baynet()
    elif name_generator == 'privbayes':
        return privbayes(epsilon)
    elif name_generator == 'CTGAN':
        return ctgan()
    elif name_generator == 'SYNTHPOP':
        return synthpop()
    elif name_generator == 'INDHIST':
        return indhist()
    elif name_generator == 'DPCTGAN':
        return dpctgan()
    else:
        print('Not a valid generator.')
