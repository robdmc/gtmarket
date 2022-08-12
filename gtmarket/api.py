import importlib
import sys

"""
This module is a little funkey, but here's why it exists the way it does.

I want a module that lazy loads all the submodules.  It'll frequently be the case
that required libaries in those submodules are not installed, and I don't want to 
barf the whole package import just because those modules are not installed when all
I need a a tool that doesn't require them.

So the strategy here is to create a class with lazy-loading attributes that themselves
import the classes I want.

I could have tried to use getattr on a module as detailed here, but I wanted autocomplete
in jupyter, and couldn't figure out how to do that.
https://stackoverflow.com/questions/2447353/getattr-on-a-module
"""


class Importable:
    """
    This is a descriptor that knows how to load modules
    """
    def __init__(self, module_path, artifact_name):
        self._module_path = module_path
        self._artifact_name = artifact_name

    def __get__(self, instance, owner):
        module = importlib.import_module(self._module_path)
        return getattr(module, self._artifact_name)

    def __set__(self, instance, value):
        raise NotImplementedError('You cannot set this attribute')


class GTM:
    """
    Every object in the api is lazy loaded through a descriptor.  The actual
    attributes returned are classes.
    """
    OrderProducts = Importable('simbiz.live_opps', 'OrderProducts')
    OppLoader = Importable('simbiz.live_opps', 'OppLoader')
    AccountLoader = Importable('simbiz.live_opps', 'AccountLoader')
    OppHistoryLoader = Importable('simbiz.live_opps', 'OppHistoryLoader')
    PipeStats = Importable('simbiz.live_opps', 'PipeStats')
    ModelParams = Importable('simbiz.predictor.predictor', 'ModelParams')
    ModelParamsHist = Importable('simbiz.predictor.predictor', 'ModelParamsHist')
    SDRTeam = Importable('simbiz.predictor.predictor', 'SDRTeam')
    Deals = Importable('simbiz.predictor.predictor', 'Deals')

    # This allows from gtmarket.gtm import * to be used
    __all__ = list(set(vars().keys()) - {'__module__', '__qualname__', 'importlib', 'sys', 'Importable'})


# Instantiate a GTM object
gtm = GTM()

# Add it do the modules dict so that it can be imported with module syntax
sys.modules['gtmarket.gtm'] = gtm
__all__ = ['gtm']
