class Importable:
    """
    This is a descriptor that knows how to load modules
    """
    def __init__(self, module_path, artifact_name, instantiate=False):
        self.instantiate = instantiate
        self._module_path = module_path
        self._artifact_name = artifact_name

    def __get__(self, instance, owner):
        # Import the module
        import importlib
        module = importlib.import_module(self._module_path)

        # Get the item (class/function/object) from the module
        thing = getattr(module, self._artifact_name)

        # If an object was requested, instantiate it
        if self.instantiate:
            thing = thing()
        return thing

    def __set__(self, instance, value):
        raise NotImplementedError('You cannot set this attribute')


class API:
    """
    Every object in the api is lazy loaded through a descriptor.  The actual
    attributes returned are classes/functions unless the instantiate flag is set.

    The lazy loading ensures that you only run imports if they are needed.
    """
    OrderProducts = Importable('gtmarket.core', 'OrderProducts')
    OppLoader = Importable('gtmarket.core', 'OppLoader')
    AccountLoader = Importable('gtmarket.core', 'AccountLoader')
    OppHistoryLoader = Importable('gtmarket.core', 'OppHistoryLoader')
    PipeStats = Importable('gtmarket.core', 'PipeStats')
    ModelParams = Importable('gtmarket.predictor', 'ModelParams')
    ModelParamsHist = Importable('gtmarket.predictor', 'ModelParamsHist')
    SDRTeam = Importable('gtmarket.predictor', 'SDRTeam')
    Deals = Importable('gtmarket.predictor', 'Deals')
