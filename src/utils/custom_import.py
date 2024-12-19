import inspect

def custom_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

ARCHITECTURE = {
    'loss': {
        'module': 'losses',
        'provider': 'LossProvider',
        'provider_short': 'LP'
    },
    'model': {
        'module': 'models',
        'provider': 'ModelProvider',
        'provider_short': 'MP'
    },
    'dataset': {
        'module': 'datasets',
        'provider': 'DatasetProvider',
        'provider_short': 'DP'
    },
    'activation': {
        'module': 'activations',
        'provider': 'ActivationProvider',
        'provider_short': 'AP'
    }
}

def check(architectural_module: str):
    if architectural_module not in ARCHITECTURE.keys():
        raise ModuleNotFoundError(f'Architecture module not found: {architectural_module}')
    return ARCHITECTURE[architectural_module]

def custom_import_class(class_name: str, architectural_module: str):
    mod = check(architectural_module)
    return custom_import(f"{mod['module']}.{class_name}_{mod['provider_short']}")

def get_possible_classes(architectural_module: str):
    mod = check(architectural_module)
    ds = __import__(mod['module'])
    return [ name.replace(f"_{mod['provider_short']}", "") for (name, _) \
            in inspect.getmembers(ds, inspect.isclass) if name != mod['provider']]
