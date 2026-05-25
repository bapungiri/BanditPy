def __getattr__(name):
    if name == 'Qlearn2Arm':
        from .qlearn import Qlearn2Arm
        return Qlearn2Arm
    if name == 'Logistic2Arm':
        from .regression_models import Logistic2Arm
        return Logistic2Arm
    if name == 'BanditTrainer2Arm':
        from .rnn_models import BanditTrainer2Arm
        return BanditTrainer2Arm
    if name == 'Thompson2Arm':
        from .thompson_models import Thompson2Arm
        return Thompson2Arm
    if name == 'DecisionModel':
        from .model import DecisionModel
        return DecisionModel
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
