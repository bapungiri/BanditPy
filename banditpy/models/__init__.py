from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DecisionModel
    from .regression_models import Logistic2Arm
    from .rnn_fit import VanillaRNNFit2Arm, VanillaRNNModel, VanillaRNNTrainer2Arm
    from .rnn_models import BanditTrainer2Arm

__all__ = [
    "BanditTrainer2Arm",
    "DecisionModel",
    "Logistic2Arm",
    "VanillaRNNFit2Arm",
    "VanillaRNNModel",
    "VanillaRNNTrainer2Arm",
]


def __getattr__(name):
    if name == "Logistic2Arm":
        from .regression_models import Logistic2Arm

        return Logistic2Arm
    if name == "BanditTrainer2Arm":
        from .rnn_models import BanditTrainer2Arm

        return BanditTrainer2Arm
    if name == "DecisionModel":
        from .model import DecisionModel

        return DecisionModel
    if name == "VanillaRNNTrainer2Arm":
        from .rnn_fit import VanillaRNNTrainer2Arm

        return VanillaRNNTrainer2Arm
    if name == "VanillaRNNModel":
        from .rnn_fit import VanillaRNNModel

        return VanillaRNNModel
    if name == "VanillaRNNFit2Arm":
        from .rnn_fit import VanillaRNNFit2Arm

        return VanillaRNNFit2Arm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
