import functools

from BayesForge.Models.surv import survival
from BayesForge.NBDA.NBDA import NBDA
from BayesForge.Models.GMM import *
from BayesForge.Models.DPMM import dpmm
from BayesForge.Models.BNNC import bnnc
from BayesForge.Models.PCA import *


class models():
    """
    The models class serves as a high-level interface for managing and utilizing various Bayesian models within the BF framework. It encapsulates different model types such as DPMM, GMM, NBDA, and others, providing a unified structure for model initialization, fitting, and diagnostics. This class is designed to simplify the process of working with complex Bayesian models, allowing users to easily switch between different model types and access their functionalities through a consistent API.
    """
    def __init__(self,parent):
        """
        Initialize the models class and bind every entry to ``parent``.

        The entries of ``self.available`` are heterogeneous; how to use each:

        - ``gmm``: function, call it — ``m.fit(m.models.gmm, obs={'data': X,
          'K': 3})``. ``.predict`` and ``.plot`` shortcuts are attached to this
          instance's wrapper only, never to ``BayesForge.Models.GMM.gmm``.
        - ``pca``: class, instantiate it — ``m.fit(m.models.pca(type='classic'))``.
        - ``nbda``: class, build the object then fit its method —
          ``nbda = m.models.nbda(network, status); m.fit(nbda.model)``.
        - ``dpmm`` and ``bnnc``: already-built instances bound to this parent;
          use them as they are, do not call them.
        - ``survival``: the survival instance bound to this parent; import data
          on it and fit its model — ``m.fit(m.models.survival.model)``.
        """
        self.parent = parent

        # Per-instance wrapper around the gmm function: attaching predict/plot
        # to gmm itself would add class-level attributes to the module object
        # and leak the shortcuts into every other user of the GMM module.
        @functools.wraps(gmm)
        def gmm_model(*args, **kwargs):
            return gmm(*args, **kwargs)

        gmm_model.predict = predict_gmm
        gmm_model.plot = plot_gmm
        self.gmm = gmm_model
        self.dpmm = dpmm(parent)
        self.bnnc = bnnc(parent)
        self.pca =  pca
        self.nbda = NBDA
        self.survival = survival(parent)

        self.available = {
            "gmm": self.gmm,
            "dpmm": self.dpmm,
            "bnnc": self.bnnc,
            "nbda": self.nbda,
            "pca": self.pca,
            "survival": self.survival
        }
