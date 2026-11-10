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
        Initialize the models class. Currently empty but can be extended for initialization needs.
        """
        pass
        self.gmm = gmm
        self.gmm.predict = predict_gmm
        self.gmm.plot = plot_gmm
        self.dpmm = dpmm(parent)
        self.bnnc = bnnc(parent)
        self.pca =  pca
        self.nbda = NBDA.model
        self.survival = survival(parent)

        self.available = {
            "gmm": self.gmm,
            "dpmm": self.dpmm,
            "bnnc": self.bnnc,
            "nbda": self.nbda,
            "pca": self.pca,
            "survival": self.survival
        }


    