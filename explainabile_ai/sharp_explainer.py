import os
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt


class SHAPExplainer:

    def __init__(self, model, background_data):

        self.model = model

        self.model.eval()

        if isinstance(background_data, torch.Tensor):
            self.background = background_data.float()
        else:
            self.background = torch.tensor(
                background_data,
                dtype=torch.float32
            )

    # -------------------------------------------------------

    def predict(self, x):

        if isinstance(x, np.ndarray):

            x = torch.tensor(
                x,
                dtype=torch.float32
            )

        with torch.no_grad():

            output = self.model(x)

        return output.numpy()

    # -------------------------------------------------------

    def explain(self, sample):

        if isinstance(sample, torch.Tensor):

            sample = sample.numpy()

        explainer = shap.KernelExplainer(

            self.predict,

            self.background[:100]

        )

        shap_values = explainer.shap_values(

            sample,

            nsamples=100

        )

        return shap_values

    # -------------------------------------------------------

    def summary_plot(

        self,

        shap_values,

        sample,

        feature_names,

        save_path=None

    ):

        shap.summary_plot(

            shap_values,

            sample,

            feature_names=feature_names,

            show=False

        )

        if save_path:

            plt.savefig(

                save_path,

                dpi=300,

                bbox_inches="tight"

            )

        plt.close()

    # -------------------------------------------------------

    def waterfall_plot(

        self,

        shap_values,

        sample,

        feature_names

    ):

        shap.plots._waterfall.waterfall_legacy(

            explainer_expected_value=0,

            shap_values=shap_values[0],

            features=sample[0],

            feature_names=feature_names

        )

    # -------------------------------------------------------

    def feature_importance(

        self,

        shap_values,

        feature_names

    ):

        importance = np.abs(

            shap_values

        ).mean(axis=0)

        ranking = sorted(

            zip(feature_names, importance),

            key=lambda x: x[1],

            reverse=True

        )

        return ranking