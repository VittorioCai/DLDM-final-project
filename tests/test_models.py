import pytest
import torch

from shortcut_learning.models import build_model


@pytest.mark.parametrize("name", ["lenet", "resnet18_mnist"])
def test_models_map_colored_mnist_to_ten_logits(name):
    model = build_model(name)
    inputs = torch.randn(2, 3, 28, 28)

    logits = model(inputs)
    features = model.extract_features(inputs)

    assert logits.shape == (2, 10)
    assert features.ndim == 2
    assert features.shape[1] == model.feature_dim


@pytest.mark.parametrize("name", ["lenet", "resnet18_mnist"])
def test_reset_classifier_replaces_only_the_classification_head(name):
    model = build_model(name)
    feature_parameters_before = [id(parameter) for parameter in model.feature_parameters()]
    classifier_before = id(model.classifier)

    model.reset_classifier()

    assert id(model.classifier) != classifier_before
    assert [id(parameter) for parameter in model.feature_parameters()] == feature_parameters_before
    assert model.classifier.in_features == model.feature_dim
    assert model.classifier.out_features == 10


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("transformer")
