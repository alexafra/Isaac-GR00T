from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.training.training_config import TrainingConfig


def _finetune_config(**kwargs) -> FinetuneConfig:
    return FinetuneConfig(
        base_model_path="/tmp/base-model",
        dataset_path="/tmp/dataset",
        embodiment_tag="NEW_EMBODIMENT",
        **kwargs,
    )


def test_duplicate_root_model_save_is_skipped_by_default() -> None:
    assert _finetune_config().skip_final_model_save is True
    assert TrainingConfig().skip_final_model_save is True


def test_duplicate_root_model_save_can_be_explicitly_enabled() -> None:
    assert _finetune_config(skip_final_model_save=False).skip_final_model_save is False
    assert TrainingConfig(skip_final_model_save=False).skip_final_model_save is False
