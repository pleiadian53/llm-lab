"""Minimal entry point that exercises the training stack."""

from llm_lab.pretrain_llm import LanguageModelTrainer, PretrainingConfig


def main() -> None:
    config = PretrainingConfig()
    config.trainer.num_train_steps = 5
    trainer = LanguageModelTrainer.from_config(config)
    trainer.run()


if __name__ == "__main__":
    main()
