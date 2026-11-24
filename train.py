from diff_env.simulator import DiffSimulator
from training.toy import ToyMLPPolicy
from training.trainer import TrainConfig, Trainer


def main():
    sim = DiffSimulator(policy=ToyMLPPolicy())
    cfg = TrainConfig(num_steps=2000, log_interval=50, run_tag="toy")
    Trainer(sim, config=cfg).train()


if __name__ == "__main__":
    main()