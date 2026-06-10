from mpdiffuser import *
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    dset = instantiate(cfg.dataset)
    model_cfg = OmegaConf.merge(cfg.model, {"nx": dset.nx, "nu": dset.nu})

    net = instantiate(model_cfg)
    print(net)

    trainer = Trainer(net, cfg)
    trainer.train(dset, dset)


if __name__ == "__main__":
    main()
