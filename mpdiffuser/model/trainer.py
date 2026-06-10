import os
import logging
import wandb
import os.path as op
from tqdm import tqdm
from tensorboardX import SummaryWriter
from mpdiffuser.utils import accumulate_metrics, create_iter_dataset
import jax.numpy as jnp

import jax
from flax.serialization import to_state_dict, from_state_dict
import orbax.checkpoint as ocp
from functools import partial
from omegaconf import OmegaConf
from mpdiffuser.model.networks import BaseModel, TrainState
import zipfile
import yaml
import coloredlogs

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: BaseModel, cfg, output_dir=None, verbose=1):
        self.model = model
        self.cfg = cfg
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            if self.cfg.resume:
                self.output_dir = self.get_model_identifier()
            else:
                self.output_dir = self.get_output_dir()
        self.verbose = verbose
        if cfg.save_every_n_epochs > 0:
            folder = op.join(self.output_dir, 'checkpoints')
            self.ckptr = self.get_ckpt_manager(folder)

    def get_output_dir(self):
        identifier = output_dir = self.get_model_identifier()
        num = 1
        while os.path.exists(output_dir):
            output_dir = f"{identifier}-{num}"
            num += 1
        return output_dir

    def setup_train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Write source code to output dir
        self.write_file_contents(self.output_dir)

        if self.cfg.use_wandb:
            self.setup_wandb()
        self.logger = SummaryWriter(self.output_dir)

        # Log messages to file
        if self.verbose:
            coloredlogs.install(
                datefmt='%d/%m %H:%M:%S.%f',
                fmt='%(asctime)s %(levelname)s %(message)s',
                level='INFO',
            )
            root_logger = logging.getLogger()
            num = 1
            msg_file = op.join(self.output_dir, 'messages.log')
            while os.path.exists(msg_file):
                msg_file = op.join(self.output_dir, f'messages_{num}.log')
                num += 1
            file_handler = logging.FileHandler(msg_file)
            file_handler.setFormatter(root_logger.handlers[0].formatter)
            for handler in root_logger.handlers[1:]:  # all except stdout
                root_logger.removeHandler(handler)
            root_logger.addHandler(file_handler)

    def add_scalar(self, dct: dict, flush=True):
        for tag, value in dct.items():
            self.logger.add_scalar(tag, value, global_step=self.epoch)
        if flush:
            self.logger.flush()

    def train(self, train_dataset, val_dataset=None):
        @partial(jax.pmap, axis_name='devices')
        def train_step(state, input_dict, key, step):
            return self.model.train_step(state, input_dict, key, step)
        @partial(jax.pmap, axis_name='devices')
        def val_step(state, input_dict, key, step):
            return self.model.val_step(state, input_dict, key, step)
        
        # Initialize model and optimizer
        self.setup_train()
        key = jax.random.PRNGKey(self.cfg.seed)
        
        # prepare batch generators
        train_datagen = create_iter_dataset(train_dataset, batch_size=self.cfg.batch_size)
        if val_dataset is not None:
            val_datagen = create_iter_dataset(val_dataset, batch_size=self.cfg.batch_size)

        # resume training if specified
        self.state = self.model.init_train_state(self.cfg)
        beg = 0
        if self.cfg.resume:
            beg = self.ckptr.latest_step()
            if beg > 0:
                logger.info(f'Resuming training from epoch {beg}')
                self.state = from_state_dict(
                    self.state, self.ckptr.restore(beg))
            else:
                logger.info('No checkpoint found, starting from scratch')
            

        # multi-device setup
        devices = jax.local_devices()
        num_devices = len(devices)
        self.state = jax.device_put_replicated(self.state, devices)

        # training loop
        pbar = tqdm(range(beg, self.cfg.num_epochs), desc="Epochs", position=0, leave=True)
        for self.epoch in pbar:
            # === Train epoch ===
            output_dicts = []
            for step, batch in zip(range(self.cfg.num_train_steps), train_datagen):
                if step % 100 == 0: 
                    pbar.set_postfix(
                        {"Train step": f"{step}/{self.cfg.num_train_steps}"})
                key, step_key = jax.random.split(key)

                # Reshape batch for multi-device training
                batch_sharded = jax.tree_util.tree_map(
                    lambda x: x.reshape(num_devices, x.shape[0]//num_devices, *x.shape[1:]), batch)
                step_key_sharded = jax.random.split(
                    step_key, num_devices).reshape((num_devices, -1))
                step_sharded = jnp.ones(num_devices) * step
                # Perform training step
                self.state, output_dict = train_step(
                    state=self.state, input_dict=batch_sharded, key=step_key_sharded, step=step_sharded)
                output_dicts.append(output_dict)
            output_dict = accumulate_metrics(output_dicts)
            
            # === Validation epoch ===
            if val_dataset is not None and self.cfg.num_val_steps > 0:
                val_output_dicts = []
                for step, val_batch in zip(range(self.cfg.num_val_steps), val_datagen):
                    if step % 100 == 0:
                        pbar.set_postfix(
                            {"Val step": f"{step}/{self.cfg.num_val_steps}"})
                    key, step_key = jax.random.split(key)
                    # Reshape batch for multi-device validation
                    val_batch_sharded = jax.tree_util.tree_map(
                        lambda x: x.reshape(num_devices, x.shape[0]//num_devices, *x.shape[1:]), val_batch)
                    step_key_sharded = jax.random.split(
                        step_key, num_devices).reshape((num_devices, -1))
                    step_sharded = jnp.ones(num_devices) * step
                    
                    # Perform validation step
                    val_output_dict = val_step(
                        state=self.state, input_dict=val_batch_sharded, key=step_key_sharded, step=step_sharded)
                    val_output_dicts.append(val_output_dict)
                # Accumulate validation metrics and log
                val_output_dict = accumulate_metrics(val_output_dicts)
                val_output_dict = {'val_'+k: v for k, v in val_output_dict.items()}

            # Accumulate metrics and log
            output_dict['epoch'] = self.epoch
            self.add_scalar(output_dict, flush=val_dataset is None)
            if val_dataset is not None:
                self.add_scalar(val_output_dict, flush=True)

            # === Save checkpoint ===
            if self.cfg.save_every_n_epochs > 0 and (self.epoch+1) % self.cfg.save_every_n_epochs == 0:
                self.save_checkpoint()
        
        # End of training loop
        if self.cfg.use_wandb:
            self.logger.close()
            wandb.finish()
                
        return self.state

    def setup_wandb(self):
        # Start wandb
        tags = None if self.cfg.wandb_tag == '' else [self.cfg.wandb_tag]
        wandb.init(project=self.cfg.wandb_project_name, sync_tensorboard=True, tags=tags)
        # log code
        fp = os.path.abspath(__file__)
        folder = os.path.dirname(fp)
        project_folder = os.path.dirname(folder)
        wandb.run.log_code(project_folder)
        # log hypm
        wandb.config.update(OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True))
        wandb.config.update({'log_dir': self.output_dir})
        
    def save_checkpoint(self):
        state = jax.tree.map(lambda x: x[0], self.state)
        self.ckptr.save(self.epoch+1, to_state_dict(state))

    @staticmethod
    # todo: too verbose
    def get_ckpt_manager(folder, verbose=0):
        from absl import logging
        class StandardLogger:
            def log_entry(self, msg, *args, **kwargs):
                step, dir = msg['step'], msg['directory']
                logger.info(f'Saved ckpt for epoch={step} to {dir}')
        #logging.set_verbosity(logging.DEBUG)
        folder = op.abspath(folder)
        return ocp.CheckpointManager(folder, ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                                     logger = StandardLogger(),
                                     options=ocp.CheckpointManagerOptions(step_prefix='epoch',
                                                                          step_format_fixed_length=4))
    
    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert op.isdir(target_base_dir)

        # write combined config yaml
        fpath = op.relpath(op.join(target_base_dir, 'combined.yaml'))
        with open(fpath, 'w') as f:
            f.write(yaml.dump(OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)))
        logger.info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = op.relpath(target_base_dir + '/src.zip')
        source_path = op.relpath(op.dirname(__file__) + '/../')
        def filter_(x): return x.endswith('.py') or x.endswith('.yaml')  # noqa
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = op.join(root, file_or_dir)
                    if op.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            op.join(root, file_or_dir),
                            op.relpath(op.join(root, file_or_dir),
                                       op.join(source_path, op.pardir)))
        logger.info('Written source folder to %s' % op.relpath(target_path))


    ### ---- PROJECT-SPECIFIC ---- ###
    def get_model_identifier(self):
        dset_name = self.cfg.dataset.dset_name.replace('/', '-')
        class_name = type(self.model).__name__.title()
        run_name = self.model.get_run_name()
        return os.path.join('outputs', dset_name, class_name, run_name)
