import os
import torch
import torch.nn as nn
from torch import optim
import lightning as L
from torcheval.metrics.functional import perplexity
#from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from models import Llama, LlamaHF
from utils import plot_metrics

class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module | dict,
        lr: float = 3e-3,
        weight_decay: float = 0.0,
        warmup_duration: int = 20
    ) -> None:
        '''
        args:
            model: if is dict, it shoule be in the form (see yaml file for more details)
                {
                    'name': 'Llama', # class name
                    'other param': value,
                    ...
                }
            warmup_duration: number epochs in warmup phase
        '''
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if isinstance(model, dict):
            name = model['name']
            model = {k: v for k, v in model.items() if k != 'name'}
            model = eval(name)(**model)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_duration = warmup_duration
    @classmethod
    def from_pretrained(cls, checkpoint: str):
        #wrapper_model = Transformer_Wrapper.load_from_checkpoint(checkpoint)
        wrapper_model = cls.load_from_checkpoint(checkpoint)
        return wrapper_model.model
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.lr,
            betas = (0.9, 0.95),
            eps = 1.0e-5,
            weight_decay = self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs
        cosine_lr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = max_epochs
        )
        linear_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.001,
            end_factor = 1.0,
            total_iters = self.warmup_duration
        )
        lr_sch_warmup = torch.optim.lr_scheduler.SequentialLR(
            optimizer = optimizer,
            schedulers = [linear_lr, cosine_lr],
            milestones = [self.warmup_duration]
        )
        return [optimizer], [lr_sch_warmup]
    def forward(
        self, 
        x: torch.Tensor, 
        #label: torch.Tensor,
        attn_mask: bool|None = None,
        is_causal: bool|None = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        '''
        args:
            sequence: (batch_size, seq_len)
        returns:
            logit (batch_size, seq_len, vocab_size)
        '''
        logit = self.model(x = x, attn_mask = attn_mask, is_causal = is_causal, **kwargs)
        return logit
    def training_step(self, batch):
        optimizer = self.optimizers()
        lr_sch = self.lr_schedulers()

        sequence, label = batch
        logit = self(sequence, is_causal = True)
        loss = self.model.loss_fn(logit, label, ignore_index = -1)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        acc = self.model.accuracy(label, logit)
        ppl = perplexity(logit, label, ignore_index = -1).item()

        if self.trainer.is_last_batch: #and (self.trainer.current_epoch+1)%10 == 0:
            lr_sch.step()
        log = {
            f'train_loss': loss.item(),
            'train_acc': acc,
            'train_ppl': ppl
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
        
    def validation_step(self, batch):
        sequence, label = batch
        logit = self(sequence, is_causal = True)
        loss = self.model.loss_fn(logit, label, ignore_index = -1)
        acc = self.model.accuracy(label, logit)
        ppl = perplexity(logit, label, ignore_index = -1).item()
        log = {
            f'val_loss': loss.item(),
            'val_acc': acc,
            'val_ppl': ppl
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
    def test_step(self, batch):
        sequence, label = batch
        logit = self(sequence, is_causal = True)
        loss = self.model.loss_fn(logit, label, ignore_index = -1)
        acc = self.model.accuracy(label, logit)
        ppl = perplexity(logit, label, ignore_index = -1).item()
        log = {
            f'test_loss': loss.item(),
            'test_acc': acc,
            'test_ppl': ppl
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
