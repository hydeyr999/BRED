import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from torch.utils.data import DataLoader
from .model import DiscrepancyEstimator
from .dataset import CustomDataset
from .loss import calculate_DPO_loss, calculate_DDL_loss
from .metrics import AUROC, AUPR
from accelerate import Accelerator
from tqdm import tqdm

class Trainer():
    def train(self,
              accelerator: Accelerator,
              model: DiscrepancyEstimator,
              train_dataset: CustomDataset,
              eval_dataset: CustomDataset,
              loss_fn = calculate_DDL_loss,
              learning_rate: float = 1e-4,
              num_epochs: int = 5,
              eval_freq: int = 1,
              save_freq: int = 5,
              save_directory: str='./ckpt/',
              save_name: str = None,
              DDL_target_original_crit: float = 0.,
              DDL_target_rewritten_crit: float = 100.,
              DPO_beta: float = 0.05,
              train_batch_size: int = 1,
              eval_batch_size: int = 1,
              track_with_wandb: bool = True,
              eval: bool = False,):
        assert loss_fn in [calculate_DDL_loss, calculate_DPO_loss], "Invalid loss function"
        start_time = time.time()
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                 collate_fn=eval_dataset.collate_fn)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=num_epochs * len(train_loader),
                                                            eta_min=0,
                                                            last_epoch=-1)
        model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, lr_scheduler)
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_losses = []
            epoch_original_discrepancy_train = []
            epoch_rewritten_discrepancy_train = []
            model.train()
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fine-tuning: {epoch+1} epoch", disable=not accelerator.is_main_process):
                outputs = model(batch['scoring']['original_input_ids'],
                                batch['scoring']['original_attention_mask'],
                                batch['scoring']['rewritten_input_ids'],
                                batch['scoring']['rewritten_attention_mask'],
                                batch['reference']['original_input_ids'],
                                batch['reference']['original_attention_mask'],
                                batch['reference']['rewritten_input_ids'],
                                batch['reference']['rewritten_attention_mask'])
                if loss_fn == calculate_DPO_loss:
                    loss, _, _, _, _ = loss_fn(
                        outputs['scoring_rewritten_logprob'],
                        outputs['scoring_original_logprob'],
                        outputs['reference_rewritten_logprob'],
                        outputs['reference_original_logprob'],
                        beta=DPO_beta)
                else:
                    loss = loss_fn(outputs['scoring_original_discrepancy'],
                                   outputs['scoring_rewritten_discrepancy'],
                                   target_original_crit=DDL_target_original_crit,
                                   target_rewritten_crit=DDL_target_rewritten_crit)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                train_losses.append(loss.detach().cpu().item())
                epoch_original_discrepancy_train.extend(outputs['scoring_original_discrepancy'].detach().cpu().tolist())
                epoch_rewritten_discrepancy_train.extend(outputs['scoring_rewritten_discrepancy'].detach().cpu().tolist())
                if track_with_wandb and accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": loss.detach().cpu().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=step + epoch * len(train_loader))
            accelerator.wait_for_everyone()
            all_train_losses = accelerator.gather_for_metrics(torch.tensor(train_losses, device=accelerator.device))
            all_original_train = accelerator.gather_for_metrics(torch.tensor(epoch_original_discrepancy_train, device=accelerator.device))
            all_rewritten_train = accelerator.gather_for_metrics(torch.tensor(epoch_rewritten_discrepancy_train, device=accelerator.device))
            if accelerator.is_main_process:
                fpr, tpr, train_epoch_auroc = AUROC(all_original_train.cpu().tolist(), all_rewritten_train.cpu().tolist())
                prec, recall, train_epoch_aupr = AUPR(all_original_train.cpu().tolist(), all_rewritten_train.cpu().tolist())
                epoch_loss = torch.mean(all_train_losses).item()
                original_discrepancy_mean = torch.mean(all_original_train).item()
                original_discrepancy_std = torch.std(all_original_train).item()
                rewritten_discrepancy_mean = torch.mean(all_rewritten_train).item()
                rewritten_discrepancy_std = torch.std(all_rewritten_train).item()

                # Prepare log dictionary
                log_dict = {
                    "train/auroc": train_epoch_auroc,
                    "train/aupr": train_epoch_aupr,
                    "train/original_discrepancy_mean": original_discrepancy_mean,
                    "train/original_discrepancy_std": original_discrepancy_std,
                    "train/rewritten_discrepancy_mean": rewritten_discrepancy_mean,
                    "train/rewritten_discrepancy_std": rewritten_discrepancy_std,
                }

                accelerator.print(f'Epoch: {epoch + 1} | Time: {time.time() - epoch_start_time:.3f} sec')
                accelerator.print(f'Train Loss: {epoch_loss:.8f}')
                accelerator.print(f'Original Discrepancy: {original_discrepancy_mean:.2f} ± {original_discrepancy_std:.2f} | Rewritten Discrepancy: {rewritten_discrepancy_mean:.2f} ± {rewritten_discrepancy_std:.2f}')
                accelerator.print(f'Train AUROC: {train_epoch_auroc:.4f} | Train AUPR: {train_epoch_aupr:.4f}')
            
            if eval:
                if (epoch + 1) % eval_freq == 0 or (epoch + 1) == num_epochs:
                    epoch_original_discrepancy_eval, epoch_rewritten_discrepancy_eval = self.eval(model, eval_loader, show_progress_bar=accelerator.is_main_process)
                    accelerator.wait_for_everyone()
                    all_original_eval = accelerator.gather_for_metrics(torch.tensor(epoch_original_discrepancy_eval, device=accelerator.device))
                    all_rewritten_eval = accelerator.gather_for_metrics(torch.tensor(epoch_rewritten_discrepancy_eval, device=accelerator.device))
                    if accelerator.is_main_process:
                        fpr, tpr, eval_epoch_auroc = AUROC(all_original_eval.cpu().tolist(), all_rewritten_eval.cpu().tolist())
                        prec, recall, eval_epoch_aupr = AUPR(all_original_eval.cpu().tolist(), all_rewritten_eval.cpu().tolist())
                        original_discrepancy_mean = torch.mean(all_original_eval).item()
                        original_discrepancy_std = torch.std(all_original_eval).item()
                        rewritten_discrepancy_mean = torch.mean(all_rewritten_eval).item()
                        rewritten_discrepancy_std = torch.std(all_rewritten_eval).item()
                        log_dict.update({
                            "eval/auroc": eval_epoch_auroc,
                            "eval/aupr": eval_epoch_aupr,
                            "eval/original_discrepancy_mean": original_discrepancy_mean,
                            "eval/original_discrepancy_std": original_discrepancy_std,
                            "eval/rewritten_discrepancy_mean": rewritten_discrepancy_mean,
                            "eval/rewritten_discrepancy_std": rewritten_discrepancy_std,
                        })
                        accelerator.print(f'Original Discrepancy: {original_discrepancy_mean:.2f} ± {original_discrepancy_std:.2f} | Rewritten Discrepancy: {rewritten_discrepancy_mean:.2f} ± {rewritten_discrepancy_std:.2f}')
                        accelerator.print(f'Eval AUROC: {eval_epoch_auroc:.4f} | Eval AUPR: {eval_epoch_aupr:.4f}')
                accelerator.wait_for_everyone()
                if track_with_wandb and accelerator.is_main_process:
                    accelerator.log(log_dict, step=(epoch + 1) * len(train_loader))

            if (epoch + 1) == num_epochs or (epoch + 1) % save_freq == 0:
                accelerator.wait_for_everyone()
                accelerator.print('saving model ...')
                if save_name is None:
                    raise ValueError('save_name should not be None')
                this_epoch_save_name = f'{save_name}_e{epoch+1}'
                if not os.path.exists(os.path.join(save_directory, this_epoch_save_name)):
                    os.makedirs(os.path.join(save_directory, this_epoch_save_name), exist_ok=True)
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(save_directory, this_epoch_save_name))

            if (epoch + 1) == num_epochs:
                if track_with_wandb:
                    accelerator.end_training()
                if accelerator.is_main_process:
                    accelerator.print(f'Finished Training!')
                    accelerator.print(f'Total Time: {time.time() - start_time:.3f} sec')

    def eval(self,
             model,
             eval_loader,
             show_progress_bar: bool = True,):
        epoch_original_discrepancy_eval = []
        epoch_rewritten_discrepancy_eval = []
        model.eval()
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating", disable=not show_progress_bar):
                outputs = model(
                    batch['scoring']['original_input_ids'],
                    batch['scoring']['original_attention_mask'],
                    batch['scoring']['rewritten_input_ids'],
                    batch['scoring']['rewritten_attention_mask'],
                    batch['reference']['original_input_ids'],
                    batch['reference']['original_attention_mask'],
                    batch['reference']['rewritten_input_ids'],
                    batch['reference']['rewritten_attention_mask']
                )
                epoch_original_discrepancy_eval.extend(outputs['scoring_original_discrepancy'].cpu().tolist())
                epoch_rewritten_discrepancy_eval.extend(outputs['scoring_rewritten_discrepancy'].cpu().tolist())

        return epoch_original_discrepancy_eval, epoch_rewritten_discrepancy_eval