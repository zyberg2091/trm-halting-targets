"""Train with halting; evaluate with and without it.

Source: notebooks/8digit/epochs_100/seed_0/softmean.ipynb, cell 20.
SEED and EPOCHS come from config. See CODE_PROVENANCE.md for packaging changes.
"""
from torch import nn

from src.config import SEED, EPOCHS
from src.model import TinyRModel
from src.data import (PAD_TOKEN, device, hidden_size, vocab_size, inp_seq_len,
                      out_seq_len, n_chain, train_loader, val_loader)
from src.targets import Q_LOSS

import torch
import torch.nn.functional as F
from collections import Counter

def model_training_and_validation_with_mask(c, T, n_sup, HALT_THRESHOLD):

  torch.manual_seed(SEED)

  model = TinyRModel(hidden_size, vocab_size, inp_seq_len=inp_seq_len, out_seq_len=out_seq_len).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  print('PAD_TOKEN: ', PAD_TOKEN)
  criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_TOKEN)

  epochs = EPOCHS


  print('Training Logs ------ \n')
  for epoch in range(epochs):
        model.train()

        running_loss = 0
        running_q = 0
        running_n_taken = 0

        all_halt_steps = []
        all_halt_logits = []

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_size_curr = input_ids.size(0)
            seq_len = labels.size(1)

            active_mask = torch.ones(batch_size_curr, dtype=torch.bool, device=device)

            x = input_ids
            y = torch.zeros(batch_size_curr, hidden_size, device=device)
            z = torch.zeros(batch_size_curr, hidden_size, device=device)

            step_loss = step_q = 0
            overall_loss = torch.tensor(0.0, device=device)

            # n_sup includes both final-step halts and samples that never cross
            # the threshold. The logged %never cannot separate the two.
            halt_step = torch.full((batch_size_curr,), n_sup, dtype=torch.long, device=device)


            for i in range(n_sup):
                f_out, halt, y_new, z_new = model(x, y, z, T, n=6)

                # These summaries include inactive rows until the whole batch exits.
                all_halt_logits.append(halt.squeeze(-1).detach().cpu())

                # Freeze halted states. Detach the rest so gradients stop at
                # the boundary between supervision steps.
                y = torch.where(active_mask.unsqueeze(1), y_new.detach(), y)
                z = torch.where(active_mask.unsqueeze(1), z_new.detach(), z)

                seq_len = labels.size(1)  # Padded output length

                loss_per_token = criterion(f_out.view(-1, vocab_size), labels.view(-1))

                q_loss_per_sample = Q_LOSS(halt, f_out.view(-1, seq_len, vocab_size), labels)


                token_mask = active_mask.unsqueeze(1).expand(-1, seq_len).reshape(-1).float()
                sample_mask = active_mask.float()

                num_active_tokens = token_mask.sum() + 1e-8
                num_active_samples = sample_mask.sum() + 1e-8

                # Keep the saved runs' denominator: padding contributes zero CE
                # but still counts in num_active_tokens.
                current_step_loss = (loss_per_token * token_mask).sum() / num_active_tokens
                current_step_q = (q_loss_per_sample.squeeze(-1) * sample_mask).sum() / num_active_samples

                overall_loss += (current_step_loss + c * current_step_q)

                step_loss += current_step_loss.item()
                step_q += current_step_q.item()

                halt_decision = (halt.squeeze(-1) > HALT_THRESHOLD) & active_mask

                halt_step[halt_decision] = i + 1

                active_mask = active_mask & (~halt_decision)

                # No later-step loss once every sample has halted.
                if active_mask.sum() == 0:
                  break

            all_halt_steps.append(halt_step.cpu())

            n_taken = i + 1
            # Divide by batch depth, including the step that ended the loop.
            overall_loss /= n_taken

            # Avg Steps measures batch depth. Use halt_step for sample exits.
            running_n_taken += n_taken

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()

            running_loss += step_loss / n_taken
            running_q += step_q / n_taken



        avg_loss = running_loss / len(train_loader)
        avg_q_loss = running_q / len(train_loader)

        avg_steps = running_n_taken / len(train_loader)

        print(f'Training processed epoch {epoch}')
        if epoch % 10 == 0:
            print(f" Training epoch {epoch} | CE: {avg_loss:.4f} | Q-Loss: {avg_q_loss:.4f} | Avg Steps: {avg_steps}")

            all_halt_steps = torch.cat(all_halt_steps)
            all_halt_logits = torch.cat(all_halt_logits)

            mean_h = all_halt_steps.float().mean().item()
            median_h = all_halt_steps.float().median().item()
            frac_1 = (all_halt_steps == 1).float().mean().item()
            frac_max = (all_halt_steps == n_sup).float().mean().item()
            dist = dict(Counter(all_halt_steps.tolist()))

            print(f"halt logit: mean={all_halt_logits.mean().item():.3f} "
                  f"std={all_halt_logits.std().item():.3f} "
                  f"max={all_halt_logits.max().item():.3f} "
                  f"%>0={(all_halt_logits > 0).float().mean().item():.1%} "
                  f"%>1.5={(all_halt_logits > 1.5).float().mean().item():.1%}")

            print(f"halt steps: mean={mean_h:.2f} median={median_h:.0f} "
                  f"%step1={frac_1:.1%} %never={frac_max:.1%} dist={dist}")

        running_loss = running_q = 0


        if epoch % 10 == 0:
            print('\nValidation Logs ------ \n')
            model.eval()

            def validation(use_mask):
              correct = 0
              total = 0

              step_exact     = torch.zeros(n_sup, n_chain)
              step_count     = torch.zeros(n_sup, n_chain)

              step_tok_right = torch.zeros(n_sup)
              step_tok_total = torch.zeros(n_sup)

              all_halt_steps_validation = []
              all_exact_validation = []

              with torch.no_grad():
                  for val_step, batch in enumerate(val_loader):
                      input_ids = batch['input_ids'].to(device)
                      labels = batch['labels'].to(device)
                      batch_size_curr = input_ids.size(0)

                      batch_size_curr = input_ids.size(0)
                      seq_len = labels.size(1)

                      active_mask = torch.ones(batch_size_curr, dtype=torch.bool, device=device)

                      # As in training, n_sup also includes samples that never halt.
                      val_halt_step = torch.full((batch_size_curr,), n_sup, dtype=torch.long, device=device)

                      x = input_ids
                      y = torch.zeros(input_ids.size(0), hidden_size, device=device)
                      z = torch.zeros(input_ids.size(0), hidden_size, device=device)

                      final_logits = torch.zeros(batch_size_curr, out_seq_len * vocab_size, device=device)

                      for i in range(n_sup):
                          f_out_val, val_halt, y_new, z_new = model(x, y, z, T, n=6)
                          y = torch.where(active_mask.unsqueeze(1), y_new.detach(), y)
                          z = torch.where(active_mask.unsqueeze(1), z_new.detach(), z)

                          # Only the first batch contributes these printed logit statistics.
                          if use_mask and val_step == 0:
                              print(f"sup_step_{i+1} | mean={val_halt.mean().item():.3f} "
                                    f"std={val_halt.std().item():.3f} "
                                    f"median={val_halt.median().item():.3f} "
                                    f"%>0={(val_halt > 0).float().mean().item():.1%} "
                                    f"%>1.5={(val_halt > 1.5).float().mean().item():.1%} "
                                    f"first5={val_halt[:5].squeeze().tolist()}")


                          if use_mask:
                            halt_decision = (val_halt.squeeze(-1) > HALT_THRESHOLD) & active_mask
                            val_halt_step[halt_decision] = i + 1
                            active_mask = active_mask & (~halt_decision)
                            final_logits[halt_decision] = f_out_val[halt_decision]

                            if active_mask.sum() == 0:
                                break

                          else:
                            preds_i = f_out_val.view(-1, seq_len, vocab_size).argmax(dim=-1)
                            np_i = (labels != PAD_TOKEN)
                            ex = ((preds_i == labels) | ~np_i).all(dim=-1)         # Ignore padding in exact match
                            carry = batch['carry'].to(device)

                            # Keep separate depth comparisons for each carry-chain length.

                            for k in range(n_chain):
                                m = (carry == k)
                                step_exact[i, k] += (ex & m).sum().item()
                                step_count[i, k] += m.sum().item()

                            step_tok_right[i] += ((preds_i == labels) & np_i).sum().item()
                            step_tok_total[i] += np_i.sum().item()


                      if use_mask:
                        all_halt_steps_validation.append(val_halt_step.cpu())
                        final_logits[active_mask] = f_out_val[active_mask]

                        preds = final_logits.view(-1, seq_len, vocab_size).argmax(dim=-1)
                        non_pad = (labels != PAD_TOKEN)
                        exact = ((preds == labels) | ~non_pad).all(dim=-1)
                        correct += exact.sum().item()

                        all_exact_validation.append(exact.cpu())

                      total += labels.size(0)

                  return correct, total, all_halt_steps_validation, all_exact_validation, step_exact, step_count, step_tok_right, step_tok_total


            use_mask = True

            if use_mask:
              correct, total, all_halt_steps_validation, all_exact_validation, _, _, _, _ = validation(use_mask)
              print(f"\nValidation Accuracy with mask: {100 * correct / total:.2f}%")

              all_halt_steps_validation = torch.cat(all_halt_steps_validation)

              mean_h = all_halt_steps_validation.float().mean().item()
              median_h = all_halt_steps_validation.float().median().item()
              frac_1 = (all_halt_steps_validation == 1).float().mean().item()
              frac_max = (all_halt_steps_validation == n_sup).float().mean().item()

              dist = dict(Counter(all_halt_steps_validation.tolist()))

              print(f"validation halt steps: mean={mean_h:.2f} median={median_h:.0f} "
                    f"%step1={frac_1:.1%} %never={frac_max:.1%} dist={dist}")

              all_exact_validation = torch.cat(all_exact_validation)

              for s in range(1, n_sup + 1):
                  m = (all_halt_steps_validation == s)
                  if m.sum() > 0:
                      print(f"  halt_step_{s} | n={int(m.sum())} | EM: {100*all_exact_validation[m].float().mean().item():.2f}%")


            # Same validation set, this time with halting disabled.

            _, tot2, _, _, step_exact, step_count, step_tok_right, step_tok_total = validation(False)
            print("\nForced-depth (no halting):")
            print("        " + "".join(f"  chain={k}(n={int(step_count[0,k])})" for k in range(n_chain)))

            for i in range(n_sup):
                row = "".join(f"  {100*step_exact[i,k].item()/max(step_count[i,k].item(),1):6.2f}%" for k in range(n_chain))
                em  = 100 * step_exact[i].sum().item() / tot2
                tok = 100 * step_tok_right[i].item() / max(step_tok_total[i].item(), 1)
                print(f"  step_{i+1} |{row} | all EM: {em:.2f}% | TokAcc: {tok:.2f}%")
