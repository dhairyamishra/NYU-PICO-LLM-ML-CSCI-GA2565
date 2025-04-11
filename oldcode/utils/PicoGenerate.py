# starter code by matus & o1-pro
import torch
import torch.nn.functional as F


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cuda:0" if torch.cuda.is_available() else "cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

# def nucleus_sampling(logits, p=0.95):
#     return torch.argmax(logits).item()

def nucleus_sampling(logits, p=0.95):
    # Step 1: Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Step 2: Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Step 3: Compute cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Step 4: Identify the cutoff point where cumulative probability exceeds p
    cutoff = torch.searchsorted(cumulative_probs, p)

    # Step 5: Truncate to the top-k tokens
    truncated_probs = sorted_probs[:cutoff + 1]
    truncated_indices = sorted_indices[:cutoff + 1]

    # Step 6: Normalize the truncated probabilities
    truncated_probs = truncated_probs / truncated_probs.sum()

    # Step 7: Sample from the truncated distribution
    sampled_idx = torch.multinomial(truncated_probs, 1)
    return truncated_indices[sampled_idx].item()



def generate_text(model, enc, 
                  init_text, 
                  max_new_tokens, 
                  device,
                  top_p,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  conversation_history=None):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if conversation_history:
            history_text = ""
            for user_input, bot_reply in conversation_history[-5:]:
                history_text += f"User: {user_input}\nBot: {bot_reply}\n"
            init_text = history_text + f"User: {init_text}\nBot:"

        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text