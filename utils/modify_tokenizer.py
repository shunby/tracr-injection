import os
import requests
from transformers import GPT2TokenizerFast

# ------------------------------------------------------------------------
# Main entry point: returns a tokenizer according to the requested task
# ------------------------------------------------------------------------
def get_tokenizer(task):
    if task == 'int-sum':
        # Pass the "skip multi-digit" filter
        return create_filtered_tokenizer(
            filter_fn=skip_merge_if_multi_digit,
            merges_outfile="modified_merges_int_sum.txt"
        )
    elif task == 'dyck':
        # Pass the "skip bracket merges" filter
        return create_filtered_tokenizer(
            filter_fn=skip_merge_if_bracket,
            merges_outfile="modified_merges_dyck.txt"
        )
    else:
        # Default GPT2 tokenizer
        return GPT2TokenizerFast.from_pretrained("gpt2")


# ------------------------------------------------------------------------
# 1. The base function that does the main work
# ------------------------------------------------------------------------
def create_filtered_tokenizer(filter_fn, merges_outfile, 
                              base_model="gpt2"):
    """
    1. Downloads GPT-2 vocab/merges if needed
    2. Applies `filter_fn` to skip merges
    3. Writes final merges to `merges_outfile`
    4. Creates and returns a GPT2TokenizerFast
    """
    base_url = f"https://huggingface.co/{base_model}/resolve/main/"
    vocab_file = "vocab.json"
    merges_file = "merges.txt"

    # 1. Download vocab/merges if needed
    download_if_not_exists(base_url, vocab_file)
    download_if_not_exists(base_url, merges_file)

    # 2. Read original merges
    with open(merges_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    header, merge_rules = split_header_and_rules(lines)

    # 3. Apply the filter function
    filtered_merge_rules = []
    for rule in merge_rules:
        pair = rule.split()
        if len(pair) == 2:
            t1, t2 = pair
            # If filter_fn(t1, t2) is True, skip the merge
            if filter_fn(t1, t2):
                continue
        filtered_merge_rules.append(rule)

    # 4. Write the new merges file
    with open(merges_outfile, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for line in filtered_merge_rules:
            f.write(line + "\n")

    # 5. Create and return the tokenizer
    return GPT2TokenizerFast(
        vocab_file=vocab_file, 
        merges_file=merges_outfile
    )


# ------------------------------------------------------------------------
# 2. Filter functions (one for int-sum, one for dyck)
# ------------------------------------------------------------------------

def skip_merge_if_multi_digit(t1, t2):
    """
    For 'int-sum' task: skip merges that form multi-digit numbers.
    """
    combined = (t1 + t2).replace("Ġ", " ")
    combined_stripped = combined.strip()
    # Return True if we *should skip* this merge
    return (combined_stripped.isdigit() and len(combined_stripped) > 1)

def skip_merge_if_bracket(t1, t2):
    """
    For 'dyck' task: skip merges that include '(' or ')' or '[' or ']' or '{' or '}'.
    """
    bracket_set = {"(", ")", "[", "]", "{", "}"}
    s1 = t1.replace("Ġ", "").strip()
    s2 = t2.replace("Ġ", "").strip()
    # If either token is a bracket, skip the merge
    return (s1 in bracket_set or s2 in bracket_set)


# ------------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------------

def download_if_not_exists(base_url, filename):
    """
    Download `filename` from `base_url` if it does not exist.
    """
    if not os.path.exists(filename):
        url = base_url + filename
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

def split_header_and_rules(lines):
    """
    Split merges file lines into comment header and actual merge rules.
    """
    header = []
    merge_rules = []
    for line in lines:
        if line.startswith('#'):
            header.append(line)
        else:
            merge_rules.append(line)
    return header, merge_rules


# ------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------
if __name__ == '__main__':
    # int-sum example
    sum_tokenizer = get_tokenizer('int-sum')
    print(sum_tokenizer("1+1=2"))
    print(sum_tokenizer("1 + 1 = 2"))

    # dyck example
    dyck_tokenizer = get_tokenizer('dyck')
    print(dyck_tokenizer("( [ { ) ] }"))
    print(dyck_tokenizer("test ( [ ) ] words"))

