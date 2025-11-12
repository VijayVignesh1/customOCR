import Levenshtein as Lev

def compute_cer(gt_texts, pred_texts):
    """
    Compute Character Error Rate (CER) for a batch of strings.
    CER = (edit distance) / (total characters in GT)
    
    Args:
        gt_texts: list of ground truth strings
        pred_texts: list of predicted strings
    Returns:
        cer: float
    """
    total_edits = 0
    total_chars = 0
    
    for gt, pred in zip(gt_texts, pred_texts):
        total_edits += Lev.distance(gt, pred)
        total_chars += len(gt)
    
    return total_edits / total_chars if total_chars > 0 else 0.0


def compute_wer(gt_texts, pred_texts):
    """
    Compute Word Error Rate (WER) for a batch of strings.
    WER = (edit distance between word sequences) / (total words in GT)
    
    Args:
        gt_texts: list of ground truth strings
        pred_texts: list of predicted strings
    Returns:
        wer: float
    """
    total_edits = 0
    total_words = 0
    
    for gt, pred in zip(gt_texts, pred_texts):
        gt_words = gt.split()
        pred_words = pred.split()
        total_edits += Lev.distance(" ".join(gt_words), " ".join(pred_words))
        total_words += len(gt_words)
    
    return total_edits / total_words if total_words > 0 else 0.0