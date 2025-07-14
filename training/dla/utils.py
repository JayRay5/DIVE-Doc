import torch
import numpy as np
import torch.nn.functional as F


def collate_fn(batch):
    dict = {"pixel_values":[],"labels":[]}
    for example in batch:
        dict["pixel_values"].append(example[1])
        dict["labels"].append(example[0])

    dict["pixel_values"] = np.array(dict["pixel_values"])
    dict["labels"] = np.array(dict["labels"])

    dict["pixel_values"] = torch.tensor(dict["pixel_values"])
    dict["labels"] = torch.tensor(dict["labels"])
    return dict

def compute_metrics(eval_pred,metric,num_labels=11,ignore_index=255):
    with torch.no_grad():
        logits, labels = eval_pred

        if logits.shape[-2:] != labels.shape[-2:]:
            print(logits.shape)
            print(labels.shape)
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear"
            )
        logits = F.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)

        labels = labels.cpu().numpy().tolist()
        logits = logits.cpu().numpy().tolist()


        metrics = metric.compute(
            predictions=logits,
            references=labels,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False,
        )
        print(metrics)
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics
    
id2label = {
                "1": "Caption",
                "2": "Footnote",
                "3": "Formula",
                "4": "List-item",
                "5": "Page-footer",
                "6": "Page-header",
                "7": "Picture",
                "8": "Section-header",
                "9": "Table",
                "10": "Text",
                "11": "Title"
                }