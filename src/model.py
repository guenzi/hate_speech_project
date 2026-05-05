import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTweetMultiTask(nn.Module):
    def __init__(self, model_name="vinai/bertweet-base", dropout_rate=0.1):
        super(BERTweetMultiTask, self).__init__()
        
        # 1. Bertweet backbone
        self.bertweet = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # 2. racism detection, output: 1 logit (racism vs non-racism)
        self.racism_classifier = nn.Linear(hidden_size, 1)

        # 3. irony detection, output: 1 logit (irony vs non-irony)
        self.irony_classifier = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use of the [CLS] token representation for both tasks
        cls_representation = outputs.last_hidden_state[:, 0, :]
        cls_representation = self.dropout(cls_representation)

        # Logits computing for both tasks
        racism_logits = self.racism_classifier(cls_representation)
        irony_logits = self.irony_classifier(cls_representation)

        return {
            "racism_logits": racism_logits,
            "irony_logits": irony_logits,
            "racism_probs": self.sigmoid(racism_logits),
            "irony_probs": self.sigmoid(irony_logits)
        }

def compute_multitask_loss(outputs, labels_racism, labels_irony, lambda_irony=0.5):
    """
    Compute the loss for the multi-task model, combining the losses of both tasks with a weighting factor lambda_irony to balance their importance.
    """
    criterion = nn.BCEWithLogitsLoss()
    
    loss_racism = criterion(outputs["racism_logits"].squeeze(), labels_racism.float())
    loss_irony = criterion(outputs["irony_logits"].squeeze(), labels_irony.float())
    
    # lambda equal to 0.5 means we give equal importance to both tasks, but this can be tuned based on validation performance
    total_loss = loss_racism + (lambda_irony * loss_irony)
    
    return total_loss, loss_racism, loss_irony
