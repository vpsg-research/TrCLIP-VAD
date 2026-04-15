import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):

    model.to(device)
    model.eval()    #   Set model to evaluate mode

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):   # item: (visual, visual_label, visual_length, text, text_label, text_length)
            visual = item[0].squeeze(0)
            visual_length = item[2]

            text = item[3].squeeze(0)
            text_length = item[5]
            visual_length = int(visual_length)
            text_length = int(text_length)
            len_cur = visual_length
            len_cur_text = int(text_length)
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            if len_cur_text < maxlen:
                text = text.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(visual_length / maxlen) + 1)
            for j in range(int(visual_length / maxlen) + 1):
                if j == 0 and visual_length < maxlen:
                    lengths[j] = visual_length
                elif j == 0 and visual_length > maxlen:
                    lengths[j] = maxlen
                    visual_length -= maxlen
                elif visual_length > maxlen:
                    lengths[j] = maxlen
                    visual_length -= maxlen
                else:
                    lengths[j] = visual_length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, text, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1 ,0 # averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.visual_test_list, args.text_test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, device)
    # model_param = torch.load(args.model_path)
    model_param = torch.load(args.model_path, weights_only=False)
    # model.load_state_dict(model_param)
    model.load_state_dict(model_param, strict=False)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)