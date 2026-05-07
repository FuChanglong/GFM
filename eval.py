import heapq
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from data.load import load_data
from models import TIGFM
from args import Arguments
from utils.process import split_dataloader, load_parsed_target_data, parse_target_data
from utils.augmentation import graph_aug


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

eval_template={
    'cora': "this paper has a topic on {c}",
    'citeseer': "good paper of {c} ",
    'pubmed': "this paper has a topic on {c}",
    'arxiv_2023': "it belongs to {c} research area",
    'wikics': "it belongs to {c} research area",
    'photo':  "this product belongs to {c}",
    'computer':  "is {c} category",
    'history': "this book belongs to {c}",
    'instagram': "{c}",
    'reddit': "{c}"
}


@torch.no_grad()
def encode_class_text_embs(model, tokenizer, classes, c_descs, dataset_name, device):
    model.eval()
    text_inputs = [
        eval_template[dataset_name].format(c=c) + desc
        for c, desc in zip(classes, c_descs)
    ]
    batch_t = tokenizer(
        text_inputs,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512
    ).to(device)
    text_embs = model.encode_text(
        batch_t["input_ids"],
        batch_t["token_type_ids"],
        batch_t["attention_mask"]
    )
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    return text_embs


def _push_topk(heap, score, emb, k):
    if torch.is_tensor(score):
        score = score.item()
    item = (score, emb.detach())
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if score > heap[0][0]:
            heapq.heapreplace(heap, item)

@torch.no_grad()
def build_extended_prototypes_from_views(
        model,
        loader,
        text_embs,
        classes,
        view_params,
        device,
        topk_per_class=10,
        score_reduce="mean"  # "mean" or "min"
):
    stable_class_counter = defaultdict(int)
    model.eval()
    num_classes = len(classes)
    heaps = {c: [] for c in range(num_classes)}
    all_stable_margins = []
    for batch in loader:
        data_list = batch.to_data_list()
        preds_list = []
        scores_list = []
        embs_list = []
        for (pn, pe) in view_params:
            aug_list = [graph_aug(g, pn, pe) for g in data_list]
            aug_batch = type(batch).from_data_list(aug_list).to(device)
            g_emb, _ = model.encode_graph(aug_batch)
            g_emb = g_emb / g_emb.norm(dim=-1, keepdim=True)
            sim = g_emb @ text_embs.T
            pred = sim.argmax(dim=1)
            score = sim.max(dim=1).values
            preds_list.append(pred)
            scores_list.append(score)
            embs_list.append(g_emb)
        preds_stack = torch.stack(preds_list, dim=0)
        scores_stack = torch.stack(scores_list, dim=0)
        embs_stack = torch.stack(embs_list, dim=0)
        stable_mask = (preds_stack == preds_stack[0:1]).all(dim=0)
        if stable_mask.sum() == 0:
            continue
        stable_class = preds_stack[0][stable_mask]
        for c in stable_class.tolist():
            stable_class_counter[int(c)] += 1
        stable_emb = embs_stack[:, stable_mask, :].mean(dim=0)
        stable_emb = stable_emb / stable_emb.norm(dim=-1, keepdim=True)
        if score_reduce == "min":
            stable_score = scores_stack[:, stable_mask].min(dim=0).values
        else:
            stable_score = scores_stack[:, stable_mask].mean(dim=0)
        for i in range(stable_emb.size(0)):
            node_emb = stable_emb[i]
            pred_c = int(stable_class[i].item())
            pred_sim = torch.dot(node_emb, text_embs[pred_c]).item()
            sim_all = torch.matmul(
                node_emb.unsqueeze(0), text_embs.T
            ).squeeze(0)
            other_mask = torch.ones_like(sim_all, dtype=torch.bool)
            other_mask[pred_c] = False
            other_sim_mean = sim_all[other_mask].mean().item()
            margin = pred_sim - other_sim_mean
            all_stable_margins.append(margin)
            s = stable_score[i].item()
            _push_topk(heaps[pred_c], s, node_emb, topk_per_class)

    class_protos = {}
    for c in range(num_classes):
        if len(heaps[c]) == 0:
            continue
        items = sorted(heaps[c], key=lambda x: x[0], reverse=True)
        class_protos[c] = torch.stack(
            [it[1] for it in items], dim=0
        ).to(device)

    return class_protos, all_stable_margins, stable_class_counter


@torch.no_grad()
def evaluate_fusion_desc_and_protos(
        model,
        loader,
        text_embs,
        class_protos,
        device,
        alpha=0.2,
        name='none'
):
    model.eval()
    correct = 0
    total = 0

    fused_class_embs = text_embs.clone()
    for c in range(text_embs.size(0)):
        protos = class_protos[c]
        protos = protos / protos.norm(dim=-1, keepdim=True)
        proto_emb = protos.mean(dim=0, keepdim=True)  # [1, D]
        proto_emb = proto_emb / proto_emb.norm(dim=-1, keepdim=True)
        fused_emb = (1 - alpha) * text_embs[c: c + 1] + alpha * proto_emb
        fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)  # 归一化
        fused_class_embs[c: c + 1] = fused_emb
    all_graph_embs = []
    all_true_labels = []
    for batch in loader:
        batch = batch.to(device)
        graph_embs, _ = model.encode_graph(batch)
        graph_embs = graph_embs / graph_embs.norm(dim=-1, keepdim=True)
        final_sim = graph_embs @ fused_class_embs.T
        preds = final_sim.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += preds.size(0)
        all_graph_embs.append(graph_embs.cpu())
        all_true_labels.append(batch.y.cpu())
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    config = Arguments().parse_args()
    set_seed(666)

    device = torch.device("cuda:1")
    attn_kwargs = {"dropout": 0.0}
    config.batch_size = 64
    model = TIGFM(
        384,
        1024,
        12,
        attn_kwargs,
        text_id=config.text_ids
    )
    ckpt = torch.load(f"./checkpoints/{config.ckpt}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.text_ids)
    print("Model loaded.")

    view_params = [
        (0.2, 0.2),
        (0.2, 0.3),
        (0.4, 0.3),
    ]

    topk_per_class = 20
    alpha = 0.3
    target_datasets = config.target_data.split("+")
    res_str = ""
    total_time=0
    for dataset_name in target_datasets:
        print(f"\n===== {dataset_name} =====")

        data, text, classes, c_descs = load_data(dataset_name, seed=0)
        print(text[:10])
        graph = load_parsed_target_data(dataset_name)
        _, _, test_loader = split_dataloader(
            data,
            graph,
            config.batch_size,
            seed=0,
            name=dataset_name
        )

        # 1) encode class descriptions
        text_embs = encode_class_text_embs(
            model,
            tokenizer,
            classes,
            c_descs,
            dataset_name,
            device
        )

        # 2) build extended prototypes from stable nodes
        class_protos, stable_margins, stable_class_counter = build_extended_prototypes_from_views(
            model=model,
            loader=test_loader,
            text_embs=text_embs,
            classes=classes,
            view_params=view_params,
            device=device,
            topk_per_class=topk_per_class)

        # 3) evaluate fusion
        st=time.time()
        acc = evaluate_fusion_desc_and_protos(
            model,
            test_loader,
            text_embs,
            class_protos,
            device,
            alpha=alpha,
            name=dataset_name
        )
        total_time += time.time() - st
        res_str += f" {dataset_name} acc: {acc:.4f}"
        print(f" {dataset_name} acc: {acc:.4f}")
    print(total_time)
    print("\n[FINAL]", res_str)
