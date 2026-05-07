import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel

from models import GraphCLIP2
from models.dp import TextCLIP, GCLIP
from args import Arguments
from utils.augmentation import graph_aug
from utils.process import load_parsed_source_data
from tools import *


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(data_loader):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        model.graph_model.redraw_projection.redraw_projections()

        batch = [g.to(device) for g in batch]

        summary_embs = torch.stack([g.summary_emb for g in batch], dim=0).to(device)

        texts = [g.text for g in batch]
        batch_t = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with torch.no_grad():
            input_ids_1 = span_mask_80_10_10(
                batch_t["input_ids"],
                batch_t["attention_mask"],
                tokenizer,
                mask_ratio=0.15,
                span_len=(3, 8)
            )
            input_ids_2 = span_mask_80_10_10(
                batch_t["input_ids"],
                batch_t["attention_mask"],
                tokenizer,
                mask_ratio=0.20,
                span_len=(3, 5)
            )

            text_embs_1 = model_text(
                input_ids=input_ids_1,
                token_type_ids=None,
                attention_mask=batch_t["attention_mask"]
            )
            text_embs_2 = model_text(
                input_ids=input_ids_2,
                token_type_ids=None,
                attention_mask=batch_t["attention_mask"]
            )

        batch_text_1 = replace_graph_x_with_text_embs(batch, text_embs_1)
        batch_text_2 = replace_graph_x_with_text_embs(batch, text_embs_2)

        batch_1 = [graph_aug(g, 0.1, 0.2) for g in batch_text_1]
        batch_2 = [graph_aug(g, 0.2, 0.3) for g in batch_text_2]

        graph_embs_1, _ = model_graph(batch_1)
        graph_embs_2, _ = model_graph(batch_2)

        info_nce = infonce_loss(graph_embs_1, graph_embs_2, temperature=0.07)

        loss_sen_1 = text_alignment_loss(graph_embs_1, summary_embs, temperature=0.07)
        loss_sen_2 = text_alignment_loss(graph_embs_2, summary_embs, temperature=0.07)

        loss_dir_1 = directional_alignment_loss(graph_embs_1, summary_embs, tau=0.07, hard_topk=3)
        loss_dir_2 = directional_alignment_loss(graph_embs_2, summary_embs, tau=0.07, hard_topk=3)
        loss = (
                0.2 * (loss_sen_1 + loss_sen_2)
                + 2.0 * info_nce
                + loss_dir_1
                + loss_dir_2
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)

    return total_loss / len(data_loader.dataset)


if __name__ == "__main__":
    config = Arguments().parse_args()
    set_seed(88)

    device = torch.device("cuda")
    attn_kwargs = {"dropout": 0.0}

    model = GraphCLIP2(
        384,
        1024,
        12,
        attn_kwargs,
        text_id=config.text_ids
    )
    model.freeze_text()
    model.to(device)

    model_text = TextCLIP(model).to(device)
    model_graph = GCLIP(model)
    model_graph = DataParallel(model_graph)

    tokenizer = AutoTokenizer.from_pretrained(config.text_ids)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    start_epoch = 1
    if config.is_resume:
        print(f"[Resume] loading checkpoint: {config.resume}")
        ckpt = torch.load(config.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] start from epoch {start_epoch}")

    all_source_graph = []
    for source_name in config.source_data.split("+"):
        source_graph = load_parsed_source_data(source_name)
        all_source_graph.extend(source_graph)

    print(f"We have {len(all_source_graph)} pretraining graphs")

    model_text.eval()
    print("Precomputing summary embeddings...")
    all_source_graph = precompute_summary_embeddings(
        all_source_graph,
        model_text,
        tokenizer,
        device,
        batch_size=64
    )

    train_loader = DataListLoader(
        all_source_graph,
        batch_size=config.batch_size,
        shuffle=True
    )

    for epoch in range(start_epoch, config.epochs + 1):
        loss = train(train_loader)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(ckpt, f"./checkpoints/our.pt")
