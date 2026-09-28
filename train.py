"""
Дообучение bi энкодера
то же что в ноутбуке 05_finetune но скриптом и с продолжением.
"""
import sys, json, random, argparse, time
sys.path.insert(0, ".")
from pathlib import Path
import polars as pl
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers

from src.text import item_text, query_text

BASE_MODEL = "deepvk/USER-bge-m3"
OUT = Path("data/models/user-bge-m3-ft1")
BATCH = 256 # столько негативов будет видеть каждый запрос
MINI_BATCH = 16 # столько реально влезает в память, остальное делает gradcache
MAX_LEN = 160

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=int, default=127488)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

random.seed(0)
torch.manual_seed(0)


checkpoints = sorted(OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))# если есть чекпойнт - продолжаем с него. число пар берем из него же иначе батчи не сойдутся
resume = None
n_pairs = args.pairs
if checkpoints:
    resume = str(checkpoints[-1])
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
    n_pairs = state["max_steps"] * BATCH
    print("продолжаю с", resume, "шаг", state["global_step"], "из", state["max_steps"])

# пары запрос -выбранное объявление, без валидационных запросов. тексты те же, что при поиске
train = pl.read_parquet("data/processed/train_rest.parquet",
                        columns=["search_query", "search_infm_params_text", "item_id",
                                 "item_title_raw", "item_infm_params_text", "item_description_raw"])
train = train.unique(subset=["search_query", "search_infm_params_text", "item_id"], maintain_order=True)

anchors = [query_text(q) for q in train["search_query"]]
positives = [item_text(t, p, d) for t, p, d in zip(train["item_title_raw"], train["item_infm_params_text"], train["item_description_raw"])]
order = list(range(len(anchors)))
random.shuffle(order)
order = order[:n_pairs - n_pairs % BATCH]
ds = Dataset.from_dict({"anchor": [anchors[i] for i in order], "positive": [positives[i] for i in order]})
print("пар в эпохе", len(ds))


model = SentenceTransformer(BASE_MODEL, device="mps", config_kwargs={"attention_probs_dropout_prob": 0.0})
model.max_seq_length = MAX_LEN

model[0].auto_model.embeddings.word_embeddings.weight.requires_grad_(False) # матрица токенов это половина параметров ее не учим

loss = losses.CachedMultipleNegativesRankingLoss(model, scale=20.0, mini_batch_size=MINI_BATCH)
targs = SentenceTransformerTrainingArguments(
    output_dir=str(OUT),
    num_train_epochs=1,
    per_device_train_batch_size=BATCH,
    learning_rate=args.lr,
    warmup_ratio=0.05,
    lr_scheduler_type="linear",
    batch_sampler=BatchSamplers.NO_DUPLICATES,# один позитив дважды в батче = ложный негатив
    dataloader_drop_last=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=2,
    report_to="none",
    seed=0,
)

t = time.time()
trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
trainer.train(resume_from_checkpoint=resume)
model.save(str(OUT))
print("готово за", round((time.time() - t) / 3600, 1), "ч")
