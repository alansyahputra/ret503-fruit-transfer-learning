import argparse, copy, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

ROOT, RES = Path("dataset_raw"), Path("results")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ALL_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"]

def build_model(name, mode, nc):
    m = getattr(models, name)(weights=None if mode == "scratch" else "DEFAULT")
    if name == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc); head, last = m.classifier[1], m.features[7:]
    elif name.startswith("mobilenet_v3"):
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
        head = m.classifier[3]; last = m.features[9:] if name.endswith("small") else m.features[13:]
    else:
        m.fc = nn.Linear(m.fc.in_features, nc); head, last = m.fc, m.layer4
    for p in m.parameters(): p.requires_grad = (mode == "scratch")
    for p in head.parameters(): p.requires_grad = True
    if mode == "partial":
        for p in last.parameters(): p.requires_grad = True
        groups = [{"params": list(last.parameters()), "lr": 1e-4}, {"params": list(head.parameters()), "lr": 1e-3}]
    else:
        groups = [{"params": [p for p in m.parameters() if p.requires_grad], "lr": 1e-3}]
    return m, groups

class DS(Dataset):
    def __init__(self, df, cls2idx, tf):
        self.items = [(ROOT/r.kelas/r.nama_file, cls2idx[r.kelas]) for r in df.itertuples()]; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]; return self.tf(Image.open(p).convert("RGB")), y

def set_train(m):
    m.train()
    for mod in m.modules():  # BatchNorm pada layer beku tetap eval()
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()

@torch.no_grad()
def evaluate(m, dl, dev, crit, nc):
    m.eval(); loss = n = 0; cm = np.zeros((nc, nc), int)
    for x, y in dl:
        x, y = x.to(dev), y.to(dev); o = m(x)
        loss += crit(o, y).item()*len(y); n += len(y)
        for t, p in zip(y.cpu().numpy(), o.argmax(1).cpu().numpy()): cm[t, p] += 1
    return loss/n, float(np.trace(cm)/n), cm

def run(model_name, mode, seed, epochs, bs, df, classes):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c2i = {c: i for i, c in enumerate(classes)}
    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    ev_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    mk = lambda s, tf, sh: DataLoader(DS(df[df["split"] == s], c2i, tf), batch_size=bs, shuffle=sh, drop_last=sh, num_workers=2)
    dl_tr, dl_va, dl_te = mk("train", tr_tf, True), mk("valid", ev_tf, False), mk("test", ev_tf, False)
    m, groups = build_model(model_name, mode, len(classes)); m.to(dev)
    opt = torch.optim.Adam(groups); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    n_total = sum(p.numel() for p in m.parameters()); n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    hist, best, best_state, t0 = [], (-1.0, 9e9), None, time.time()
    for ep in range(1, epochs + 1):
        set_train(m); tl = tc = n = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); o = m(x); loss = crit(o, y); loss.backward(); opt.step()
            tl += loss.item()*len(y); tc += (o.argmax(1) == y).sum().item(); n += len(y)
        sch.step()
        vl, va, _ = evaluate(m, dl_va, dev, crit, len(classes))
        hist.append(dict(epoch=ep, train_loss=tl/n, train_acc=tc/n, val_loss=vl, val_acc=va))
        if (va, -vl) > (best[0], -best[1]): best, best_state = (va, vl), copy.deepcopy(m.state_dict())
    train_time = time.time() - t0
    m.load_state_dict(best_state)
    _, ta, cm = evaluate(m, dl_te, dev, crit, len(classes))
    accs = [h["val_acc"] for h in hist]
    first = next((h["epoch"] for h in hist if h["val_acc"] >= 0.9), None)
    stable = next((i + 1 for i in range(len(accs)) if all(a >= 0.9 for a in accs[i:])), None)
    out = RES/f"{model_name}_{mode}_s{seed}"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out/"history.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out/"confusion_test.csv")
    json.dump(dict(model=model_name, mode=mode, seed=seed, params_total=n_total, params_trained=n_train,
                   best_val_acc=float(best[0]), test_acc=ta, epoch_first_90=first, epoch_stable_90=stable,
                   train_time_s=round(train_time, 1), n_val=len(dl_va.dataset), n_test=len(dl_te.dataset),
                   classes=classes), open(out/"summary.json", "w"), indent=1)
    print(f"{model_name:18s} {mode:8s} seed={seed} val={best[0]:.3f} test={ta:.3f} waktu={train_time:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--modes", nargs="+", default=["feature", "partial", "scratch"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    df = pd.read_csv(ROOT/"metadata.csv"); classes = sorted(df.kelas.unique())
    for mn in (ALL_MODELS if a.model == "all" else [a.model]):
        for mode in a.modes:
            for s in a.seeds: run(mn, mode, s, a.epochs, a.bs, df, classes)
