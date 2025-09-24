import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import joblib
import time
import torch
import pandas as pd
from xfeat.modules.xfeat import XFeat  # import do XFeat

# ----------------------------
# Configurações
# ----------------------------
train_dir = Path('E:/MVP1_2024')              # imagens para treino (BoW)
query_dir = Path('E:/nadir_images')           # imagens para consulta
preprocessed_dir = Path('E:/MVP1_2024/preprocessed')
ground_truth_file = Path('E:/MVP1_2024/ground_truth_mvp1.csv')

n_clusters = 1000
max_keypoints = 900
n_matches = 5
force_preprocess = False

# ----------------------------
# XFeat
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
xfeat = XFeat().to(device)
xfeat.eval()
print("✅ XFeat instanciado. Device:", device)

# ----------------------------
# Gabarito (Ground Truth)
# ----------------------------
if ground_truth_file.exists():
    gt_df = pd.read_csv(ground_truth_file)
    gt_df = gt_df.apply(lambda x: x.str.replace('.png', '', regex=False))
    query_paths = sorted(
        query_dir.glob("*.png"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
    )
    query_names = [query_path.name.replace('.png', '') for query_path in query_paths]
    ground_truth_lists = gt_df[['Top1', 'Top2', 'Top3', 'Top4', 'Top5']].values.tolist()
    gt_dict = dict(zip(query_names, ground_truth_lists))
    print(f"✅ Gabarito carregado: {len(gt_dict)} queries")
else:
    gt_dict = {}
    print("⚠️ Nenhum arquivo de gabarito encontrado, acurácia não será calculada.")

# ----------------------------
# Funções auxiliares
# ----------------------------
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Falha ao carregar {path}")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, (640, 480))
    return img

def extract_xfeat_descriptors(img, top_k=max_keypoints):
    img_tensor = torch.from_numpy(img / 255.).float()[None, None].to(device)  # (1,1,H,W)
    with torch.no_grad():
        out = xfeat({'image': img_tensor})
    desc = out['descriptors'][0].cpu().numpy().T  # (N, 256)
    if desc.shape[0] == 0:
        return np.zeros((0,256), dtype=np.float32)
    if top_k < desc.shape[0]:
        desc = desc[:top_k]
    return desc.astype(np.float32)

# ----------------------------
# Pré-processamento
# ----------------------------
required_files = ['vocab.npy', 'histograms.npy', 'image_names.npy', 'kmeans_model.pkl']
preprocessed_exists = all((preprocessed_dir / f).exists() for f in required_files)

if not preprocessed_exists or force_preprocess:
    print("Pré-processamento iniciado...")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(
        train_dir.glob("*.png"),
        key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else int(x.stem)
    )
    print(f"Total de imagens de treino: {len(train_paths)}")
    if len(train_paths) == 0:
        raise RuntimeError("Nenhuma imagem .png encontrada no diretório de treino!")

    images = [(load_image(p), p.name) for p in train_paths]
    image_names = [Path(name).stem for _, name in images]

    all_descriptors = []
    descriptors_dict = {}
    for img, name in images:
        desc = extract_xfeat_descriptors(img)
        if desc.shape[0] > 0:
            all_descriptors.append(desc)
        descriptors_dict[Path(name).stem] = desc

    all_descriptors = np.vstack(all_descriptors)
    print(f"Descritores concatenados: {all_descriptors.shape}")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000).fit(all_descriptors)
    vocab = kmeans.cluster_centers_

    histograms = []
    for name in image_names:
        desc = descriptors_dict[name]
        if desc.shape[0] > 0:
            labels = kmeans.predict(desc)
            hist, _ = np.histogram(labels, bins=n_clusters, range=(0,n_clusters), density=True)
        else:
            hist = np.zeros(n_clusters)
        histograms.append(hist)
    histograms = np.array(histograms)

    np.save(preprocessed_dir / 'vocab.npy', vocab)
    np.save(preprocessed_dir / 'histograms.npy', histograms)
    np.save(preprocessed_dir / 'image_names.npy', image_names)
    joblib.dump(kmeans, preprocessed_dir / 'kmeans_model.pkl')
    print("Pré-processamento concluído.")
else:
    print("Carregando dados pré-processados...")
    vocab = np.load(preprocessed_dir / 'vocab.npy')
    histograms = np.load(preprocessed_dir / 'histograms.npy')
    image_names = np.load(preprocessed_dir / 'image_names.npy', allow_pickle=True)
    image_names = [Path(name).stem for name in image_names]
    kmeans = joblib.load(preprocessed_dir / 'kmeans_model.pkl')
    print(f"{len(image_names)} imagens de treino carregadas, {vocab.shape[0]} clusters.")

# ----------------------------
# BallTree para busca
# ----------------------------
tree = BallTree(histograms, metric='minkowski', p=2)

# ----------------------------
# Avaliação em imagens de consulta
# ----------------------------
print(f"Total de imagens de consulta: {len(query_paths)}")
total_queries = 0
total_acertos_top5 = 0

for query_path in query_paths:
    query_name = query_path.name.replace('.png', '')
    query_img = load_image(query_path)
    query_desc = extract_xfeat_descriptors(query_img)

    if query_desc.shape[0] == 0:
        print(f"Sem descritores: {query_name}")
        continue

    labels = kmeans.predict(query_desc)
    query_hist, _ = np.histogram(labels, bins=n_clusters, range=(0,n_clusters), density=True)

    start_time = time.time()
    distances, indices = tree.query([query_hist], k=n_matches)
    elapsed = time.time() - start_time

    top_matches = [(image_names[i], distances[0][j]) for j, i in enumerate(indices[0])]

    print(f"\nQuery: {query_name} | tempo: {elapsed:.4f}s")
    for rank, (name, dist) in enumerate(top_matches):
        print(f" {rank+1}. {name} (dist={dist:.4f})")

    # Checar gabarito
    if query_name in gt_dict:
        total_queries += 1
        expected_list = gt_dict[query_name]
        if any(gt in [m[0] for m in top_matches] for gt in expected_list):
            total_acertos_top5 += 1
        acc_top5 = total_acertos_top5 / total_queries * 100
        print(f"📊 Acurácia Top-5: {acc_top5:.2f}% ({total_acertos_top5}/{total_queries})")

cv2.destroyAllWindows()
