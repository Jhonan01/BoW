import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import joblib
import time
import torch
import pandas as pd
from tqdm import tqdm
import sys

# XFeat import
sys.path.append(str(Path(__file__).parent / 'xfeat'))
from xfeat.modules.xfeat import XFeat

# Configurations
train_dir = Path('E:/MVP1_2024')        # Training images directory
query_dir = Path('E:/nadir_images')     # Query images directory
preprocessed_dir = Path('E:/MVP1_2024/preprocessed_xfeat')
ground_truth_file = Path('E:/MVP1_2024/ground_truth_mvp1.csv')

n_clusters = 1000
max_keypoints = 900
n_matches = 5
force_preprocess = False  # Set to False since preprocessing worked

# Initialize XFeat
device = 'cuda' if torch.cuda.is_available() else 'cpu'
xfeat = XFeat().to(device)
xfeat.eval()
print(f"✅ XFeat instanciado. Device: {device}")

# Global variable to store descriptor dimension
descriptor_dim = None

# Utility functions
def load_image_rgb(path, resize_to=(640, 480)):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Falha ao carregar {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    if resize_to is not None:
        img = cv2.resize(img, resize_to)
    return img

def extract_xfeat_descriptors(img, top_k=900):
    """
    Extract XFeat descriptors, return np.ndarray shape (N, D) where D is detected automatically
    """
    global descriptor_dim
    
    # Convert to tensor: shape (1, 3, H, W)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"Input tensor shape: {t.shape}")
    
    with torch.no_grad():
        desc = None
        try:
            # Try detectAndCompute
            if hasattr(xfeat, 'detectAndCompute'):
                result = xfeat.detectAndCompute(t)
                print(f"detectAndCompute output: {type(result)}, {len(result) if isinstance(result, (list, tuple)) else result}")
                
                # Handle list of dictionaries
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    result_dict = result[0]
                    kp = result_dict.get('keypoints', None)
                    desc = result_dict.get('descriptors', result_dict.get('descs', None))
                    print(f"Descriptors shape from detectAndCompute: {desc.shape if desc is not None else 'None'}")
                
                # Handle tuple (kp, desc)
                elif isinstance(result, (tuple, list)) and len(result) == 2:
                    kp, desc = result
                    print(f"Descriptors shape from detectAndCompute: {desc.shape if desc is not None else 'None'}")
                
                # Handle single tensor (descriptors)
                elif isinstance(result, torch.Tensor) and result.ndim == 2:
                    desc = result
                    print(f"Descriptors shape from detectAndCompute: {desc.shape}")
                
                if desc is not None and desc.shape[0] > 0:
                    desc = desc.cpu().numpy().astype(np.float32)
                    # Detect descriptor dimension on first call
                    if descriptor_dim is None:
                        descriptor_dim = desc.shape[1]
                        print(f"Descriptor dimension detected: {descriptor_dim}")
                    print(f"Descriptors extracted: {desc.shape}")
                    return desc[:top_k, :]
                
        except Exception as e:
            print(f"detectAndCompute error: {e}")
    
    # Fallback with correct dimension
    if descriptor_dim is None:
        descriptor_dim = 64  # Default to 64D if not detected
        print(f"Using default descriptor dimension: {descriptor_dim}")
    print("No descriptors found")
    return np.zeros((0, descriptor_dim), dtype=np.float32)

# Load ground truth
if ground_truth_file.exists():
    gt_df = pd.read_csv(ground_truth_file)
    gt_df = gt_df.apply(lambda x: x.str.replace('.png', '', regex=False))
    query_paths = sorted(query_dir.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    query_names = [p.stem for p in query_paths]
    ground_truth_lists = gt_df[['Top1', 'Top2', 'Top3', 'Top4', 'Top5']].values.tolist()
    gt_dict = dict(zip(query_names, ground_truth_lists))
    print(f"✅ Gabarito carregado: {len(gt_dict)} queries")
else:
    gt_dict = {}
    query_paths = sorted(query_dir.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    print("⚠️ Nenhum gabarito encontrado, acurácia não será calculada.")

# Preprocessing BoW
preprocessed_dir.mkdir(parents=True, exist_ok=True)
required_files = ['vocab.npy', 'histograms.npy', 'image_names.npy', 'kmeans_model.pkl']
preprocessed_exists = all((preprocessed_dir / f).exists() for f in required_files)

if not preprocessed_exists or force_preprocess:
    print("Pré-processamento iniciado...")
    train_paths = sorted(train_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else int(x.stem) if x.stem.isdigit() else x.stem)
    print(f"Total de imagens de treino: {len(train_paths)}")
    if len(train_paths) == 0:
        raise RuntimeError("Nenhuma imagem de treino encontrada!")

    images = [(load_image_rgb(p), p.name) for p in train_paths]
    image_names = [Path(name).stem for _, name in images]

    all_descriptors = []
    descriptors_dict = {}
    total_desc_count = 0

    for img, name in tqdm(images, desc="Extraindo descritores"):
        desc = extract_xfeat_descriptors(img, top_k=max_keypoints)
        descriptors_dict[Path(name).stem] = desc
        total_desc_count += desc.shape[0]
        if desc.shape[0] > 0:
            all_descriptors.append(desc)

    print(f"Total descritores encontrados (soma por imagem): {total_desc_count}")
    if len(all_descriptors) == 0:
        raise RuntimeError("Nenhum descritor encontrado!")

    all_descriptors = np.vstack(all_descriptors)
    print(f"Descritores concatenados: {all_descriptors.shape}")
    print(f"Dimensão dos descritores: {all_descriptors.shape[1]}D")

    # KMeans
    print("Treinando MiniBatchKMeans ...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    kmeans.fit(all_descriptors)
    vocab = kmeans.cluster_centers_
    print("KMeans treinado.")

    # Histograms
    histograms = []
    for name in image_names:
        desc = descriptors_dict[name]
        if desc.shape[0] > 0:
            labels = kmeans.predict(desc)
            hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters), density=True)
        else:
            hist = np.zeros(n_clusters, dtype=np.float32)
        histograms.append(hist)
    histograms = np.array(histograms, dtype=np.float32)

    # Save preprocessed data
    np.save(preprocessed_dir / 'vocab.npy', vocab)
    np.save(preprocessed_dir / 'histograms.npy', histograms)
    np.save(preprocessed_dir / 'image_names.npy', image_names)
    np.save(preprocessed_dir / 'descriptor_dim.npy', np.array([descriptor_dim]))
    joblib.dump(kmeans, preprocessed_dir / 'kmeans_model.pkl')
    print("Pré-processamento concluído e salvo.")
else:
    print("Carregando dados pré-processados...")
    vocab = np.load(preprocessed_dir / 'vocab.npy')
    histograms = np.load(preprocessed_dir / 'histograms.npy')
    image_names = np.load(preprocessed_dir / 'image_names.npy', allow_pickle=True)
    image_names = [Path(name).stem for name in image_names]
    descriptor_dim = int(np.load(preprocessed_dir / 'descriptor_dim.npy')[0])
    kmeans = joblib.load(preprocessed_dir / 'kmeans_model.pkl')
    print(f"{len(image_names)} imagens carregadas, {vocab.shape[0]} clusters, {descriptor_dim}D descriptors.")

# BallTree
tree = BallTree(histograms, metric='minkowski', p=2)

# Evaluation
print(f"Total de imagens de consulta: {len(query_paths)}")
total_queries = 0
total_acertos_top5 = 0

for query_path in tqdm(query_paths, desc="Consultas"):
    query_name = query_path.stem
    query_img = load_image_rgb(query_path)
    query_desc = extract_xfeat_descriptors(query_img, top_k=max_keypoints)
    if query_desc.shape[0] == 0:
        print(f"⚠️ Sem descritores: {query_name}")
        continue

    labels = kmeans.predict(query_desc)
    query_hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters), density=True)

    start_time = time.time()
    distances, indices = tree.query([query_hist], k=n_matches)
    elapsed = time.time() - start_time

    top_matches = [(image_names[i], distances[0][j]) for j, i in enumerate(indices[0])]
    print(f"\nQuery: {query_name} | tempo: {elapsed:.4f}s")
    for rank, (name, dist) in enumerate(top_matches):
        print(f" {rank+1}. {name} (dist={dist:.4f})")

    if query_name in gt_dict:
        total_queries += 1
        expected_list = gt_dict[query_name]
        if any(gt in [m[0] for m in top_matches] for gt in expected_list):
            total_acertos_top5 += 1
            print("✅ ACERTOU Top-5")
        else:
            print("❌ ERROU Top-5")
        acc_top5 = total_acertos_top5 / total_queries * 100
        print(f"📊 Acurácia Top-5: {acc_top5:.2f}% ({total_acertos_top5}/{total_queries})")

cv2.destroyAllWindows()
print("Fim.")