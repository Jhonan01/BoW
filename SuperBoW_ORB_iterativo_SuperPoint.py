import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import joblib
import time

# Configurações
image_dir = Path('E:/nadir_images')
preprocessed_dir = Path('E:/nadir_images/preprocessed')
n_clusters = 1000
max_keypoints = 1000
n_matches = 5
force_preprocess = False

# ORB
orb = cv2.ORB_create(nfeatures=max_keypoints, scaleFactor=1.5, nlevels=12)

# Funções
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Falha ao carregar {path}")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, (640, 480))
    return img

def extract_orb_descriptors(img):
    _, desc = orb.detectAndCompute(img, None)
    return desc.astype(np.float32) if desc is not None else np.zeros((0,32), dtype=np.float32)

# Verifica se já existe pré-processamento
required_files = ['vocab.npy', 'histograms.npy', 'image_names.npy', 'kmeans_model.pkl']
preprocessed_exists = all((preprocessed_dir / f).exists() for f in required_files)

if not preprocessed_exists or force_preprocess:
    print("Pré-processamento iniciado...")
    preprocessed_dir.mkdir(exist_ok=True)

    image_paths = sorted(image_dir.glob('*.png'), key=lambda x: int(x.stem))
    images = [(load_image(p), p.name) for p in image_paths]
    image_names = [name for _, name in images]

    all_descriptors = []
    descriptors_dict = {}
    for img, name in images:
        desc = extract_orb_descriptors(img)
        if desc.shape[0] > 0:
            all_descriptors.append(desc)
        descriptors_dict[name] = desc

    all_descriptors = np.vstack(all_descriptors)
    print(f"Descritores concatenados: {all_descriptors.shape}")

    # KMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000).fit(all_descriptors)
    vocab = kmeans.cluster_centers_

    # Histogramas
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

    # Salvar
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
    kmeans = joblib.load(preprocessed_dir / 'kmeans_model.pkl')
    print(f"{len(image_names)} imagens carregadas, {vocab.shape[0]} clusters.")

# BallTree para busca
tree = BallTree(histograms, metric='minkowski', p=2)

# Avaliação de acurácia
total_queries = 0
correct_matches = 0

# Iterar todas as imagens como query
image_paths = sorted(image_dir.glob('*.png'), key=lambda x: int(x.stem))

for query_path in image_paths:
    query_name = query_path.name
    query_img = load_image(query_path)
    query_desc = extract_orb_descriptors(query_img)

    if query_desc.shape[0] == 0:
        print(f"Sem descritores: {query_name}")
        continue

    labels = kmeans.predict(query_desc)
    query_hist, _ = np.histogram(labels, bins=n_clusters, range=(0,n_clusters), density=True)

    start_time = time.time()
    distances, indices = tree.query([query_hist], k=n_matches)
    elapsed = time.time() - start_time

    # Mantém o self-match
    top_matches = [(image_names[i], distances[0][j]) for j, i in enumerate(indices[0])]

    print(f"\nQuery: {query_name} | tempo: {elapsed:.4f}s")
    for rank, (name, dist) in enumerate(top_matches):
        print(f" {rank+1}. {name} (dist={dist:.4f})")

    # Verificação de acerto (melhor correspondência == query)
    best_match_name = top_matches[0][0]
    if best_match_name == query_name:
        correct_matches += 1
    total_queries += 1

    # Mostrar query e melhor correspondência (rank 1)
    best_match_img = load_image(image_dir / best_match_name)
    combined = np.hstack((query_img, best_match_img))
    cv2.imshow("Query | Melhor correspondência", combined)

    key = cv2.waitKey(300)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Resultado final de acurácia
accuracy = (correct_matches / total_queries) * 100 if total_queries > 0 else 0
print(f"\nAcurácia final: {accuracy:.2f}% ({correct_matches}/{total_queries})")
