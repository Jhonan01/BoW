# BoW_Xfeat_full.py
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
from modules.xfeat import XFeat  # seu módulo local XFeat

# ----------------------------
# Configurações (edite conforme necessário)
# ----------------------------
train_dir = Path('E:/MVP1_2024')              # imagens para treino (BoW)
query_dir = Path('E:/nadir_images')           # imagens para consulta
preprocessed_dir = Path('E:/MVP1_2024/preprocessed_xfeat')
ground_truth_file = Path('E:/MVP1_2024/ground_truth_mvp1.csv')

n_clusters = 1000
n_matches = 5
force_preprocess = False
debug = False  # True para prints extras

# ----------------------------
# Inicializa XFeat
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
xfeat = XFeat()   # sua versão local
# não fazemos xfeat.to(device) porque XFeat encapsula seu backend
print(f"✅ XFeat instancia criada. Device detectado (torch): {device}")

# opcional: mostrar métodos disponíveis (útil para debug)
if debug:
    print("Métodos/atributos de xfeat:", [m for m in dir(xfeat) if not m.startswith('_')])

# ----------------------------
# Carrega gabarito (ground truth) se existir
# ----------------------------
if ground_truth_file.exists():
    gt_df = pd.read_csv(ground_truth_file)
    # remove .png se houver
    gt_df = gt_df.apply(lambda x: x.str.replace('.png', '', regex=False))
    query_paths = sorted(query_dir.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    query_names = [p.stem for p in query_paths]
    ground_truth_lists = gt_df[['Top1','Top2','Top3','Top4','Top5']].values.tolist()
    gt_dict = dict(zip(query_names, ground_truth_lists))
    print(f"✅ Gabarito carregado: {len(gt_dict)} queries")
else:
    gt_dict = {}
    query_paths = sorted(query_dir.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    print("⚠️ Nenhum arquivo de gabarito encontrado, acurácia não será calculada.")

# ----------------------------
# Funções utilitárias
# ----------------------------
def load_image_gray(path, resize_to=(640,480)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Falha ao carregar {path}")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    if resize_to is not None:
        img = cv2.resize(img, resize_to)
    return img

def to_torch_tensor_gray(img_gray):
    """Retorna tensor [1,1,H,W] float32 no CPU/GPU conforme availability"""
    t = torch.from_numpy(img_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t.to(torch.float32)

def to_torch_tensor_rgb(img_bgr):
    """Retorna tensor [1,3,H,W] float32"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return t.to(torch.float32)

def ensure_numpy_desc(desc):
    """
    Normaliza a saída de descritores para np.ndarray shape (N,D), dtype float32.
    Aceita: torch.Tensor, np.ndarray, list, etc.
    Retorna None se não conseguir.
    """
    if desc is None:
        return None
    if isinstance(desc, torch.Tensor):
        return desc.cpu().numpy().astype(np.float32)
    if isinstance(desc, np.ndarray):
        return desc.astype(np.float32)
    # se for lista de listas
    try:
        arr = np.asarray(desc)
        if arr.size == 0:
            return None
        return arr.astype(np.float32)
    except Exception:
        return None

# ----------------------------
# Wrapper robusto para extrair descritores com XFeat
# tenta várias APIs (detectAndCompute, extract_features, run, callable, match_xfeat_star)
# ----------------------------
def extract_xfeat_descriptors(img_gray, img_bgr=None, top_k=5000):
    """
    img_gray: imagem em grayscale (H,W) uint8 ou float
    img_bgr: imagem em BGR colorida (opcional) — necessária para alguns métodos
    Retorna: np.ndarray (N, D) dtype float32, ou array vazio (0, D) se nada encontrado.
    """
    # prepare tensors/inputs
    # várias APIs aceitam formatos diferentes — vamos montar alguns candidatos
    guesses = []

    # 1) [1,1,H,W] tensor (muito comum)
    try:
        t_gray = to_torch_tensor_gray(img_gray)
        if torch.cuda.is_available():
            t_gray = t_gray.cuda()
        guesses.append(('tensor_gray', t_gray))
    except Exception:
        pass

    # 2) [1,3,H,W] tensor RGB (ou BGR convertido)
    if img_bgr is None and img_gray is not None:
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if img_bgr is not None:
        try:
            t_rgb = to_torch_tensor_rgb(img_bgr)
            if torch.cuda.is_available():
                t_rgb = t_rgb.cuda()
            guesses.append(('tensor_rgb', t_rgb))
        except Exception:
            pass

    # 3) numpy HxW float [0,1] (algumas impls)
    try:
        arr_gray = img_gray.astype(np.float32) / 255.0
        guesses.append(('np_gray_norm', arr_gray))
    except Exception:
        pass

    # 4) numpy HxWx3 RGB uint8/float for match_xfeat_star
    if img_bgr is not None:
        try:
            arr_bgr = img_bgr  # keep BGR as cv2 read
            guesses.append(('np_bgr', arr_bgr))
        except Exception:
            pass

    # Instances / methods to try (in order)
    # We'll try multiple calling styles and inspect the return to extract descriptors.
    tries = [
        ('detectAndCompute', lambda inp: getattr(xfeat, 'detectAndCompute')(inp) if hasattr(xfeat, 'detectAndCompute') else None),
        ('extract_features', lambda inp: getattr(xfeat, 'extract_features')(inp) if hasattr(xfeat, 'extract_features') else None),
        ('extract', lambda inp: getattr(xfeat, 'extract')(inp) if hasattr(xfeat, 'extract') else None),
        ('run', lambda inp: getattr(xfeat, 'run')(inp) if hasattr(xfeat, 'run') else None),
        ('callable', lambda inp: xfeat(inp) if callable(xfeat) else None),
        ('match_xfeat_star', lambda inp: getattr(xfeat, 'match_xfeat_star')(inp, inp, top_k=top_k) if hasattr(xfeat, 'match_xfeat_star') else None),
        ('match_xfeat_star_bgr', lambda inp: getattr(xfeat, 'match_xfeat_star')(img_bgr, img_bgr, top_k=top_k) if (hasattr(xfeat, 'match_xfeat_star') and img_bgr is not None) else None)
    ]

    last_err = None

    for guess_name, inp in guesses:
        for method_name, method_call in tries:
            # skip incompatible combos early
            if method_name == 'match_xfeat_star' and guess_name.startswith('tensor'):
                # match_xfeat_star expects images (np arrays), not tensors in many impls
                continue

            try:
                # pick input: for detectAndCompute/extract_features we prefer tensor inputs;
                # for match_xfeat_star use numpy BGR
                input_for_call = inp
                res = method_call(input_for_call)
            except Exception as e:
                last_err = (guess_name, method_name, e)
                if debug:
                    print(f"tentativa falhou: guess={guess_name}, method={method_name}, erro={e}")
                res = None

            if res is None:
                continue

            # Agora interpretar retorno (pode ser dict, list, tuple, tensor, np.ndarray)
            # Casos:
            # - dict com chave 'descriptors' (tensor/ndarray/list)
            # - list/tuple onde um elemento é descriptors
            # - diretamente tensor/ndarray
            desc = None

            # 1) dict
            if isinstance(res, dict):
                # várias formas: res['descriptors'], res.get('descs'), etc.
                for key in ('descriptors', 'descs', 'desc', 'descriptor'):
                    if key in res:
                        desc = res[key]
                        break
                # às vezes o dict é {'keypoints':..., 'descriptors': tensor}
                # se não encontrou, tentamos adivinhar por tipo
                if desc is None:
                    # tentar localizar primeiro tensor/ndarray dentro do dict
                    for k, v in res.items():
                        if isinstance(v, (torch.Tensor, np.ndarray, list)):
                            # heurística: desc tem shape (N,D) ou list com len>0
                            # preferir chaves cujo valor tenha 2D shape
                            desc = v
                            break

            # 2) list/tuple
            elif isinstance(res, (list, tuple)):
                # procurar o primeiro elemento que pareça ser descritor (tensor/ndarray/list)
                for v in res:
                    if isinstance(v, (torch.Tensor, np.ndarray, list)):
                        # heurística: shape dims >= 2
                        desc = v
                        break

            # 3) tensor/ndarray directly
            elif isinstance(res, (torch.Tensor, np.ndarray)):
                desc = res

            # 4) else: ignore

            # Normalize desc to numpy array if possible
            desc_np = ensure_numpy_desc(desc)
            if desc_np is not None and desc_np.size != 0:
                # sanity check: ensure second dimension exists
                if desc_np.ndim == 1:
                    # single descriptor -> make (1, D)
                    desc_np = desc_np.reshape(1, -1)
                if desc_np.ndim == 2:
                    if debug:
                        print(f"✅ Extraído com sucesso usando guess={guess_name}, method={method_name}; shape={desc_np.shape}")
                    return desc_np.astype(np.float32)
                else:
                    # Unexpected shape; skip this result
                    if debug:
                        print(f"⚠️ Descriptor com formato inesperado (ndim={desc_np.ndim}), tentando próximo. guess={guess_name}, method={method_name}")
                    continue

            # se chegamos aqui, esta tentativa não produziu descritores válidos
            # continuar tentando
    # fim for tries

    # se nenhum método funcionou, opcionalmente podemos tentar a "gambiarra" de match_xfeat_star com imagens
    if hasattr(xfeat, 'match_xfeat_star') and img_bgr is not None:
        try:
            res = xfeat.match_xfeat_star(img_bgr, img_bgr, top_k=top_k)
            # retorno típico: (mkpts0, mkpts1, desc0, desc1) ou dict
            desc_candidate = None
            if isinstance(res, tuple) and len(res) >= 3:
                desc_candidate = res[2]
            elif isinstance(res, dict) and 'desc0' in res:
                desc_candidate = res['desc0']
            elif isinstance(res, dict) and 'descriptors' in res:
                desc_candidate = res['descriptors']
            dnp = ensure_numpy_desc(desc_candidate)
            if dnp is not None and dnp.ndim == 2:
                return dnp.astype(np.float32)
        except Exception as e:
            last_err = ('fallback_match_xfeat_star', e)
            if debug:
                print("fallback match_xfeat_star falhou:", e)

    # nada encontrado: retorna array vazio (0,256)
    if debug and last_err is not None:
        print("Último erro observado durante extração:", last_err)
    return np.zeros((0, 256), dtype=np.float32)

# ----------------------------
# Pré-processamento BoW
# ----------------------------
preprocessed_dir.mkdir(parents=True, exist_ok=True)
required_files = ['vocab.npy', 'histograms.npy', 'image_names.npy', 'kmeans_model.pkl']
preprocessed_exists = all((preprocessed_dir / f).exists() for f in required_files)

if not preprocessed_exists or force_preprocess:
    print("Pré-processamento iniciado...")
    # encontra imagens de treino e ordena numericamente caso necessário
    train_paths = sorted(train_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else int(x.stem) if x.stem.isdigit() else x.stem)
    print(f"Total de imagens de treino: {len(train_paths)}")
    if len(train_paths) == 0:
        raise RuntimeError("Nenhuma imagem .png encontrada no diretório de treino!")

    # carrega imagens e extrai descritores
    images = [(load_image_gray(p), cv2.imread(str(p), cv2.IMREAD_COLOR), p.name) for p in train_paths]
    image_names = [Path(name).stem for _, _, name in images]

    all_descriptors = []
    descriptors_dict = {}
    total_desc_count = 0

    for img_gray, img_bgr, name in tqdm(images, desc="Extraindo descritores"):
        desc = extract_xfeat_descriptors(img_gray, img_bgr, top_k=max(5000, 2000))
        descriptors_dict[Path(name).stem] = desc
        if desc.shape[0] > 0:
            all_descriptors.append(desc)
            total_desc_count += desc.shape[0]

    print(f"Total descritores encontrados (soma por imagem): {total_desc_count}")
    if len(all_descriptors) == 0:
        raise RuntimeError("Nenhum descritor encontrado em nenhuma imagem! Verifique a API do XFeat / formato das entradas.")

    all_descriptors = np.vstack(all_descriptors)
    print(f"Descritores concatenados: {all_descriptors.shape}")

    # Treinar kmeans (MiniBatch)
    print("Treinando MiniBatchKMeans ...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    kmeans.fit(all_descriptors)
    vocab = kmeans.cluster_centers_
    print("KMeans treinado.")

    # gerar histogramas por imagem
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

    # salvar
    np.save(preprocessed_dir / 'vocab.npy', vocab)
    np.save(preprocessed_dir / 'histograms.npy', histograms)
    np.save(preprocessed_dir / 'image_names.npy', image_names)
    joblib.dump(kmeans, preprocessed_dir / 'kmeans_model.pkl')
    print("Pré-processamento concluído e salvo.")
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

for query_path in tqdm(query_paths, desc="Consultas"):
    query_name = query_path.stem
    query_img_gray = load_image_gray(query_path)
    query_img_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)

    query_desc = extract_xfeat_descriptors(query_img_gray, query_img_bgr)

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

    # visualização (combinada)
    best_match_name = top_matches[0][0]
    best_match_img = load_image_gray(train_dir / (best_match_name + '.png'))
    combined = np.hstack((query_img_gray, best_match_img))
    cv2.imshow("Query | Melhor correspondência", combined)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Fim.")
