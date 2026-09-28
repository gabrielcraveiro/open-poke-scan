# Servidor de reconhecimento

Serviço que o site chama para identificar a carta (`POST /recognize`). O site publicado usa o nosso servidor. Esta pasta mostra como montar o seu do zero: o modelo e o índice de cartas são gerados por você, a partir de fontes públicas.

## O que roda

1. **Embedding:** [DINOv2-small](https://huggingface.co/facebook/dinov2-small) (Meta, Apache-2.0), entrada 336×336, pesos fp16 (~44MB). Sem treino extra: é o modelo público.
2. **Busca:** similaridade de cosseno contra o índice de cartas (~22 mil, do [TCGdex](https://tcgdex.dev)).
3. **Desempate:** pHash da arte quando dois prints têm quase a mesma imagem.
4. **OCR do número** (Tesseract): só quando o match é ambíguo. Número + total impresso é quase uma chave única.

Roda em CPU. Um scan leva ~0,3–1s numa máquina de 1 vCPU.

## Passo a passo

Precisa de Python 3.12 e do Tesseract (`apt install tesseract-ocr` / `brew install tesseract`).

### 1. Exportar o modelo (uma vez)

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-build.txt      # inclui PyTorch CPU, ~1GB
python scripts/export_model.py --out models/model.onnx
```

O script baixa o DINOv2-small do Hugging Face e grava `models/model.onnx`. Verificamos que o resultado é idêntico ao modelo do servidor publicado (cosseno 1,0 entre os embeddings).

### 2. Gerar o índice de cartas

```bash
python scripts/build_index.py --sets sv07,sv08      # teste rápido: só alguns sets
python scripts/build_index.py                       # catálogo inteiro
```

O catálogo inteiro baixa ~600MB de imagens do TCGdex e leva ~1h a 1h30 numa CPU comum (~5 cartas/s). O script pode ser interrompido e retomado: imagens e vetores ficam em `.index-cache/`. Para adicionar um set novo depois, rode de novo: só as cartas novas são processadas.

Saída em `models/`: `emb.f16.bin` (vetores) e `meta.json` (dados das cartas, na mesma ordem).

### 3. Rodar

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
curl -F file=@minha-carta.jpg http://localhost:8000/recognize
```

Aponte o site para ele: `VITE_RECOGNIZE_URL=http://localhost:8000 npm run dev` na raiz do repo.

### 4. Publicar no Fly.io (opcional)

```bash
cp fly.toml.example fly.toml       # troque o nome do app
fly launch --copy-config --no-deploy
fly deploy
```

Com `auto_stop_machines = "suspend"`, a máquina dorme quando ninguém usa e acorda em 1–3s já com o modelo carregado. Numa máquina de 1 vCPU e 1GB, o custo fica em poucos dólares por mês. O Docker também roda em qualquer outro lugar: `docker build -t recognizer . && docker run -p 8000:8000 recognizer`.

## Configuração

| Variável | Padrão | Uso |
|---|---|---|
| `MODELS_DIR` | `./models` | Pasta com `model.onnx`, `emb.f16.bin` e `meta.json`. |
| `ALLOW_ORIGINS` | `*` | Origens liberadas no CORS, separadas por vírgula. Coloque a URL do seu site. |
| `MAX_CONCURRENCY` | `1` | Reconhecimentos em paralelo. Com 1 vCPU, deixe 1: dois em paralelo levam o dobro cada. |
| `DEBUG_DIR` | vazio | Se definido, guarda cada foto recebida por 6 horas (para depurar erros). |

## Diferenças para o servidor publicado

O nosso servidor tem, além disto, um detector de cartas próprio (YOLO) que acha a carta quando o site não consegue recortar sozinho, e alguns sets que não estão no TCGdex. Nenhum dos dois é necessário: o site já manda a carta recortada na grande maioria dos scans.

## Imagens e dados

As imagens das cartas pertencem a Nintendo / Creatures / GAME FREAK / The Pokémon Company. O índice é gerado na sua máquina a partir da API pública do TCGdex. Não redistribua as imagens nem o índice gerado.
