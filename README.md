# open-poke-scan

Scanner de cartas Pokémon TCG pela câmera do celular, direto no navegador. Aponte a câmera para a carta e o site diz qual é, com preço de referência e links de busca para a Liga Pokémon e a MYP Cards.

*Pokémon TCG card scanner for the phone browser. Point the camera at a card to identify it.*

## O que faz

- **Scan:** a retícula acompanha a carta e captura sozinha quando a imagem fica parada e nítida. O botão redondo força a captura.
- **Resultado:** imagem, nome, set e número da carta; preço de referência em real (mercado brasileiro); links de busca na Liga e na MYP.
- **Sem certeza:** quando o servidor não está confiante, a tela pergunta "Qual destas é a sua?" e mostra as candidatas grandes, com o set e o número em destaque. Um toque escolhe.
- **"Não é essa":** volta para a câmera. As outras candidatas aparecem no resultado para escolher com um toque.
- **Modo sling:** para lotes. Jogue as cartas uma a uma sob a câmera e cada carta reconhecida com confiança entra na lista sozinha. Na dúvida, a carta não entra: tire e ponha de novo.
- **Lista:** fica só no aparelho (localStorage). Dá para copiar como texto ou baixar em CSV.

Cartas em japonês ou coreano são identificadas pela arte e caem na versão em inglês equivalente. O servidor publicado também tem algumas coleções japonesas inteiras (ex.: Storm Emeralda), com a imagem da carta japonesa.

## Como funciona

```
câmera → retícula viva (scanic, segue os 4 cantos da carta)
       → foto nítida → YOLO acha os 4 cantos → corte com correção de perspectiva
       → POST /recognize (servidor)
           DINOv2-S @336 → similaridade de cosseno contra ~20 mil cartas
           → menos a penalidade das cartas "ímã" (as que parecem com qualquer foto)
           → desempate por pHash quando dois prints têm a mesma arte
           → OCR do número (Tesseract), lido no rodapé em alta resolução, só quando o match é ambíguo
       → resultado + preço de referência + links de busca
```

O reconhecimento roda num servidor, não no celular. Rodar o modelo no navegador travava a tela e levava segundos por carta. No servidor quente, um scan leva ~0,5s.

O site publicado usa o nosso servidor. Para usar o reconhecimento no seu projeto, veja [Usar o reconhecimento no seu projeto](#usar-o-reconhecimento-no-seu-projeto).

Decisões de arquitetura e o porquê de cada uma: [`docs/arquitetura.md`](docs/arquitetura.md).

## Rodar localmente

```bash
npm install
cp .env.example .env    # ajuste VITE_RECOGNIZE_URL se usar outro servidor
npm run dev             # http://localhost:5173
```

A câmera exige HTTPS, mas `localhost` é exceção. Para testar no celular, publique o build (veja abaixo) ou use um túnel HTTPS.

```bash
npm run typecheck
npm run build           # gera dist/, um site estático
```

## Publicar

O `dist/` é estático e roda em qualquer host (Vercel, Netlify, GitHub Pages, Cloudflare Pages). Configure as variáveis no build:

| Variável | Uso |
|---|---|
| `VITE_RECOGNIZE_URL` | URL do servidor de reconhecimento. Vazio = o servidor publicado (`https://cartinhas-recognize.fly.dev`). |
| `VITE_FEEDBACK_URL` | Opcional. Link do botão "Mandar feedback". |
| `VITE_TELEMETRY_URL` | Opcional. Endpoint que recebe a telemetria anônima (`POST`, corpo JSON em texto puro). Vazio = sem telemetria. |

## Usar o reconhecimento no seu projeto

Não existe um modelo treinado por nós. O modelo é o [DINOv2-small](https://huggingface.co/facebook/dinov2-small) público (Meta, Apache-2.0), sem ajuste. O que faz o reconhecimento funcionar é o que está em volta dele:

- **O índice de cartas:** um vetor por carta, a partir das imagens oficiais.
- **A penalidade das cartas "ímã":** medida com fotos reais de scans.
- **Os limites calibrados:** "não é carta", "confiante" e quando ler o número. Foram calibrados com 160 scans reais conferidos à mão.

Tudo isso está no código de [`server/`](server/), com os scripts para gerar o modelo e o índice. Há duas formas de usar:

### 1. Rodar o seu servidor (recomendado)

Siga o [`server/README.md`](server/README.md). O atalho é baixar da [última release](https://github.com/gabrielcraveiro/open-poke-scan/releases/latest) o modelo, o índice e a penalidade prontos, e pular direto para rodar. Do zero:

1. Exporte o modelo, que é idêntico ao do servidor publicado.
2. Gere o índice a partir do TCGdex.
3. Opcional: gere a penalidade com as suas fotos de scan.
4. Publique, por exemplo num Fly.io de 1 vCPU (poucos dólares por mês).
5. Aponte o site com `VITE_RECOGNIZE_URL`.

Diferenças para o servidor publicado:
- O seu índice tem só o que o TCGdex tem.
- O publicado tem, além disso, cartas que o TCGdex lista sem imagem (promos completadas com a imagem da Liga e do pokemontcg.io), algumas coleções japonesas e um detector YOLO para fotos sem recorte.

### 2. Usar o servidor publicado

Um build sem `VITE_RECOGNIZE_URL` já usa o nosso servidor (`https://cartinhas-recognize.fly.dev`), que aceita chamadas de qualquer site. Serve para testar e para uso pessoal, com estes limites:

- É **uma máquina de 1 vCPU**, compartilhada com o app que deu origem a este projeto. Os scans de todos entram na mesma fila, e um pico de uso deixa todo mundo lento.
- **Não há garantia** de disponibilidade nem de que o endereço ou o contrato da API continuem iguais.
- A máquina dorme quando ninguém usa. O primeiro scan depois disso leva até ~20s.
- As fotos recebidas ficam até ~6 horas em disco, para depurar erros (veja [Privacidade](#privacidade)).

Para um site com tráfego de verdade, rode o seu servidor (opção 1).

## Contrato do servidor

`POST {VITE_RECOGNIZE_URL}/recognize`, `multipart/form-data`:

- `file`: JPEG da carta (lado maior até 900px).
- `pre`: `1` quando a imagem já é só a carta recortada. Sem ele, o servidor detecta a carta na foto.
- `footer` (opcional, só com `pre=1`): JPEG dos 28% de baixo da mesma carta, em resolução cheia (até 1200px de largura). O servidor lê o número da carta nesta imagem. No `file` reduzido, o número tem ~12px e o OCR não lê.

Resposta:

```json
{
  "card": { "api_id": "sv07-158", "name": "Lapras ex", "number": "158", "set_id": "sv07",
            "set_name": "Stellar Crown", "printed_total": 142,
            "image_url": "https://assets.tcgdex.net/en/sv/sv07/158/low.webp", "cos": 0.91 },
  "confident": true,
  "ocr": { "number": "158", "total": "142", "match": true },
  "candidates": [ "… até 5 cartas no mesmo formato de card …" ]
}
```

`GET /health` acorda o servidor. O site chama essa rota ao abrir, porque a máquina suspende quando fica ociosa e o primeiro scan depois disso pode levar ~20s.

## Privacidade

A foto da carta vai para o servidor só para ser identificada. Ela fica no máximo ~6 horas em disco para depuração e depois é apagada. O site não tem conta nem cookie. A lista de cartas fica só no seu aparelho.

A versão publicada coleta telemetria anônima para medir o acerto do scanner: um ID aleatório por aparelho (gerado no localStorage), o país (vindo da CDN), a origem do acesso, os eventos de scan (cartas candidatas, confiança, tempo, "não é essa") e os dados técnicos da câmera que o navegador informa (nome da lente, modo e distância de foco, resolução). Não vai foto, IP nem nada que identifique a pessoa. O código está em `src/telemetry.ts`. Ele só envia quando o build define `VITE_TELEMETRY_URL`, então um build seu não manda nada.

## Créditos

- Detecção de cantos: [scanic](https://github.com/marquaye/scanic) (MIT).
- Cantos da carta na foto capturada: modelo [duclvQ/tcg-card-detector](https://huggingface.co/duclvQ/tcg-card-detector) (YOLOv8-pose, AGPL-3.0). O site baixa o modelo do Hugging Face ao abrir. O modelo não faz parte deste repositório.
- Dados, imagens e preços de referência: [TCGdex](https://tcgdex.dev).
- Pokémon e as imagens das cartas são marcas e propriedade de Nintendo, Creatures, GAME FREAK e The Pokémon Company. Este projeto não tem vínculo com elas.
- Nasceu do scanner do CartinhasDaJu, um app pessoal de coleção.

## Licença

[MIT](LICENSE)
