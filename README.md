# open-poke-scan

Scanner de cartas Pokémon TCG pela câmera do celular, direto no navegador. Aponte a câmera para a carta e o site diz qual é, com preço de referência e links de busca para a Liga Pokémon e a MYP Cards.

*Pokémon TCG card scanner for the phone browser. Point the camera at a card to identify it.*

## O que faz

- **Scan:** a retícula acompanha a carta e captura sozinha quando a imagem fica parada e nítida. O botão redondo força a captura.
- **Resultado:** imagem, nome, set e número da carta; preço de referência em real (mercado brasileiro) e lá fora (Cardmarket em EUR e TCGplayer em USD, via [TCGdex](https://tcgdex.dev)); links de busca na Liga e na MYP.
- **"Não é essa":** volta para a câmera. As outras candidatas aparecem no resultado para escolher com um toque.
- **Modo sling:** para lotes. Jogue as cartas uma a uma sob a câmera e cada carta reconhecida com confiança entra na lista sozinha. Na dúvida, a carta não entra: tire e ponha de novo.
- **Lista:** fica só no aparelho (localStorage). Dá para copiar como texto ou baixar em CSV.

Cartas em japonês ou coreano são identificadas pela arte e caem na versão em inglês equivalente.

## Como funciona

```
câmera → retícula viva (scanic, detecta os 4 cantos da carta)
       → corte com correção de perspectiva
       → POST /recognize (servidor)
           DINOv2-S @336 → similaridade de cosseno contra ~22 mil cartas
           → desempate por pHash quando dois prints têm a mesma arte
           → OCR do número (Tesseract) só quando o match é ambíguo
       → resultado + preço de referência + links de busca
```

O reconhecimento roda num servidor, não no celular. Rodar o modelo no navegador travava a tela e levava segundos por carta. No servidor quente, um scan leva ~0,5s.

Este repositório tem só o cliente web. O servidor de reconhecimento é um serviço separado.

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
| `VITE_RECOGNIZE_URL` | URL do servidor de reconhecimento. |
| `VITE_FEEDBACK_URL` | Opcional. Link do botão "Mandar feedback". |
| `VITE_TELEMETRY_URL` | Opcional. Endpoint que recebe a telemetria anônima (`POST`, corpo JSON em texto puro). Vazio = sem telemetria. |

## Contrato do servidor

`POST {VITE_RECOGNIZE_URL}/recognize`, `multipart/form-data`:

- `file`: JPEG da carta (lado maior até 1280px).
- `pre`: `1` quando a imagem já é só a carta recortada. Sem ele, o servidor detecta a carta na foto.

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

A versão publicada coleta telemetria anônima para medir o acerto do scanner: um ID aleatório por aparelho (gerado no localStorage), o país (vindo da CDN), a origem do acesso e os eventos de scan (cartas candidatas, confiança, tempo, "não é essa"). Não vai foto, IP nem nada que identifique a pessoa. O código está em `src/telemetry.ts`. Ele só envia quando o build define `VITE_TELEMETRY_URL`, então um build seu não manda nada.

## Créditos

- Detecção de cantos: [scanic](https://github.com/marquaye/scanic) (MIT).
- Dados, imagens e preços de referência: [TCGdex](https://tcgdex.dev).
- Pokémon e as imagens das cartas são marcas e propriedade de Nintendo, Creatures, GAME FREAK e The Pokémon Company. Este projeto não tem vínculo com elas.
- Nasceu do scanner do CartinhasDaJu, um app pessoal de coleção.

## Licença

[MIT](LICENSE)
