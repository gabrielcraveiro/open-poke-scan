# Arquitetura e decisões

Como o open-poke-scan funciona por dentro e por que ele é assim. Para rodar e publicar, veja o [README](../README.md). Para montar o seu próprio servidor de reconhecimento, veja [`server/README.md`](../server/README.md).

- Site: https://open-poke-scan.vercel.app
- Código: https://github.com/gabrielcraveiro/open-poke-scan

## Visão geral

```
navegador (este repo, site estático na Vercel)
  câmera → retícula viva → corte da carta → POST /recognize ──► servidor de reconhecimento (Fly.io)
  resultado ← preço em R$ (GET /api/price-brl, rewrite da Vercel) ◄── serviço de preços
  eventos anônimos (sendBeacon) ──► endpoint de telemetria (só na versão publicada)
```

O site não tem banco, login nem backend próprio. Tudo que ele guarda (a lista de cartas escaneadas, a preferência do modo sling) fica no `localStorage` do aparelho.

## Mapa do código

| Arquivo | O que faz |
|---|---|
| `src/main.ts` | Liga a UI ao scanner: tela de resultado, alternativas, modo sling, lista, exportação, eventos. |
| `src/scanner.ts` | Loop da câmera (tick de 150ms): retícula viva, gates de estabilidade e foco, captura, warp, edge-trigger do sling. |
| `src/quad.ts` | Geometria do contorno da carta: ordenar cantos, rejeitar detecções absurdas, suavizar, desenhar. |
| `src/yolo.ts` | Detector de cantos YOLOv8-pose, rodado uma vez na foto capturada. |
| `src/camera.ts` | Escolha da lente traseira com foco automático, foco no centro e ajustes de câmera que mantêm a resolução. |
| `src/pixels.ts` | Métricas baratas sobre o proxy de 160px: diferença entre frames, nitidez (Brenner), presença de carta, reflexo. |
| `src/recognize.ts` | Cliente do `POST /recognize`: orçamento de tempo quente/frio, reenvio só em falha rápida, URL da imagem. |
| `src/prices.ts` | Preço em R$ por carta, com cache em memória. |
| `src/links.ts` | Monta os links de busca na Liga e na MYP (só a URL; nenhum dado é lido desses sites). |
| `src/session.ts` | Lista local de cartas escaneadas; exporta texto e CSV. |
| `src/telemetry.ts` | Eventos anônimos de uso; desligado sem `VITE_TELEMETRY_URL`. |
| `server/` | Servidor de reconhecimento para rodar do zero (FastAPI + scripts de modelo e índice). |

## Fluxo de um scan

1. **Câmera.** Abre a traseira em 1920×1440. Se o navegador informa que a lente não tem foco automático, troca para a traseira de menor número que tenha (em celular com várias lentes, o Chrome às vezes abre a grande-angular de foco fixo). Liga o foco contínuo no centro.
2. **Retícula viva.** A cada ~200ms, o [scanic](https://github.com/marquaye/scanic) clássico procura os 4 cantos da carta num proxy de 480px do quadro inteiro. Um salto de posição só é aceito quando duas detecções seguidas confirmam o lugar novo, senão o contorno pularia para reflexos.
3. **Gates.** A captura dispara quando o contorno fica parado e a imagem nítida: nitidez a 85% do pico visto, com o pico decaindo aos poucos para não travar em "Focando…". Se os gates nunca abrem (mão tremendo), dispara mesmo assim depois de 3s (2,5s no sling).
4. **Foto e corte.** A foto sai do `takePhoto()` (até 2400px, no enquadramento do vídeo). No Firefox e no iOS, que não têm `ImageCapture`, sai do quadro do vídeo.
   - O YOLO de cantos roda uma vez nessa foto e dá o contorno do corte, mesmo quando a retícula não travou. Enquanto o modelo baixa, vale o scanic clássico na foto.
   - A carta é recortada com correção de perspectiva e reamostrada para 63:88. A retícula mostra o contorno do corte.
   - Se o recorte sai com proporção fora de 0,62–0,85, tenta o contorno da retícula. Se também falhar, vale a caixa do contorno com 6% de folga, e o servidor acha a carta lá dentro. O corte fixo da retícula só entra sem contorno nenhum.
5. **Reconhecimento.** A carta vai em JPEG de até 900px com `pre=1`, mais o rodapé (28% de baixo) em resolução cheia no campo `footer`, para o OCR ler o número.
6. **Resultado.** No disparo com a carta na mão, a imagem congela e a prévia da carta aparece na hora, antes da foto nítida e do servidor (~2s). Servidor confiante: tela com a carta, o preço, as alternativas e os links. Sem confiança: "Qual destas é a sua?", com as 6 candidatas grandes e o set em destaque quando há nomes repetidos (mesma arte em outro set); um toque escolhe. Modo sling: entra direto na lista, se o servidor estiver confiante. Quadro sem carta (`not_card`): aviso e volta para a câmera.

## Decisões

### Detector leve no overlay, YOLO só na foto
O scanic clássico é rápido (~270ms no celular), mas escolhe o maior retângulo de bordas fortes e perde a carta com dedo no canto. O YOLO de cantos acha a borda externa quase sempre, mas leva ~400-1000ms por quadro no celular: contínuo, o contorno ficava para trás da mão. Então o clássico cuida da retícula e da estabilidade, e o YOLO roda uma vez, na foto (~600ms). No teste de 2026-09-30, 9 de 10 scans foram adicionados, sem nenhum "não é essa". O scanic ML (DocCornerNet) foi testado e descartado: põe os cantos para dentro da carta e corta a borda e o número.

O modelo é o [duclvQ/tcg-card-detector](https://huggingface.co/duclvQ/tcg-card-detector) (AGPL-3.0), baixado do Hugging Face fixado num commit. Ele não faz parte deste repositório.

### Reconhecimento no servidor, não no navegador
O modelo roda no navegador também (onnxruntime-web), e essa foi a primeira versão do scanner de origem. Rodar o DINOv2 no celular travava a tela e levava segundos por carta. No servidor quente, o scan leva ~0,35–0,5s. O custo é depender da rede e do servidor.

### Servidor compartilhado com o app de origem
A versão publicada usa o mesmo servidor do Fly.io do app que deu origem a este projeto. Foi escolha do dono: menos infra para manter. O risco conhecido: o servidor tem 1 vCPU, e um pico de acessos deixa os dois apps lentos. A saída, se isso acontecer, é um app separado no Fly com a mesma imagem.

### Vanilla TypeScript, sem React
O loop da câmera roda a cada 150ms e mexe no DOM direto (contorno SVG, barra de progresso). Um framework de re-render nesse ritmo só adicionaria custo. O site inteiro tem ~55KB gzip.

### Limiares vindos de telemetria real
Os números do scanner (proporção 0,62–0,85, limiar de diferença 40, presença 16, foco 85%) foram calibrados com scans reais no app de origem. Os comentários no código dizem de onde veio cada um. Mudar um deles muda o acerto, não só o visual.

### Modo sling conservador
- Só adiciona sozinho quando o servidor responde `confident`: número+total conferido pelo OCR, ou cosseno ≥ 0,72 com folga ≥ 0,03 sobre a 2ª carta (~99% de precisão num eval de 160 scans reais). Na dúvida a carta não entra: um frame ruim durante a troca de carta já adicionou a carta errada. O corte antigo (cosseno ≥ 0,62) aceitava cartas de arte igual em sets diferentes.
- A mesma carta só é somada de novo depois de sair do quadro. Uma janela de tempo não servia: a carta parada no quadro furava a janela.
- Rearma com o quadro vazio, com carta nova (diferença grande contra a carta capturada) ou depois de 4s desarmado.

### "Não é essa" volta para a câmera
A primeira versão abria as alternativas embaixo do resultado. No primeiro teste no celular, a expectativa foi voltar para a câmera. Agora o botão fecha o resultado e volta a escanear. As alternativas ficam sempre visíveis no resultado, para quem reconhece a carta certa.

### Cartas em japonês e coreano
O índice tem só cartas em inglês. Uma carta JP ou coreana é reconhecida pela arte e cai na versão em inglês equivalente, que foi o comportamento pedido. O número JP não atrapalha: a fusão com o OCR só age entre as 25 cartas mais parecidas e exige que o total impresso bata.

### Preço só em real
O público é brasileiro. A primeira versão mostrava também EUR e USD (TCGdex), e isso foi tirado.

### Chamadas de preço pela mesma origem
`/api/price-brl` é um rewrite da Vercel (`vercel.json`) e, no `npm run dev`, um proxy do Vite. Sem isso, a chamada precisaria de CORS. Uma versão anterior buscava o TCGdex direto do navegador, e a CDN dele às vezes respondia sem o header de CORS: o preço simplesmente não aparecia.

### Sem conta, lista no aparelho
Para pedir feedback, qualquer atrito de cadastro espanta. A lista fica no `localStorage` e sai como texto ou CSV.

### Telemetria anônima, só na versão publicada
- Envia: um ID aleatório por aparelho, o país (vindo da CDN), a origem do acesso e os eventos de scan (candidatas, confiança, tempo, "não é essa", "escanear outra" como `skip`, cliques nos links, exportações).
- Não envia: foto, IP ou qualquer dado pessoal.
- `src/telemetry.ts` só envia quando o build define `VITE_TELEMETRY_URL`. Um fork não manda nada para ninguém.
- Os eventos vão por `sendBeacon` com corpo em texto puro: não há preflight de CORS e o envio sobrevive ao fechar a aba.

### Privacidade das fotos
O servidor guarda a foto recebida por no máximo ~6 horas, para depurar erros de reconhecimento. O aviso está na tela inicial. O servidor de `server/` só guarda fotos se `DEBUG_DIR` estiver definido.

### Servidor aberto, modelo e índice gerados por quem roda
- O modelo é o [DINOv2-small](https://huggingface.co/facebook/dinov2-small) público, sem treino extra. `server/scripts/export_model.py` gera um ONNX idêntico ao do servidor publicado: cosseno 1,0 entre os embeddings nas cartas testadas.
- Para exportar em 336×336, o position embedding é pré-calculado antes do export. O onnxruntime não aceita o Resize bicúbico que o DINOv2 faz para reduzir.
- O índice é gerado da API pública do TCGdex por `server/scripts/build_index.py`, na máquina de quem roda. O repo não redistribui imagens nem índice: as imagens são da The Pokémon Company.
- O servidor de `server/` não tem o detector YOLO do servidor publicado. Ele só entra quando o site não consegue recortar a carta (~17% dos scans no app de origem), e a licença AGPL dele obrigaria o projeto inteiro a ser AGPL.

## Configuração

| Variável (build) | Uso |
|---|---|
| `VITE_RECOGNIZE_URL` | URL do servidor de reconhecimento. Padrão: o servidor publicado. |
| `VITE_TELEMETRY_URL` | Endpoint da telemetria. Vazio = sem telemetria. |
| `VITE_FEEDBACK_URL` | Link do botão "Mandar feedback". Vazio = sem botão. |

Na Vercel, `VITE_RECOGNIZE_URL` e `VITE_TELEMETRY_URL` estão definidas em Production. Cada push na `main` publica sozinho.

## Limitações conhecidas

- Cartas fora do TCGdex só são reconhecidas pelo servidor publicado, que tem alguns sets a mais.
- Sem preço para cartas que o serviço de preços não cobre.
- Pico de acessos: ver "Servidor compartilhado", acima.
- A câmera só funciona em HTTPS (ou `localhost`).

## Histórico

| Data | Mudança |
|---|---|
| 2026-09-28 | Versão inicial: scan, sling, lista local, links de busca, preço de referência. |
| 2026-09-28 | Preço em R$; "Não é essa" volta para a câmera; chamadas de preço pela mesma origem. |
| 2026-09-29 | Cantos em ordem cíclica (carta em pé até ±85°), recorte 63:88, re-detecção só perto do contorno da tela, caixa do contorno como reserva, sling pelo `confident` do servidor, aviso de quadro sem carta. |
| 2026-09-29 | Contorno deitado descartado; "Escanear outra" registra `skip`, separado de "não é essa". |
| 2026-09-29 | Ordem de cantos e filtro do contorno voltam ao original (as mudanças acima pioraram a carta na mão). Foto nítida do `takePhoto`, rodapé em alta resolução para o OCR. |
| 2026-09-30 | Lente com foco automático; YOLO de cantos na foto capturada; modos de teste do detector removidos. |
| 2026-09-30 | Prévia instantânea no disparo; escolha entre as candidatas quando o servidor não está confiante. |
| 2026-10-01 | `server/` alcança o servidor publicado: filtro do TCG Pocket, `not_card` e confiança calibrada, OCR nos cantos do `footer`, OCR pulado quando já é confiante, penalidade das cartas ímã a partir de fotos reais (`scripts/build_hub_penalty.py`). |
| 2026-09-28 | Telemetria anônima (só na versão publicada). |
| 2026-09-28 | `server/` para rodar o reconhecimento do zero. |
| 2026-09-28 | Só preço em real (sem EUR/USD). |
