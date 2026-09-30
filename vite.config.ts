import { defaultClientConditions, defineConfig } from "vite";

// Em produção esses caminhos são rewrites do vercel.json. O proxy repete o
// mesmo mapeamento no `npm run dev`.
export default defineConfig({
  build: { target: "es2022" },
  // O onnxruntime-web vem, por padrão, com o WASM embutido no bundle (12 MB).
  // O app baixa o WASM do CDN (src/yolo.ts, wasmPaths), então usa a variante
  // sem o WASM embutido e o site não publica peso morto.
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  server: {
    proxy: {
      "/api/price-brl": {
        target: "https://cartinhaspoke.vercel.app",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/price-brl/, "/api/public/card-price-brl"),
      },
    },
  },
});
