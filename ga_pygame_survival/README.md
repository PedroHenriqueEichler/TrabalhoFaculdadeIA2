
# Evolução Inteligente: AG + Pygame (Sobrevivência)

Projeto acadêmico: algoritmo genético otimiza agentes num jogo estilo "Flappy".

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

## Como rodar
1) Instalar dependências:
```bash
pip install -r requirements.txt
```
2) Treinar (headless), salvando métricas e melhor genoma:
```bash
python jogo_genetico.py --train --gens 30 --pop 50 --out runs/run1 --plot
```
- Saídas:
  - `runs/run1/fitness.csv` (histórico de fitness)
  - `runs/run1/fitness.png` (gráfico, se usar `--plot`)
  - `runs/run1/best_genome.json`

3) Visualizar o melhor agente no Pygame (usa o JSON salvo):
```bash
python jogo_genetico.py --play --out runs/run1
```

## Explicação rápida
- **Genes** `[w1, w2, w3, threshold, jump_impulse]` controlam uma política linear que decide quando pular.
- **Fitness**: tempo de sobrevivência + bônus ao atravessar o vão.
- **GA**: elitismo, torneio, crossover BLX-α e mutação gaussiana.

## Dicas
- Trave a semente (via `random.seed`) se quiser reprodutibilidade.
- Aumente `--gens` se a evolução “empacar”.
- `--pop` maior = mais diversidade, porém mais tempo.
- O modo `--play` reinicia automaticamente quando morre, para observação contínua.

## Estrutura
```
.
├── jogo_genetico.py
├── requirements.txt
└── runs/
```
