
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolução Inteligente: AG + Pygame (estilo Flappy)
- Treina headless e salva melhor genoma
- Renderiza o melhor agente em modo visual
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# Importes opcionais (só exigidos no modo --play ou --viz)
try:
    import pygame
except Exception:
    pygame = None

try:
    import numpy as np
except Exception:
    np = None

# -------------- Configurações de jogo --------------
WIDTH, HEIGHT = 800, 600
FPS = 60
GROUND_Y = HEIGHT - 40

OBSTACLE_WIDTH = 70
GAP_HEIGHT = 180
OBSTACLE_SPEED = 5
OBSTACLE_SPACING = 260  # distância entre pilares

AGENT_X = 200
GRAVITY = 0.6
JUMP_IMPULSE_DEFAULT = -10.5
TERMINAL_VEL = 12

# -------------- Representação do Agente / Genoma --------------
# Política linear: decision = w1*dy + w2*dx_next_gap + w3*(gap_center_y - y) - threshold
# Se decision > 0 -> pular (aplicar impulso)
# Genes = [w1, w2, w3, threshold, jump_impulse]
GENE_LOW = [-2.0, -0.02, -0.03, -12.0, -15.0]
GENE_HIGH= [ 2.0,  0.02,  0.03,  12.0,  -6.0]

@dataclass
class Genome:
    genes: List[float]

    def clamp(self):
        for i, (lo, hi) in enumerate(zip(GENE_LOW, GENE_HIGH)):
            self.genes[i] = max(lo, min(hi, self.genes[i]))

    @staticmethod
    def random():
        return Genome([random.uniform(lo, hi) for lo, hi in zip(GENE_LOW, GENE_HIGH)])

    @staticmethod
    def crossover(a: 'Genome', b: 'Genome') -> 'Genome':
        # BLX-alpha (blend) simples
        alpha = 0.4
        child = []
        for ga, gb, lo, hi in zip(a.genes, b.genes, GENE_LOW, GENE_HIGH):
            cmin, cmax = min(ga, gb), max(ga, gb)
            I = cmax - cmin
            low = max(lo, cmin - alpha*I)
            high = min(hi, cmax + alpha*I)
            child.append(random.uniform(low, high))
        return Genome(child)

    def mutate(self, prob=0.1, sigma=0.15):
        for i in range(len(self.genes)):
            if random.random() < prob:
                self.genes[i] += random.gauss(0, sigma * (GENE_HIGH[i] - GENE_LOW[i]))
        self.clamp()


# -------------- Ambiente headless (sem Pygame obrigatório) --------------
@dataclass
class ObstacleState:
    x: float
    gap_y: float  # centro do vão (em y)

@dataclass
class AgentState:
    y: float = HEIGHT//2
    dy: float = 0.0
    alive: bool = True
    score: float = 0.0

def generate_track(seed: int, max_time_steps: int) -> List[ObstacleState]:
    random.seed(seed)
    obstacles = []
    x = WIDTH + 200
    while x < WIDTH + 200 + max_time_steps * (OBSTACLE_SPEED):
        gap_center = random.uniform(150, HEIGHT-200)
        obstacles.append(ObstacleState(x=x, gap_y=gap_center))
        x += OBSTACLE_SPACING
    return obstacles

def step_physics(agent: AgentState):
    agent.dy = max(-TERMINAL_VEL, min(TERMINAL_VEL, agent.dy + GRAVITY))
    agent.y += agent.dy
    if agent.y < 0: agent.y = 0
    if agent.y > GROUND_Y: agent.y = GROUND_Y

def collide(agent: AgentState, obstacles: List[ObstacleState]) -> bool:
    # colisão vertical com o vão
    ax1, ax2 = AGENT_X - 15, AGENT_X + 15
    ay1, ay2 = agent.y - 15, agent.y + 15
    for obs in obstacles:
        # retângulos dos pilares
        ox1, ox2 = obs.x - OBSTACLE_WIDTH//2, obs.x + OBSTACLE_WIDTH//2
        gap_top = obs.gap_y - GAP_HEIGHT/2
        gap_bottom = obs.gap_y + GAP_HEIGHT/2

        # checa sobreposição no eixo x
        overlap_x = (ax1 < ox2) and (ax2 > ox1)
        if overlap_x:
            # se fora do vão vertical => bateu
            if (ay1 < gap_top) or (ay2 > gap_bottom):
                return True
    # chão/teto
    if ay2 >= GROUND_Y or ay1 <= 0:
        return True
    return False

def advance_obstacles(obstacles: List[ObstacleState]):
    for obs in obstacles:
        obs.x -= OBSTACLE_SPEED

def get_next_obstacle(obstacles: List[ObstacleState]) -> ObstacleState:
    # próximo obstáculo à frente do agente
    future = [o for o in obstacles if o.x + OBSTACLE_WIDTH//2 >= AGENT_X - 5]
    if not future:
        return obstacles[-1]
    return sorted(future, key=lambda o: o.x)[0]

def policy_action(genome: Genome, agent: AgentState, next_obs: ObstacleState) -> bool:
    w1, w2, w3, thr, jump_impulse = genome.genes
    dx = next_obs.x - AGENT_X
    dy = agent.dy
    target = next_obs.gap_y - agent.y
    decision = w1*dy + w2*dx + w3*target - thr
    return decision > 0.0

def simulate_agent(genome: Genome, seed: int, max_steps: int=2500) -> float:
    obstacles = generate_track(seed, max_steps)
    agent = AgentState()
    t = 0
    while t < max_steps and agent.alive:
        # Obstáculos andam
        advance_obstacles(obstacles)
        # Escolhe ação
        next_obs = get_next_obstacle(obstacles)
        if policy_action(genome, agent, next_obs):
            # gene[4] é impulso (negativo = pulo para cima)
            agent.dy += genome.genes[4]
        # Física
        step_physics(agent)
        # Colisão
        if collide(agent, obstacles):
            agent.alive = False
        # Pontuação (sobrevivência + progressão)
        agent.score += 1.0
        if next_obs.x < AGENT_X and abs(agent.y - next_obs.gap_y) < GAP_HEIGHT/2:
            agent.score += 50.0  # bônus por passar o vão
        t += 1
    return agent.score

# -------------- Algoritmo Genético --------------
@dataclass
class GAConfig:
    population: int = 60
    generations: int = 40
    elite_frac: float = 0.1
    tournament_k: int = 3
    mutation_prob: float = 0.15
    mutation_sigma: float = 0.08
    seeds_per_eval: int = 3  # média de várias pistas para reduzir sorte
    max_steps: int = 2600

@dataclass
class GAResult:
    best_genome: Genome
    history_best: List[float] = field(default_factory=list)
    history_mean: List[float] = field(default_factory=list)

def evaluate(genome: Genome, cfg: GAConfig) -> float:
    # média em vários seeds p/ estabilidade
    seeds = [random.randint(0, 10**9) for _ in range(cfg.seeds_per_eval)]
    scores = [simulate_agent(genome, s, cfg.max_steps) for s in seeds]
    return sum(scores)/len(scores)

def selection_tournament(pop: List[Genome], fitness: List[float], k: int) -> Genome:
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best]

def run_ga(cfg: GAConfig, out_dir: str) -> GAResult:
    os.makedirs(out_dir, exist_ok=True)
    pop = [Genome.random() for _ in range(cfg.population)]
    best_global = None
    best_fit_global = -1e9

    # CSV log
    csv_path = os.path.join(out_dir, "fitness.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("generation,mean_fitness,best_fitness\n")

    history_best, history_mean = [], []

    for g in range(cfg.generations):
        fitness = [evaluate(ind, cfg) for ind in pop]
        gen_best_idx = max(range(len(pop)), key=lambda i: fitness[i])
        gen_best = pop[gen_best_idx]
        gen_best_fit = fitness[gen_best_idx]
        mean_fit = sum(fitness)/len(fitness)

        if gen_best_fit > best_fit_global:
            best_fit_global = gen_best_fit
            best_global = Genome(list(gen_best.genes))

        history_best.append(gen_best_fit)
        history_mean.append(mean_fit)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{g},{mean_fit:.3f},{gen_best_fit:.3f}\n")

        # Nova população
        new_pop = []
        # elitismo
        elite_n = max(1, int(cfg.elite_frac * cfg.population))
        elite_sorted = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)[:elite_n]
        for i in elite_sorted:
            new_pop.append(Genome(list(pop[i].genes)))
        # reprodução
        while len(new_pop) < cfg.population:
            parent1 = selection_tournament(pop, fitness, cfg.tournament_k)
            parent2 = selection_tournament(pop, fitness, cfg.tournament_k)
            child = Genome.crossover(parent1, parent2)
            child.mutate(cfg.mutation_prob, cfg.mutation_sigma)
            new_pop.append(child)
        pop = new_pop
        print(f"Geração {g+1}/{cfg.generations} | melhor: {gen_best_fit:.1f} | média: {mean_fit:.1f}")

    # Salva melhor genoma
    best_path = os.path.join(out_dir, "best_genome.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_global.genes, f, ensure_ascii=False, indent=2)
    return GAResult(best_genome=best_global, history_best=history_best, history_mean=history_mean)

# -------------- Visualização com Pygame (modo play) --------------
def run_visual(best_genome_path: str):
    if pygame is None:
        raise RuntimeError("pygame não está instalado. Rode: pip install pygame")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AG Pygame - Melhor Agente")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    # Carrega melhor genoma
    with open(best_genome_path, "r", encoding="utf-8") as f:
        genes = json.load(f)
    genome = Genome(genes)
    seed = random.randint(0, 10**9)
    obstacles = generate_track(seed, 999999)

    agent = AgentState()
    running = True
    score = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        advance_obstacles(obstacles)
        next_obs = get_next_obstacle(obstacles)
        if policy_action(genome, agent, next_obs):
            agent.dy += genome.genes[4]
        step_physics(agent)
        if collide(agent, obstacles):
            # reinicia automaticamente
            seed = random.randint(0, 10**9)
            obstacles = generate_track(seed, 999999)
            agent = AgentState()
            score = 0.0

        score += 1.0
        if next_obs.x < AGENT_X and abs(agent.y - next_obs.gap_y) < GAP_HEIGHT/2:
            score += 50.0

        # draw
        screen.fill((30, 30, 35))
        # chão
        pygame.draw.rect(screen, (60, 60, 60), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        # obstáculos
        for obs in obstacles:
            x = int(obs.x - OBSTACLE_WIDTH//2)
            top_rect = pygame.Rect(x, 0, OBSTACLE_WIDTH, int(obs.gap_y - GAP_HEIGHT/2))
            bot_rect = pygame.Rect(x, int(obs.gap_y + GAP_HEIGHT/2), OBSTACLE_WIDTH, HEIGHT - int(obs.gap_y + GAP_HEIGHT/2))
            pygame.draw.rect(screen, (100, 200, 120), top_rect)
            pygame.draw.rect(screen, (100, 200, 120), bot_rect)

        # agente
        pygame.draw.circle(screen, (230, 230, 70), (AGENT_X, int(agent.y)), 15)

        # UI
        txt = font.render(f"Genes: {', '.join(f'{g:.2f}' for g in genome.genes)} | Score: {int(score)}", True, (240,240,240))
        screen.blit(txt, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

# -------------- Plot do fitness (opcional) --------------
def plot_fitness(csv_path: str, out_png: str):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        raise RuntimeError("matplotlib/pandas não instalados. Rode: pip install matplotlib pandas")

    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df['generation'], df['mean_fitness'], label='média')
    plt.plot(df['generation'], df['best_fitness'], label='melhor')
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Evolução do Fitness")
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    print(f"Gráfico salvo em {out_png}")

# -------------- CLI --------------
def main():
    parser = argparse.ArgumentParser(description="AG + Pygame Survival")
    parser.add_argument("--train", action="store_true", help="rodar treinamento headless")
    parser.add_argument("--play", action="store_true", help="visualizar melhor agente com Pygame")
    parser.add_argument("--out", type=str, default="runs/run1", help="pasta de saída (treino)")
    parser.add_argument("--gens", type=int, default=30, help="nº de gerações")
    parser.add_argument("--pop", type=int, default=50, help="tamanho da população")
    parser.add_argument("--plot", action="store_true", help="gera gráfico de fitness após treino")
    args = parser.parse_args()

    if args.train:
        cfg = GAConfig(population=args.pop, generations=args.gens)
        res = run_ga(cfg, args.out)
        if args.plot:
            csv_path = os.path.join(args.out, "fitness.csv")
            out_png = os.path.join(args.out, "fitness.png")
            plot_fitness(csv_path, out_png)
        print("Treino finalizado.")
        print(f"Melhor genoma salvo em: {os.path.join(args.out, 'best_genome.json')}")
        return

    if args.play:
        best_path = os.path.join(args.out, "best_genome.json")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {best_path}. Treine primeiro com --train.")
        run_visual(best_path)
        return

    # padrão: treinar
    cfg = GAConfig(population=args.pop, generations=args.gens)
    run_ga(cfg, args.out)
    print("Concluído (modo padrão: treino). Use --play para visualizar.")

if __name__ == "__main__":
    main()
