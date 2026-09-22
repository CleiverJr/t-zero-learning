# Material para o relatório — Atividade DQN (CartPole-v1)

Este arquivo reúne tudo o que é preciso para escrever o relatório final: enunciado, dados, figuras, números e texto de referência.

---

## 1. Requisitos do enunciado (resumo)

- **Entregável:** um PDF de **no máximo 3 páginas** com três seções (Q1, Q2 e Q3). Cada seção segue a ordem **previsão → varredura → gráficos → explicação**. O PDF precisa trazer o **link do fork** com o `algorithms/dqn.py` completo.
- **Protocolo de cada questão:**
  1. **Previsão**, feita *antes* de rodar, em no máximo 2–3 frases: o que os gráficos devem mostrar e por quê.
  2. **Varredura**: pelo menos 2 valores além do baseline, indo de pequeno a grande; 2 seeds nas configurações importantes.
  3. **Gráficos**: todos os valores da varredura sobrepostos, com os runs identificados.
  4. **Explicação**: 1–2 parágrafos sobre o **mecanismo**, citando os gráficos ("o gráfico de q_values mostra…") e dizendo se a previsão se confirmou.
- **Q1** — `dqn.target_network_frequency`. Gráficos: `episodic_return_mean_last100`, `td_loss`, `q_values`. Perguntas: qual o papel da target network? Por que o retorno pode piorar enquanto a td_loss parece boa? O que q_values faz nos valores extremos?
- **Q2** — `dqn.buffer_size`, de minúsculo a generoso. Gráficos: `episodic_return_mean_last100`, `q_values`. Pergunta: quais **dois problemas distintos** um buffer muito pequeno causa? (correlação dentro do minibatch e quais dados a rede consegue rever)
- **Q3 (extra)** — hiperparâmetro livre. É preciso **justificar a escolha dos gráficos**. Não vale variar só a seed.
- Trabalho individual (Cleiver Batista da Silva Junior).

## 2. Links e identificação

- **Autor:** Cleiver Batista da Silva Junior (individual)
- **Fork:** https://github.com/CleiverJr/t-zero-learning
- **Implementação:** https://github.com/CleiverJr/t-zero-learning/blob/main/algorithms/dqn.py
- **Repositório original:** https://github.com/BrunoBSM/t-zero-learning
- **Projeto wandb (24 runs, já sincronizados):** https://wandb.ai/jr-cleiver-federal-university-of-goi-s/dqn-assignment
- Cada run tem URL própria: `https://wandb.ai/jr-cleiver-federal-university-of-goi-s/dqn-assignment/runs/<ID>` — os IDs estão nas tabelas de cada questão. Ex.: baseline seed 1 = https://wandb.ai/jr-cleiver-federal-university-of-goi-s/dqn-assignment/runs/5w0roffb

## 3. Figuras (já prontas, PNG 2400 px de largura, 200 dpi)

| Figura | Arquivo local | URL no GitHub |
|---|---|---|
| Fig. 1 — Q1 (retorno, td_loss log, q_values symlog) | `report/figs/q1_target_network_frequency.png` | https://github.com/CleiverJr/t-zero-learning/blob/main/report/figs/q1_target_network_frequency.png |
| Fig. 2 — Q2 (retorno, q_values) | `report/figs/q2_buffer_size.png` | https://github.com/CleiverJr/t-zero-learning/blob/main/report/figs/q2_buffer_size.png |
| Fig. 3 — Q3 (retorno, q_values symlog, td_loss log) | `report/figs/q3_gamma.png` | https://github.com/CleiverJr/t-zero-learning/blob/main/report/figs/q3_gamma.png |

Caminho local completo: `/Users/cleiver/Documents/Estudos/BIA/RL/t-zero-learning/report/figs/`

Convenções dos gráficos:
- Cada valor da varredura tem uma cor. **Seed 1 = linha cheia, seed 2 = tracejada.**
- A TD loss aparece suavizada com EMA 0.9 e os q_values com EMA 0.6, só para facilitar a leitura.
- **Linha pontilhada nos gráficos de Q = 1/(1−γ)**, o maior valor de Q possível no CartPole (recompensa 1 por passo). Na Q3 há uma linha dessas para cada γ, na cor correspondente.

O relatório final (3 páginas, gerado a partir de `report/relatorio.html`) está em `report/relatorio.pdf`, já com os links do fork e do projeto no wandb no cabeçalho.

## 4. Implementação (Partes 1 e 2)

Os 7 testes de `tests/test_dqn.py` passam. O baseline chega a retorno 500 (eval greedy 500 ± 0 nas duas seeds), em cerca de 1 min de CPU.

```python
# Parte 1a — ReplayBuffer.add
self.observations[self.pos] = obs
self.next_observations[self.pos] = next_obs
self.actions[self.pos] = action
self.rewards[self.pos] = reward
self.dones[self.pos] = done
self.pos = (self.pos + 1) % self.capacity          # ponteiro circular: FIFO quando cheio
self.size = min(self.size + 1, self.capacity)

# Parte 1b — ReplayBuffer.sample
idx = np.random.randint(0, self.size, size=batch_size)   # uniforme, com reposição, só slots preenchidos
return Batch(observations=..., actions=..., next_observations=..., rewards=..., dones=...)  # tensores em self.device

# Parte 2 — compute_td_targets
with torch.no_grad():
    next_q_max, _ = target_network(batch.next_observations).max(dim=1)
return batch.rewards.flatten() + gamma * next_q_max * (1.0 - batch.dones.flatten())
```

Pontos que valem mencionar:
- O max vem da **target network**.
- `(1 − done)` interrompe o bootstrap apenas em **terminações**. O truncamento em 500 passos não é terminação, então ali o bootstrap continua.
- `rewards` e `dones` têm shape (B,1) e são achatados para (B,). Sem isso, o broadcasting geraria um tensor (B,B).

## 5. Protocolo experimental

- Config base `configs/dqn_cartpole.yml`: 500k passos, lr 2.5e-4, buffer 10000, γ 0.99, tnf 500, batch 128, ε 1→0.05 em 50% do treino, learning_starts 10000, train_frequency 10.
- Cada run muda **só** o hiperparâmetro da questão, via `--override`. **Todas** as configurações rodaram com **2 seeds** (1 e 2), num total de 24 runs.
- Scripts no fork: `scripts/run_sweeps.sh` roda todos os runs e `scripts/plot_report.py` gera as figuras e o `report/runs_summary.csv`.

---

## 6. Q1 — `target_network_frequency` ∈ {1, 50, **500 (baseline)**, 10000, 50000}

### Previsão (rascunho — reescrever com suas palavras)
Com tnf=1 o alvo se move junto com a rede treinada. Espero que q_values cresça sem controle (superestimação que se realimenta) e que o retorno fique instável. Com tnf muito grande o alvo fica quase congelado: o treino deve ser estável, com td_loss baixa, mas lento, porque o valor só se propaga a cada sincronização.

### Dados

| tnf | retorno final (s1 / s2) | eval greedy (s1 / s2) | Q máx. | Q médio >400k | TD loss mediana | wandb run ids (s1, s2) |
|---|---|---|---|---|---|---|
| 1 | 215 / 198 | 223 / 208 | 2631 / 2931 | 69 / 70 | 60 / 55 | nxv89hnf, 68hrq80o |
| 50 | 500 / 130 | 500 / 479 | 1208 / 1053 | 103 / 212 | 4.8 / 3.0 | 7z89c72w, 51ye09dc |
| 500 (baseline) | 500 / 485 | 500 / 500 | 216 / 232 | 103 / 98 | 0.24 / 0.48 | 5w0roffb, ai4c257z |
| 10000 | 130 / 130 | 115 / 130 | 37.5 / 37.5 | 33 / 35 | 0.06 / 0.11 | yggog0vi, 1u52wg0d |
| 50000 | 372 / 111 | 467 / 107 | 10.3 / 9.9 | 9.5 / 9.0 | 0.016 / 0.014 | e9b9ui66, 09hrkdph |

Primeiro passo em que o retorno atinge ≥ 450:
- baseline: 244k / 245k
- tnf=50: 394k (s1) / 255k (s2)
- tnf=1: nunca (s1) / 279k (s2)
- tnf=10000 e tnf=50000: nunca

### O que os gráficos mostram (Fig. 1)
- **tnf=1:** Q chega a ~1300 aos 100k passos e a **~2600–2900 perto de 320–340k**, 26–29× acima do limite de 100. O retorno desaba (seed 1 de ~290 para ~20) **exatamente no pico de Q**, e a TD loss chega a ~10⁴.
- **tnf=50:** mesmo efeito, mais fraco. Q ~1200 no início; a seed 2 diverge de novo aos 470k (Q→1053, retorno 490→130). A diferença entre as seeds é grande.
- **tnf grande:** TD loss **minúscula** (mediana 0.016 com tnf=50000, 15× *menor* que o baseline), mas a política é ruim (~110–130). No gráfico de Q aparece uma **escada**: um degrau a cada sincronização. A TD loss também dá saltos a cada 50k passos.
- **Resultado quantitativo importante:** depois de k sincronizações, Q ≈ Σ_{i<k} γ^i.
  - tnf=50000: ~9 syncs, Q ≈ **9.5** (observado).
  - tnf=10000: ~49 syncs, (1−0.99⁴⁹)/0.01 = **38.9** (previsto) contra **37.5** (observado).

### Explicação (texto de referência)
O alvo de TD depende da própria Q que está sendo aprendida. A target network é uma cópia defasada que transforma cada janela de tnf passos numa regressão supervisionada comum, com rótulos fixos. Com tnf=1 essa proteção some: o max_a' escolhe sistematicamente os erros positivos de Q, e esses erros viram o rótulo do passo seguinte. A superestimação se realimenta até divergir, e a política colapsa junto. **Previsão confirmada**, inclusive a sensibilidade à seed.

Com tnf grande, a TD loss só mede a distância de Q a um alvo definido pela própria rede. Com o alvo congelado, a rede o ajusta com facilidade e a perda fica baixa **sem que Q esteja certo**. Por isso o retorno pode piorar enquanto a td_loss parece "boa". Cada sincronização propaga um único passo do backup de Bellman. A rede então "enxerga" só ~10–40 passos à frente e não distingue "falha em 50 passos" de "nunca falha". Previsão de "lento" confirmada, mas o efeito foi maior que o esperado: o treino não converge em 500k passos. A seed 1 com tnf=50000 subiu no fim, mas isso não se repetiu na seed 2.

---

## 7. Q2 — `buffer_size` ∈ {16, 64, 200, 2000, **10000 (baseline)**, 100000}

Os valores 16 e 64 foram acrescentados depois. Com 200 o treino ainda funcionava perfeitamente, então foi preciso descer mais para expor os efeitos.

### Previsão (rascunho — reescrever com suas palavras)
Um buffer minúsculo deve encher o minibatch de transições consecutivas e quase idênticas, e fazer a rede esquecer situações que não revisita (p. ex. estados de queda depois que aprende a equilibrar). O resultado esperado é oscilação do retorno e Q ruidoso. Um buffer enorme deve ser estável, mas mais lento, por manter por muito tempo os dados da fase aleatória (ε≈1). A expectativa era que 200 já fosse pequeno demais.

### Dados (batch_size = 128 em todos)

| buffer | retorno final (s1 / s2) | eval greedy (s1 / s2) | 1º passo com retorno ≥ 450 | Q máx. | Q >400k: média ± desvio | wandb run ids (s1, s2) |
|---|---|---|---|---|---|---|
| 16 | 240 / 415 | 312 / 284 | 428k / nunca | 205 / 257 | 131 ± 43 / 174 ± 58 | zyf6boas, uoeeq6pd |
| 64 | 385 / 284 | 500 / 500 | 438k / nunca | 216 / 230 | 158 ± 41 / 124 ± 40 | 11jpsr7u, 5yun2dex |
| 200 | 500 / 500 | 500 / 500 | 240k / 243k | 257 / 269 | 112 ± 5 / 109 ± 9 | sjlrp5u7, 4uwqh0tq |
| 2000 | 497 / 482 | 500 / 500 | 250k / 237k | 205 / 222 | 100 ± 5 / 101 ± 6 | nznb3q7v, 9aqf7dgw |
| 10000 (baseline) | 500 / 485 | 500 / 500 | 244k / 245k | 216 / 232 | 103 ± 1 / 98 ± 6 | 5w0roffb, ai4c257z |
| 100000 | 497 / 414 | 500 / 500 | 405k / 447k | 354 / 529 | 101 ± 2 / 108 ± 3 | k9jj7qlv, p0z6nmdh |

### O que os gráficos mostram (Fig. 2)
- **Buffers 16 e 64:** as curvas de Q são uma **faixa larga de ruído** (desvio de 40–58 na fase final, contra ~1–6 no baseline). A partir de ~350k, a média de Q fica em 124–174, **acima do limite de 100**. O retorno aprende e desaprende:
  - buffer=64 s1 cai de ~425 para ~175 aos ~340k e depois se recupera;
  - buffer=16 s1 cai de ~450 para ~240 no fim.
  - Nenhuma das 4 seeds se estabiliza em 500.
- **Buffers 200 a 10000:** praticamente idênticos, chegando a 500 por volta de 240–250k.
- **Buffer 100000:** leva quase o dobro de passos para chegar a 450 (405–447k contra ~245k) e tem o maior pico de Q (354 / 529, por volta de 150–160k). Depois converge normalmente (Q ~101–108, desvio de 2–3).

### Explicação (texto de referência)
**Problema 1 — correlação dentro do minibatch.** Com buffer=16 e batch=128, cada minibatch repete ~8 vezes as mesmas ~16 transições consecutivas, todas de um mesmo trecho de uma única trajetória. O gradiente deixa de estimar o erro médio no espaço de estados e passa a ajustar a rede à situação do momento, puxando Q para lá e para cá. É isso que produz a faixa de ruído no gráfico de q_values.

**Problema 2 — cobertura / esquecimento.** A rede só consegue rever os últimos 16–64 passos. Quando a política melhora, o buffer passa a conter só estados "bons", perto do centro. Os estados de queda somem do treino, seus valores deixam de ser corrigidos e a rede os esquece. Os Q dessas regiões derivam para cima (média acima de 100), e a política volta a errar e reaprende, o que gera os ciclos no retorno.

Previsão **parcialmente refutada** quanto ao limite: buffer=200 funciona tão bem quanto o baseline. No CartPole, 200 transições já cobrem vários episódios no início do treino e o espaço de estados relevante é pequeno. No extremo oposto, com buffer=100000 nada é descartado até 100k passos. Por volta de 150k, a maioria das amostras ainda vem da fase com ε alto: dados antigos e fora da política, e cada transição recente é revista com menos frequência. O resultado é convergência lenta e um pico de superestimação, mas sem instabilidade no fim (previsto).

---

## 8. Q3 (extra) — `gamma` ∈ {0.9, **0.99 (baseline)**, 0.999}

### Previsão (rascunho — reescrever com suas palavras)
Com recompensa 1 por passo, Q deve convergir para ≈1/(1−γ): 10, 100 e 1000. γ=0.9 tem horizonte efetivo de ~10 passos, curto para "ver" uma queda que se desenha em dezenas de passos, então espero desempenho pior. γ=0.999 deve funcionar, mas com alvos 10× maiores, TD loss maior e mais instabilidade.

### Gráficos escolhidos e por quê (parte da resposta!)
- `charts/episodic_return_mean_last100`: é o resultado final.
- `losses/q_values`, com linhas em 1/(1−γ): mostra diretamente o que γ muda (a escala e o horizonte do valor) e permite verificar a convergência para o valor teórico.
- `losses/td_loss` em escala log: mostra que **a TD loss não é comparável entre valores de γ**, porque cresce com a escala dos alvos (o erro quadrático acompanha a magnitude de Q ao quadrado).
- `charts/epsilon` **não foi incluído** porque o cronograma de exploração é idêntico em todos os runs.

### Dados

| γ | retorno final (s1 / s2) | eval greedy (s1 / s2) | 1º passo com retorno ≥ 450 | Q médio >400k (s1 / s2) | TD loss mediana | wandb run ids (s1, s2) |
|---|---|---|---|---|---|---|
| 0.9 | 355 / 272 | 161 / 285 | 443k / nunca | 10.0 / 9.8 | 0.034 / 0.024 | 0u39md1a, x5wx63an |
| 0.99 (baseline) | 500 / 485 | 500 / 500 | 244k / 245k | 103 / 98 | 0.24 / 0.48 | 5w0roffb, ai4c257z |
| 0.999 | 498 / 500 | 500 / 500 | 248k / 251k | 1252 / 1310 | 6.5 / 5.9 | 6g6ode8m, x2ok2xpa |

### O que os gráficos mostram (Fig. 3)
- Q converge para **10.0 / ~100 / ~1250–1310**. Com γ=0.999 há uma superestimação de 25–30% acima de 1000.
- Com γ=0.9, Q satura em 10 já por volta de 50k passos. O retorno sobe devagar, oscila e termina em 272–355.
- Com γ=0.999, a curva de retorno acompanha o baseline. A TD loss fica uma ordem de grandeza acima durante todo o treino, com picos de ~10³ aos 400k e 470k que não derrubam o retorno.

### Explicação (texto de referência)
Um estado que falha em T passos vale (1−γ^T)/(1−γ). Com γ=0.9 e T ≥ 30, isso já é ≥ 9.6: a diferença entre "cai daqui a 30 passos" e "nunca cai" é menor que 0.4, abaixo do ruído da regressão. O agente só aprende a evitar quedas iminentes e não corrige derivas lentas (p. ex. o carrinho indo para a borda). **Previsão confirmada.**

Com γ=0.999, o horizonte (~1000) cobre o episódio inteiro de 500 passos e o aprendizado acompanha o baseline. O custo aparece na td_loss: alvos 10× maiores fazem a mesma taxa de aprendizado dar passos efetivamente maiores. A superestimação de 25–30% é consistente com o viés do max amplificado. A instabilidade prevista aparece na perda, mas nesta tarefa não chegou a afetar a política.

---

## 9. Observações para quem for escrever o relatório

- **Previsões:** o protocolo exige previsões feitas *antes* de rodar. Os textos de "Previsão" acima são rascunhos baseados na teoria, escritos depois de parte dos resultados já ser conhecida. Reescreva com suas palavras e marque com honestidade o que se confirmou e o que não.
- **Limite de 3 páginas:** a versão de referência (`report/relatorio.pdf`) cabe em exatamente 3 páginas A4, com fonte de 8.6 pt, figuras em largura total (Q2 a 78%) e tabelas compactas.
- Sempre citar os gráficos na explicação ("o gráfico de q_values mostra…"), como pede o enunciado.
- O cabeçalho precisa do link do fork: https://github.com/CleiverJr/t-zero-learning
