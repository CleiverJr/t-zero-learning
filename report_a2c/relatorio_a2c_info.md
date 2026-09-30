# Material para o relatório — Atividade A2C (CartPole-v1 + extra LunarLander-v3)

Tudo o que é preciso para escrever ou refazer o relatório: enunciado, implementação, dados, figuras e texto de referência.

---

## 1. Requisitos do enunciado (resumo)

- **Entregável:** PDF de **no máximo 3 páginas**, com as **quatro** questões no formato **previsão → varredura → gráficos → explicação**, e link para o fork com `algorithms/a2c.py` e `networks/discrete_actor_critic.py` completos.
- **Q1 — `num_envs`** (baseline 8; testar 1 e algo bem maior). Gráficos: `episodic_return_mean_last100`, `policy_loss`, `SPS`. Deve responder: por que vários atores ajudam um método on-policy; o que o ruído do `policy_loss` com 1 ambiente diz sobre o gradiente; **separar o efeito do paralelismo do efeito do tamanho do lote** (lote = `num_envs × num_steps`); o que o paralelismo compra em tempo.
- **Q2 — `a2c.num_steps`** (baseline 5; testar 1, 32, 128). Gráficos: `episodic_return_mean_last100`, `value_loss`, `explained_variance`. Deve responder: quem tem mais viés e quem tem mais variância; efeito no crítico e por que o ator sofre junto; **a armadilha da `explained_variance` parecer ótima com `num_steps=1`**.
- **Q3 — `a2c.ent_coef`** (baseline 0,01; testar 0 e 0,1; 2 seeds no 0). Gráficos: `episodic_return_mean_last100`, `entropy`. Deve responder: o que o bônus impede; por que o retorno estaciona com coeficiente grande; por que o CartPole é benevolente com `ent_coef=0` e o que mudaria com mais ações ou recompensa rara.
- **Q4 — `a2c.use_baseline=false`** (2 seeds). Gráficos: `episodic_return_mean_last100`, `advantage_std`, `advantage_mean`, `entropy`. Deve confirmar que o baseline não muda o gradiente esperado mas muda a variância, e explicar por que um peso sempre positivo colapsa a entropia.
- **Extra — LunarLander-v3:** treinar, reportar a curva e os gráficos que justificam as escolhas, dizer o que mudou em relação ao CartPole e por quê. Chegar perto sem resolver vale, desde que se explique o que limita.

## 2. Links

- **Autor:** Cleiver Batista da Silva Junior (individual)
- **Fork:** https://github.com/CleiverJr/t-zero-learning
- **Arquivos da implementação:** [`algorithms/a2c.py`](https://github.com/CleiverJr/t-zero-learning/blob/main/algorithms/a2c.py) e [`networks/discrete_actor_critic.py`](https://github.com/CleiverJr/t-zero-learning/blob/main/networks/discrete_actor_critic.py)
- **Projeto wandb:** https://wandb.ai/jr-cleiver-federal-university-of-goi-s/a2c-assignment (runs online; cada run em `.../runs/<ID>`, e os IDs estão nas tabelas)

## 3. Arquivos gerados

| O quê | Caminho |
|---|---|
| Relatório pronto (3 páginas) | `report_a2c/relatorio_a2c.pdf` (fonte: `relatorio_a2c.html`) |
| Figuras | `report_a2c/figs/q1_num_envs.png`, `q2_num_steps.png`, `q3_ent_coef.png`, `q4_use_baseline.png`, `extra_lunarlander.png` |
| Tabela de todos os runs | `report_a2c/runs_summary.csv` |
| Benchmark limpo de SPS | `report_a2c/sps_benchmark.csv` |
| Scripts | `scripts/run_sweeps_a2c.sh` (varreduras), `scripts/run_lunar_screen.sh` (LunarLander), `scripts/plot_report_a2c.py` (figuras) |
| Config do extra | `configs/a2c_lunarlander.yml` |

Convenções das figuras: uma cor por valor da varredura; **seed 1 = linha cheia, seed 2 = tracejada**; EMA nas curvas ruidosas, exceto `policy_loss`, mostrada crua porque o ruído é o objeto de estudo; linha pontilhada em ln 2 (entropia uniforme no CartPole) e em ln 4 (LunarLander).

## 4. Implementação (Partes 1, 2 e 3) — 14/14 testes passam

```python
# Parte 1 — compute_n_step_returns (algorithms/a2c.py)
returns = torch.zeros_like(rewards)
R = next_value
for t in reversed(range(rewards.shape[0])):
    R = rewards[t] + gamma * (1.0 - dones[t]) * R    # (1-done) corta a recursão
    returns[t] = R
return returns

# Parte 2 — compute_policy_loss
weights = returns - values if use_baseline else returns
return -(logprobs * weights.detach()).mean()          # detach: peso é constante p/ o gradiente

# Parte 3 — DiscreteActorCritic.get_action_and_value (networks/discrete_actor_critic.py)
logits = self.actor(x)
probs = Categorical(logits=logits)
if action is None:
    action = torch.argmax(logits, dim=1) if deterministic else probs.sample()
return action, probs.log_prob(action), probs.entropy(), self.critic(x)
```

Pontos que valem mencionar: cada coluna (ambiente) é independente no retorno de n passos; o `detach` é o que os testes verificam (sem ele o gradiente de política vazaria para o crítico); quando uma ação é passada, ela é devolvida sem alteração e apenas a log-probabilidade *dela* é avaliada.

## 5. Protocolo

- Base: `configs/a2c_cartpole.yml` — 500k passos, 8 ambientes, `num_steps=5`, lr 7e-4, γ 0,99, `vf_coef` 0,5, `ent_coef` 0,01, `max_grad_norm` 0,5.
- Cada run muda só o parâmetro da questão via `--override`; **2 seeds em todas as configurações** (22 runs no CartPole).
- "Retorno final" nas tabelas = média dos últimos 20% do treino (mais estável que o último ponto numa curva oscilante).
- **SPS medido à parte**, um run por vez, 200k passos, `track=false`: o SPS registrado durante a varredura mede disputa de CPU entre runs simultâneos, não paralelismo.

---

## 6. Q1 — `num_envs` ∈ {1, **8**, 32, 64} + run de controle

### Previsão (rascunho — reescrever com suas palavras)
Mais atores devem decorrelacionar as transições do lote e reduzir a variância do gradiente, deixando a `policy_loss` menos ruidosa e a curva de retorno mais suave. O SPS deve crescer até saturar no número de núcleos. Com `num_envs` muito grande, cabem menos atualizações nos mesmos 500k passos, o que deve atrasar o aprendizado.

### Dados

| num_envs | lote | atualizações em 500k | retorno final (s1/s2) | eval greedy | desvio da policy_loss | pico negativo | SPS | run ids |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 100 000 | 351 / 315 | 462 / 256 | 4,5 / 7,8 | −63 / −104 | 4 056 | 7zz8i4sv, tdw48yei |
| 8 (baseline) | 40 | 12 500 | 478 / 461 | 500 / 482 | 1,4 / 1,3 | −14 / −10 | 22 683 | 5vukxjqp, q77wk364 |
| 32 | 160 | 3 125 | 417 / 469 | 500 / 375 | 0,74 / 0,84 | −3,6 / −5,0 | 47 593 | u1gors56, o1mbpzz8 |
| 64 | 320 | 1 562 | 221 / 159 | 267 / 167 | 0,57 / 0,46 | −1,8 / −0,3 | 38 843 | jikj7s7y, enq07wm0 |
| **controle: 1 env, n=40** | 40 | 12 500 | 386 / 405 | 454 / 298 | 11,2 / 9,7 | −54 / −72 | — | abqwmzce, vm06tqa6 |

### Explicação (texto de referência)
O argumento do A3C é sobre **correlação**, não sobre quantidade de dados. Sem replay, as transições consecutivas de um único ator vêm do mesmo trecho de trajetória: o lote é uma amostra enviesada de um instante do episódio. O gráfico de `policy_loss` mostra isso: com `num_envs=1` há picos de −63 e −104 e desvio 4,5–7,8; com 8 atores os picos ficam em ~−10 e o desvio cai para 1,3; com 32 e 64 o traço é uma faixa fina.

O **run de controle separa os dois efeitos**: `num_envs=1` com `num_steps=40` tem o mesmo lote (40) e o mesmo número de atualizações (12 500) do baseline, e mesmo assim é o mais ruidoso de todos (desvio 9,7–11,2), com retorno 386–405 contra 461–478. Logo, o ganho vem da **independência** das amostras, não do tamanho do lote. O efeito oposto aparece com `num_envs=64`: gradiente limpíssimo, mas só **1 562 atualizações** em 500k passos (contra 12 500), e o retorno para em 159–221. O SPS sobe de 4 056 (1 env) para 47 593 (32) e **cai** para 38 843 com 64, porque a máquina tem 10 núcleos. Previsão confirmada nos três pontos; o melhor compromisso ficou em 8–32 atores, não no maior valor.

---

## 7. Q2 — `a2c.num_steps` ∈ {1, **5**, 32, 128}

### Previsão (rascunho)
Com `num_steps=1` o alvo é quase todo bootstrap: muito viés, pouca variância. Com 128 é quase Monte Carlo: pouco viés, muita variância. Espero `value_loss` crescendo com o horizonte e um crítico pior nos extremos, o que arrasta o ator, já que a vantagem usa V.

### Dados

| num_steps | lote | retorno final (s1/s2) | eval greedy | value_loss mediana | expl. var. (fim) | média dos pesos A | desvio dos pesos A | run ids |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 431 / 433 | 500 / 500 | ~0,00 | 0,26 / 0,90 | −0,25 / −0,12 | 0,70 / 0,36 | jj88rgp5, agglp2k7 |
| 5 (baseline) | 40 | 478 / 461 | 500 / 482 | ~0,00 | 0,70 / 0,44 | −0,07 / −0,71 | 0,39 / 2,39 | 5vukxjqp, q77wk364 |
| 32 | 256 | 340 / 485 | 180 / 500 | 51 / 77 | 0,00 / 0,30 | 0,81 / 1,09 | 10,1 / 8,1 | immov8y6, dfms8aic |
| 128 | 1024 | 400 / 414 | 371 / 461 | 505 / 498 | 0,00 / 0,00 | 23,5 / 23,4 | 17,8 / 17,7 | tqwv3nh9, 9g7tdlj1 |

### Explicação (texto de referência)
n controla o compromisso viés–variância. Com **n=1** o alvo `r + γV(s')` depende quase só do crítico: baixa variância, mas todo erro de V entra no alvo — viés máximo. Com **n=128** o alvo é quase a soma empírica de um episódio: quase sem viés, mas com toda a variância da trajetória, o que aparece na `value_loss` indo de ~0 para **~500**. A consequência prática está na vantagem: com n=128 os pesos têm média **23,5** e desvio **17,8**, contra média ~0 e desvio 0,4–2,4 no baseline. Média longe de zero significa V sistematicamente abaixo do retorno, e peso quase sempre positivo empurra para cima toda ação amostrada — o mesmo mecanismo da Q4.

**A armadilha da explained_variance:** com n=1 a seed 2 marca **0,90**, melhor que o baseline. Mas a EV é calculada sobre o lote, que ali tem **8 pontos**, de estados quase idênticos, cujos alvos `r + γV` a própria rede acabou de gerar. Explicar a variância de 8 números correlacionados produzidos por você mesmo é fácil e nada diz sobre o valor verdadeiro (a métrica oscila entre −1 e 1 de uma atualização para outra). No outro extremo, com n=32 e 128 a EV fica presa em **0,00** por motivo oposto: com a política já boa, quase todos os estados do lote têm o mesmo retorno, a variância do alvo vai a zero e a razão perde sentido. EV só informa quando o denominador tem variância real. Previsão confirmada quanto a viés/variância; errei ao prever que n=1 seria o pior em retorno — ele ficou entre os melhores na avaliação greedy (500 nas duas seeds), porque no CartPole a recompensa é densa e o viés do bootstrap é tolerável.

---

## 8. Q3 — `a2c.ent_coef` ∈ {0, **0,01**, 0,1}

### Previsão (rascunho)
O bônus impede que a política vire determinística cedo demais e pare de explorar. Com 0,1 o termo deve dominar a perda e manter a entropia perto de ln 2, travando o retorno. Com 0 a entropia deve cair livremente e o CartPole, fácil, provavelmente ainda funciona.

### Dados

| ent_coef | retorno final (s1/s2) | eval greedy | 1º passo com retorno ≥ 450 | entropia no fim | run ids |
|---|---|---|---|---|---|
| 0 | 489 / 499 | 500 / 500 | 129k / 131k | 0,47 / 0,31 | 4223izo4, dsyhmjsz |
| 0,01 (baseline) | 478 / 461 | 500 / 482 | 228k / 274k | 0,55 / 0,55 | 5vukxjqp, q77wk364 |
| 0,1 | 368 / 340 | 357 / 425 | nunca / nunca | 0,60 / 0,60 | 34521kv7, ue9iv084 |

### Explicação (texto de referência)
O bônus impede o **colapso prematuro**: sem ele, algumas atualizações com vantagem positiva podem levar π(a|s) para perto de 1 antes de a alternativa ser testada, e a política deixa de gerar a variabilidade de que o próprio gradiente precisa. Com `ent_coef=0,1` a entropia fica **colada em 0,60**, quase o valor uniforme ln 2 = 0,69: o termo −0,1·H domina a perda e maximizar entropia passa a valer mais que retorno. A política age quase aleatoriamente em parte das decisões e, como uma única ação errada derruba o bastão, o retorno estaciona em 340–368 e **nunca** chega a 450 — ela não aprende mal, é impedida de executar o que aprendeu.

Com `ent_coef=0` a entropia cai a 0,31–0,47 e o retorno atinge 450 em **129–131k passos**, contra 228–274k do baseline, terminando em 500 nas duas seeds. O CartPole é benevolente por três motivos combinados: só **2 ações** (mesmo colapsando, há 50% de chance de ter colapsado na ação certa, e o erro é corrigível em poucos passos); recompensa **densa** (+1 por passo, sinal imediato sem depender de exploração); e término no erro, o que dá crédito rápido e local. Com muitas ações ou recompensa rara — o próprio LunarLander, ou Montezuma's Revenge — uma política que colapsa cedo deixa de visitar os estados que dariam recompensa e não há gradiente que a tire de lá.

---

## 9. Q4 — Ablação do baseline (`use_baseline=false`)

### Previsão (rascunho)
Subtrair V(s) não muda o gradiente esperado, mas reduz a variância. Sem baseline os pesos devem ter média muito positiva (no CartPole toda recompensa é +1) e desvio bem maior; como todo peso é positivo, toda ação amostrada tem a probabilidade aumentada, o que deve colapsar a entropia e degradar o retorno.

### Dados

| use_baseline | retorno final (s1/s2) | eval greedy | média dos pesos A | desvio dos pesos A | entropia no fim | policy_loss mínima | run ids |
|---|---|---|---|---|---|---|---|
| sim — R − V(s) | 478 / 461 | 500 / 482 | −0,07 / −0,71 | 0,39 / 2,39 | 0,55 / 0,55 | −14 / −10 | 5vukxjqp, q77wk364 |
| não — R | 98 / 97 | 57 / 95 | 39,2 / 39,5 | 16,1 / 16,3 | 0,004 / 0,007 | 0,0 / 0,0 | kiwifws0, 9ogc2rkl |

### Explicação (texto de referência)
Os gráficos de `advantage_*` confirmam o resultado teórico literalmente: com baseline a **média** dos pesos fica em zero (−0,07 e −0,71) e o **desvio** em 0,4–2,4; sem baseline a média salta para **39** e o desvio para **16** (7 a 40× maior). Como E[∇logπ] = 0, subtrair uma função só do estado não altera o gradiente esperado, mas altera a variância do estimador — e é a variância que decide quantas amostras são necessárias para a direção média emergir do ruído.

O efeito na entropia é o que destrói a política. No CartPole toda recompensa é +1, então R é **sempre positivo**: o peso nunca muda de sinal, e a `policy_loss` sem baseline tem mínimo exatamente **0,0** (com baseline vai a −10/−14). Peso sempre positivo significa que **toda** ação amostrada tem a probabilidade aumentada; a única forma de uma ação perder probabilidade é ser sorteada menos. É realimentação positiva: a ação um pouco mais provável é sorteada mais, sobe mais, e a distribuição desaba sobre ela. A entropia cai a **0,004** contra 0,55 com baseline, e o retorno, que chegou a ~480 na seed 1, desmorona para ~60–95. A curva mostra os dois estágios: sobe enquanto a política é estocástica, trava quando a entropia zera.

---

## 10. Extra — LunarLander-v3

Config em `configs/a2c_lunarlander.yml`; variações por `--override` (script `scripts/run_lunar_screen.sh`).

| configuração | passos | pico | retorno no fim | eval greedy | entropia no fim |
|---|---|---|---|---|---|
| lr 7e-4, n=16, 16 envs, ent 0,01 (hiperparâmetros do CartPole) | 5M | 4 | −126 | −112 | 0,72 |
| lr 3e-4, n=32, 16 envs, ent 0,01 | 5M | 39 | −92 | −36 | 0,87 |
| lr 1e-4, n=32, 16 envs, ent 0,001 | 8M | 12 | −23 | −9 | 0,93 |
| lr 1e-4, n=32, 16 envs, ent 0 | 8M | 14 | −42 | −54 | 0,90 |
| lr 1e-4, n=32, 32 envs, ent 0,001 | 8M | 4 | −43 | −45 | 0,93 |
| lr 3e-4, n=16, 32 envs, ent 0,01, γ=0,999 | 5M | −6 | −135 | −131 | 0,77 |
| lr 1e-4, n=64, 16 envs, ent 0,001 | 8M | 151 | 115 | 19 | 0,82 |
| **lr 1e-4, n=64, 16 envs, ent 0,001 (mais longa)** | 20M | **174** | **121** | **157 ± 106** | 0,66 |

### Texto de referência
A primeira tentativa com os hiperparâmetros do CartPole **falha de um jeito informativo**: o retorno sobe até ~+4 aos 300k passos e depois **desaba para −125**, ficando lá pelos 5M restantes. Cada ajuste atacou uma causa: **lr 7e-4 → 1e-4** (aqui os alvos valem centenas, não ~100, então o mesmo passo de gradiente é efetivamente 10× maior e derruba a política já aprendida); **n 5 → 32 → 64** (o sinal que separa pousar de cair está quase todo no termo terminal de ±100, e com n curto essa informação precisa atravessar dezenas de bootstraps de um crítico ruim); **ent_coef 0,01 → 0,001** (com 4 ações e penalidade de combustível, entropia alta custa caro no pouso fino).

**O que limita:** a melhor configuração chega a **157 ± 106 na avaliação greedy** — perto de resolver, mas não resolve (≥ 200). Nos 10 episódios da avaliação, **6 pousam com 201 a 240**, dois batem (−6 e −38) e um fica a meio caminho (37): a política sabe pousar, falta consistência. Três limites: **(1) variância do alvo** — este A2C não normaliza a vantagem nem usa GAE, e os retornos variam de −400 a +250; só a combinação de lr baixa com n longo estabiliza, ou seja, compenso à mão uma variância que um estimador melhor trataria; **(2) orçamento de atualizações** — cada lote gera um único passo de gradiente e é descartado; a mesma configuração passou de **eval 19 com 8M passos para 157 com 20M**, sem mudar mais nada, com a curva ainda subindo no fim: o limite é quantidade de atualizações, não instabilidade; **(3) exploração residual** — a entropia cai de 0,82 (8M) para 0,66 (20M) de um máximo de 1,39, e é essa queda que converte a política de treino em política greedy útil: com 8M a determinística pairava em vez de pousar (treino 115 contra eval 19), com 20M as duas concordam. Reduzir a entropia mais cedo foi pior (`ent_coef=0` terminou em −42) — o mesmo dilema da Q3, agora num ambiente que não perdoa.

---

## 11. Observações para quem for escrever o relatório

- **Previsões:** o protocolo exige previsão *antes* de rodar. Os textos de "Previsão" acima são rascunhos teóricos escritos depois de os resultados existirem. Reescreva com suas palavras e marque com honestidade o que se confirmou.
- **Limite de 3 páginas:** a versão em `report_a2c/relatorio_a2c.pdf` cabe em 3 páginas A4 com fonte 8,2 pt, 5 figuras e 5 tabelas compactas.
- **Números honestos:** o SPS da Q1 vem de medição isolada, não da varredura (os runs paralelos disputavam CPU). O LunarLander **não foi resolvido** (eval 157 contra os 200 exigidos, com 6 dos 10 episódios acima de 200); o relatório diz isso e explica o que limita, como o enunciado permite.
