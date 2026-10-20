# Material para o relatório — Atividade SAC (Hopper-v5)

Dump completo de dados, números e decisões, para refazer ou revisar o relatório em outra sessão.

## 1. Requisitos do enunciado (resumo)

- **Entregável:** PDF de no máximo **4 páginas** com as Partes 3, 4 e 5; link do fork (com `algorithms/sac.py` e os configs usados) e link do **projeto wandb público ou compartilhado com os professores**. Todo gráfico deve identificar runs que existem no projeto, e a tabela da Parte 3 tem de bater com os configs registrados neles.
- **Partes 1–2:** `squashed_gaussian_sample` e `compute_soft_q_target`; `pytest tests/test_sac.py`; sanidade no Pendulum com eval > −200.
- **Parte 3:** 100 mil passos exatos, 3 seeds por config, 3–4 candidatos além do inicial, tabela (chave antes → depois, motivo observado citando run, eval e r_last100 como média (mín–máx), veredito) e figura inicial × final.
- **Previsões** antes de cada experimento das Partes 4 e 5.
- **Parte 4:** a partir do config final, 2 seeds: (1) `autotune=false` com α fixo escolhido pela curva de `losses/alpha`; (2) `clipped_double_q=false`. Figura + 3–5 frases + linha de `sac.py` que explica.
- **Parte 5:** SAC × PPO contínuo sem ajuste, 2 seeds, mesmo orçamento: eficiência amostral, tempo de relógio (com `SPS`, `num_updates` e o efeito do `updates_per_step`), mecanismo e estimativa de reuso por transição.

## 2. Links e identificação

- **Autores:** Cleiver Batista da Silva Junior & Bianca Visco (dupla) — **confirmar** se a dupla é a mesma do PPO.
- **Fork:** https://github.com/CleiverJr/t-zero-learning
- **wandb:** https://wandb.ai/jr-cleiver-federal-university-of-goi-s/sac-assignment
- **Config final:** `configs/sac_hopper_final.yml`

## 3. Arquivos

| O quê | Caminho |
|---|---|
| Relatório (3 páginas) | `report_sac/relatorio_sac.pdf` (fonte `relatorio_sac.html`, mesmo CSS do relatório de PPO) |
| Previsões escritas antes de rodar | `report_sac/previsoes.md` (19/10/2026 19:55) |
| Figuras | `report_sac/figs/p3_inicial_vs_final.png`, `p4_alpha_fixo.png`, `p4_um_critico.png`, `p5_sac_vs_ppo.png` |
| Tabela de todos os runs | `report_sac/runs_summary.csv` |
| Scripts | `scripts/run_sac.sh` (roda um config com N seeds), `scripts/sac_diag.py` (média e mín–máx por grupo), `scripts/plot_report_sac.py` (figuras) |

## 4. Implementação

```python
# Parte 1 — squashed_gaussian_sample (sac.py ~145)
normal = Normal(mean, log_std.exp())
u = normal.rsample()                       # reparametrização: u = mean + std*eps
y = torch.tanh(u)
action = y * action_scale + action_bias
log_prob = normal.log_prob(u) - torch.log(action_scale * (1 - y.pow(2)) + 1e-6)
log_prob = log_prob.sum(1, keepdim=True)   # (B, 1)
mean_action = torch.tanh(mean) * action_scale + action_bias

# Parte 2 — compute_soft_q_target (sac.py ~180)
soft_next_value = torch.min(next_q1, next_q2) - alpha * next_log_pi
return rewards + gamma * (1.0 - dones) * soft_next_value
```

- 16/16 testes de `tests/test_sac.py`; os 57 testes de DQN/A2C/PPO/SAC continuam passando.
- Sanidade Pendulum: **eval −142,1** (amostrado −132,9).
- **Bug do harness corrigido:** `train.py` imprimia `args.algo.learning_rate`, que o `SACConfig` não tem (`policy_lr`/`q_lr`), e todo `train.py --config sac_*` quebrava. Troquei só essa linha de log por uma que imprime os campos de lr existentes. Vale avisar o professor.

## 5. Parte 3 — dados (3 seeds, 100k passos)

| config | eval média (mín–máx) | r_last100 média (mín–máx) | α fim | Q − retorno | passos de gradiente | tempo (3 em paralelo) | runs |
|---|---|---|---|---|---|---|---|
| inicial | 2 157 (984–2 935) | 951 (542–1 195) | 0,083 | +64 | 94 900 | 5,9 min | 5qgrxbh1, y2sbgc0j, 7wucorzt |
| learning_starts 5000→1000 | 1 208 (978–1 665) | 1 128 (966–1 427) | 0,075 | −10 | 98 733 | 6,4 min | ydyu7r6a, mk8d2dtr, 28lmc8iz |
| updates_per_step 1→2 | 910 (15–2 411) | 947 (437–1 353) | 0,081 | −1 | 189 733 | 13,6 min | zia30sj6, hjqaxfip, 2a5f7q7q |
| **updates_per_step 1→4** | 2 158 (1 521–2 962) | **1 288 (1 109–1 435)** | 0,086 | +3 | 379 600 | 31,6 min | 46kduzai, 87zhex0k, mdlsrrql |

- **Veredito:** só `updates_per_step=4` passa do ruído, e só no r_last100 (intervalos se sobrepõem apenas entre 1 109 e 1 195); no eval é indistinguível, mas com piso maior (1 521 contra 984). `updates_per_step=2` teve um seed colapsado (eval 15). `learning_starts=1000` é indistinguível, e o r_last100 maior é em parte artefato (menos episódios aleatórios na janela).
- **Colapso no inicial:** seed 1 (`5qgrxbh1`) cai de 764 para 6 entre 69k e 73k passos, com `q_minus_return` subindo para +260–297 no mesmo momento.

## 6. Parte 4 — dados (2 seeds, a partir de updates_per_step=4)

| ablação | eval média (mín–máx) | r_last100 | α fim | Q − retorno | Q médio | runs |
|---|---|---|---|---|---|---|
| final (referência, 3 seeds) | 2 158 (1 521–2 962) | 1 288 (1 109–1 435) | 0,086 | +3 | 230 | 46kduzai, 87zhex0k, mdlsrrql |
| α fixo = 0,08 | 3 079 (2 985–3 173) | 1 254 (1 210–1 299) | 0,080 | −11 | 205 | qw5y3gju, 4l0tnj8k |
| um crítico só | 483 (214–752) | 339 (331–347) | **3,86** | **+6 846** | **6 164** | p2ejnwyv, s867tb2u |

- **α fixo:** entropia cai a −6,5 por volta de 12k e só volta ao alvo −3 perto de 80k; com autotune fica colada em −3. Retorno de treino indistinguível; eval maior, mas o eval amostrado do seed 1 é só 1 175. Linhas: `sac.py:344` (α fixo), `492–500` (bloco do autotune pulado), `209` (`compute_alpha_loss`), `341` (target_entropy = −3), `198` (perda do ator).
- **Um crítico:** Q do buffer chega a ~17 500 aos 40k (seed 1); `q_minus_return` passa de +20 000; α sobe a 3,86 (45× o do final). Linhas: `sac.py:370` (`pick_q2` devolve Q₁), `457` (alvo), `486` + `198` (perda do ator), `209` (α sobe enquanto a entropia está abaixo do alvo).

## 7. Parte 5 — dados

| | eval (s1/s2) | r_last100 (s1/s2) | até 300 | até 500 | até 1 000 | passos de gradiente | reuso/transição | tempo 100k |
|---|---|---|---|---|---|---|---|---|
| SAC (upd=4) | 2 962 / 1 521 | 1 321 / 1 435 | 37k / 27k | 54k / 37k | 79k / 58k | 379 600 | ≈ 970× | **17,7 min** sozinho |
| SAC (inicial, upd=1) | 2 552 / 984 | 542 / 1 116 | 24k / 33k | 52k / 55k | nunca / 91k | 94 900 | ≈ 243× | ~5 min |
| PPO contínuo | 340 / 364 | 326 / 356 | 90k / 70k | nunca | nunca | 7 680 | 10× | **19 s** |

- PPO: runs `26vrmh8i`, `m17rnsvl` (grupo `p5_ppo`), comando do enunciado mais `capture_video=false`.
- Run solo do SAC para o tempo de relógio: `c33cox8u` (grupo `p5_bench_sac`). Aos 19 s ele está em ~6 700 passos; cruza retorno 300 aos **5,4 min** e 1 000 aos **13,7 min**.
- **Por que medir sozinho:** com 3 runs em paralelo, o SAC com upd=4 ficou 2× mais lento (30k passos em ~490 s contra 247 s sozinho). Para o SAC padrão e o PPO a diferença foi pequena (84 contra 70 s; 18,7 contra 17 s).
- **Reuso:** SAC = passos de gradiente × 256 ÷ 100 mil transições; PPO = 10 épocas (lote 4 096 = 4 envs × 1 024, minibatch 128, 7 680 passos).

## 8. Pendências (suas)

1. **Commit e push** — eu não commitei nada. O merge do upstream (atividade SAC) está **preparado mas não concluído** (`git merge --no-commit`); o seu próximo commit fecha esse merge junto com o trabalho do SAC.
2. **Deixar o projeto wandb `sac-assignment` público** ou compartilhado com os professores (o enunciado exige).
3. **Confirmar a dupla** (o cabeçalho diz Cleiver & Bianca Visco, como no PPO).
4. Revisar o texto; as previsões em `previsoes.md` foram escritas antes de rodar as Partes 4 e 5.
