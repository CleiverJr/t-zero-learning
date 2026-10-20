# Previsões das Partes 4 e 5 — escritas ANTES de rodar

Registradas em **2026-10-19 19:55**, quando só existiam os runs do config inicial da Parte 3
(`p3_base`, 3 seeds). Nenhum run das Partes 4 e 5 tinha sido iniciado. O que eu sabia naquele momento:

- `losses/alpha` cai de 1,0 para ~0,08 antes de ~13k passos e termina em **0,083** (mín. 0,053–0,066 entre seeds).
- `losses/entropy` acompanha o alvo: −3,01 contra `charts/target_entropy` = −3.
- `charts/q_minus_return` termina em **+64** com Q ≈ 201: os críticos já superestimam, mesmo com min(Q1, Q2).
- Retorno com enorme variação entre seeds: eval 984–2935.

## Parte 4, ablação 1 — `sac.autotune=false` com α fixo

**Valor escolhido: α = 0,08**, o valor em que o autotune estabiliza no fim dos runs da Parte 3.
A ideia é isolar o que o *ajuste* faz, mantendo a temperatura que ele mesmo escolheu no final.

**Previsão.** O fim do treino deve ficar parecido com o do config da Parte 3, porque o α final é o mesmo.
A diferença deve estar **no começo**: com autotune, α começa em 1,0 e passa os primeiros milhares
de passos alto, o que mantém a política exploratória enquanto os críticos ainda são ruins. Com α fixo
em 0,08 desde o início, espero a entropia caindo **abaixo** do alvo de −3 logo cedo (nada a segura),
um começo de aprendizado um pouco mais rápido ou mais errático, e uma chance maior de alguma seed
ficar presa num comportamento ruim por falta de exploração. Em média, espero algo indistinguível do
ruído no retorno final, com a entropia descolando do alvo como diferença mais clara.

## Parte 4, ablação 2 — `sac.clipped_double_q=false` (um crítico só)

**Previsão.** O `min(Q1, Q2)` existe para combater o viés de superestimação do max/argmax
implícito no alvo. Sem ele, espero `charts/q_minus_return` **bem mais positivo** que os +64 atuais e
`losses/qf1_values` crescendo mais rápido, possivelmente sem estabilizar. O ator é treinado para
maximizar Q, então vai explorar exatamente os erros de superestimação, e espero retorno pior e mais
instável — com alguma seed podendo colapsar. É a mesma dinâmica que vi no DQN com
`target_network_frequency=1`, numa forma mais branda.

## Parte 5 — SAC × PPO contínuo no Hopper, 100k passos

**Eficiência amostral.** Espero o SAC muito à frente: ele reutiliza cada transição em centenas de
passos de gradiente via replay, enquanto o PPO usa cada rollout por poucas épocas e o descarta.
Com 100k passos, imagino o PPO bem abaixo de 1000 de retorno e o SAC acima de 1000.

**Tempo de relógio.** Espero o contrário: o PPO deve ter SPS bem maior, porque faz um passo de
gradiente para cada muitos passos de ambiente, enquanto o SAC faz um (ou mais) por passo. Mas,
como o SAC é tão mais eficiente por amostra, aposto que ele ainda chega mais longe no mesmo
tempo — como o PPO contra o A2C na atividade anterior. Se a Parte 3 escolher `updates_per_step` > 1,
o SAC fica proporcionalmente mais lento, e aí a resposta pode inverter.

**Reuso por transição (estimativa antes de rodar).** SAC: ~95k passos de gradiente × lote 256 ÷ 100k
transições ≈ **243 vezes** cada transição entra num gradiente (vezes `updates_per_step`).
PPO: cada transição entra em `update_epochs` épocas — algo como **~10 vezes**. Confiro os valores
reais do config do PPO quando rodar.
