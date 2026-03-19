# Sources

All papers, frameworks, and prior work referenced in this project.

> **SESSION H NOTE:** Several source descriptions below reference findings that are now dead ("our finding that system prompts operate 100% through MLPs," etc.). The sources themselves are valid. The descriptions of how they relate to this project's findings need updating — the project's findings changed fundamentally in Sessions G and H.
Organized by contribution to the findings.

## MLP Mechanism (core to Session E findings)

- **Geva et al. (2021)** — "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP.
  Established that MLP layers store factual associations as key-value pairs.
  Our finding that system prompts operate 100% through MLPs builds directly on this.
  [arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)

- **Meng et al. (2022)** — "Locating and Editing Factual Associations in GPT." NeurIPS.
  Causal tracing showed factual knowledge is in mid-layer MLPs (~1.6% attention).
  Our causal patching methodology follows ROME's approach.
  [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)

- **"Attention Retrieves, MLP Memorizes" (2025)**
  Definitive evidence for the attention/MLP division of labor.
  MLPs store, attention routes. Random static attention achieves near-perfect on many tasks.
  [arxiv.org/abs/2506.01115](https://arxiv.org/abs/2506.01115)

- **"Do All Autoregressive Transformers Remember Facts the Same Way?" (2025)**
  Found Qwen routes factual recall through attention more than other architectures.
  Makes our MLP-only finding in Qwen particularly noteworthy — system prompts diverge from facts.
  [arxiv.org/abs/2509.08778](https://arxiv.org/abs/2509.08778)

- **"Dissecting Persona-Driven Reasoning via Activation Patching" (2025)**
  Closest prior work. Found early MLPs encode persona in Qwen 1.5B and Llama 3B.
  But attention still contributed. Our 100%/0% split is more extreme.
  [arxiv.org/abs/2507.20936](https://arxiv.org/abs/2507.20936)

## System Prompt Degradation

- **SysBench (2025)** — ICLR. System prompt behavioral degradation benchmark.
  GPT-4o drops from 84.8% to 33.7% over 5 turns. Qwen2-7B drops to 1.1%.
  We added the mechanistic measurement (half-life, inversion, attention dilution).
  [arxiv.org/abs/2408.10943](https://arxiv.org/abs/2408.10943)

- **"LLMs Get Lost in Multi-Turn Conversation" (2025)**
  39% average performance decline multi-turn. "When LLMs take a wrong turn, they don't recover."
  [arxiv.org/abs/2505.06120](https://arxiv.org/abs/2505.06120)

- **"Lost in the Middle" (Liu et al., TACL 2024)**
  U-shaped performance curve for information placement. RoPE decay.
  [aclanthology.org/2024.tacl-1.9/](https://aclanthology.org/2024.tacl-1.9/)

- **"Lost in the Middle at Birth" (2026)**
  Position bias is architectural, present at initialization before training.
  Middle-context retrieval is structurally hostile.
  [arxiv.org/abs/2603.10123](https://arxiv.org/abs/2603.10123)

- **"Get My Drift?" (Abdelnabi et al., 2024)**
  Uses activation deltas to detect task drift. Binary detection, not temporal dynamics.
  [arxiv.org/abs/2406.00799](https://arxiv.org/abs/2406.00799)

- **"Context Rot" (Chroma Research, 2025)**
  All 18 tested frontier models degrade with context growth.
  [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot)

## Attention and Compression Mechanics

- **"Attention Sinks and Compression Valleys" (2025)**
  Mathematical proof: massive activations cause attention sinks AND compression valleys.
  Three-phase model: mix-compress-refine. Middle layers sacrifice granularity.
  [arxiv.org/abs/2510.06477](https://arxiv.org/abs/2510.06477)

- **"When Attention Sink Emerges" (ICLR 2025)**
  How early tokens receive disproportionate attention.
  [arxiv.org/abs/2410.10781](https://arxiv.org/abs/2410.10781)

- **Elhage et al. (2022)** — "Toy Models of Superposition." Anthropic.
  Features stored in superposition, phase changes in representation.
  [transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)

## Alignment and Safety

- **"What Is the Alignment Tax?" (2026)**
  First geometric formalization. Safety and helpfulness neurons overlap
  but require different activation patterns. The tax is measurable.
  [arxiv.org/abs/2603.00047](https://arxiv.org/abs/2603.00047)

- **"Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (ICLR 2025)**
  Safety alignment is shallow, primarily affecting first output tokens.
  [arxiv.org/abs/2406.05946](https://arxiv.org/abs/2406.05946)

- **"Refusal Is Mediated by a Single Direction" (NeurIPS 2024)**
  Across 13 models, refusal is one direction in activation space.
  [arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)

- **"Finding Safety Neurons in Large Language Models" (2024)**
  ~5% of neurons are safety-critical. MLP layers dominate safety mechanisms.
  [arxiv.org/abs/2406.14144](https://arxiv.org/abs/2406.14144)

- **"Fundamental Limitations of Alignment" (ICML 2024)**
  For any behavior with finite probability, adversarial prompts can trigger it.
  [arxiv.org/abs/2304.11082](https://arxiv.org/abs/2304.11082)

## Token-Level Effects and Prompt Sensitivity

- **Wallace et al. (2019)** — "Universal Adversarial Triggers." EMNLP.
  Nonsensical token sequences drastically change behavior across inputs.
  [aclanthology.org/D19-1221/](https://aclanthology.org/D19-1221/)

- **Zou et al. (2023)** — "Universal Adversarial Attacks on Aligned LMs."
  GCG attack: nonsensical suffixes reliably jailbreak safety-aligned models.
  [arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043)

- **Webson & Pavlick (2022)** — "Do Prompt-Based Models Really Understand Their Prompts?" NAACL.
  Models learn as fast with misleading prompts as with good ones.
  [aclanthology.org/2022.naacl-main.167/](https://aclanthology.org/2022.naacl-main.167/)

- **Sinha et al. (2021)** — "UnNatural Language Inference." ACL.
  75-90% accuracy preserved with randomly shuffled word order.
  [aclanthology.org/2021.acl-long.569/](https://aclanthology.org/2021.acl-long.569/)

- **Min et al. (2022)** — "Rethinking the Role of Demonstrations." EMNLP.
  Ground truth labels barely matter. Format and distribution matter.
  [aclanthology.org/2022.emnlp-main.759/](https://aclanthology.org/2022.emnlp-main.759/)

- **"One Trigger Token Is Enough" (2025)**
  Safety-aligned LLMs have specific safety trigger tokens.
  [arxiv.org/abs/2505.07167](https://arxiv.org/abs/2505.07167)

## Scale-Dependent Processing

- **Wei et al. (2023)** — "Larger Language Models Do In-Context Learning Differently."
  Small models rely on priors. Large models override with context.
  [arxiv.org/abs/2303.03846](https://arxiv.org/abs/2303.03846)

- **"Do Language Models Use Their Depth Efficiently?" (2025)**
  Larger models spread computation across more layers.
  [arxiv.org/abs/2505.13898](https://arxiv.org/abs/2505.13898)

- **"What Affects the Effective Depth of Large Language Models?" (2025)**
  Analyzes Qwen 1.5B-32B. Effective depth ratio stable at ~0.6-0.7.
  [arxiv.org/abs/2512.14064](https://arxiv.org/abs/2512.14064)

## Relational vs Featural Processing

- **Altabaa & Lafferty (ICML 2025)** — "Disentangling Relational and Sensory Information."
  Standard transformers entangle relational and sensory information.
  Relations only implicitly influence attention, not values.
  [arxiv.org/abs/2405.16727](https://arxiv.org/abs/2405.16727)

- **"Tabula RASA" (2025)** — Formal relational bottleneck in transformers.
  Standard transformers need O(k) layers for k-hop relational reasoning.
  [arxiv.org/abs/2602.02834](https://arxiv.org/abs/2602.02834)

## Information Theory and Entropy

- **Voita, Sennrich & Titov (2019)** — "Evolution of Representations in the Transformer." EMNLP.
  Mutual information with past context vanishes in deeper layers.
  [aclanthology.org/D19-1448/](https://aclanthology.org/D19-1448/)

- **"Entropy-Lens" (2025)** — Entropy profiling across transformer layers.
  [arxiv.org/abs/2502.16570](https://arxiv.org/abs/2502.16570)

- **"Sense and Sensitivity" (2025)** — Lexical recall degrades 2.39%, semantic recall 53-93%.
  Closest parallel to our vocabulary/semantics temporal split.
  [arxiv.org/abs/2505.13353](https://arxiv.org/abs/2505.13353)

## Cognitive Science Parallels

- **Halford, Wilson & Phillips (1998)** — "Processing Capacity Defined by Relational Complexity." BBS.
  Working memory limits are about relations, not items.

- **"Working Memory Capacity Limits Memory for Bindings" (2019)** — Journal of Cognition.
  The capacity limit is on bindings between features, not features themselves.

- **Sims (2023)** — "Rate-distortion theory of neural coding." eLife.
  Memory as rate-distortion problem. Fixed capacity, relational info compressed first.

## Computational Irreducibility

- **Wolfram (2002)** — "A New Kind of Science."
  Simple rules produce complex behavior. You can't predict the evolution
  without running the computation. The automaton frame for transformers.

## Human-AI Interaction (from original paper)

- Bajcsy & Fisac (2024) — Human-AI coupled system as unit of analysis
- Weidinger et al. (2024) — Interaction harms require interaction-level evaluation
- Kim et al. (2026) — Cognitive surrender
- Chu et al. (2025) — Emotional dynamics in human-AI relationships
- Smart, Clowes & Clark (2025) — Extended mind applied to LLMs
- Toner (2025) — Personalization as profiling
- Nature Mental Health (2025) — Technological folie à deux

## Mechanistic Interpretability Tools

- **TransformerLens** — Nanda et al. (2022). The primary tool used in Session E.
  [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

- **Anthropic Circuit Tracing (2025)** — Attribution graphs via cross-layer transcoders.
  [transformer-circuits.pub/2025/attribution-graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

- **Bricken et al. (2023)** — "Towards Monosemanticity." Sparse autoencoders.
  [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

## Self-Correction Literature (from original paper)

- Huang et al. (ICLR 2024) — LLMs cannot self-correct reasoning
- Kamoi et al. (TACL 2024) — When LLMs can correct mistakes
- Tyen et al. (ACL 2024) — LLMs cannot find reasoning errors
- "Dark Side of Intrinsic Self-Correction" (ACL 2025)
- Stechly, Valmeekam & Kambhampati (2024) — Self-verification limitations

## Methodology

- Lo (CHI 2024) — Autoethnography with AI
- Dezfouli, Nock & Dayan (PNAS 2020) — Adversarial human decision modeling
- Stanford POPPER Framework (2025) — Sequential falsification
- Sarkar (CACM 2024) — AI as provocateur
