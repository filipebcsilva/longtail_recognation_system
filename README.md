# Busca em bancos de imagens com agentes de IA

Sistema multiagentes que responde perguntas em linguagem natural sobre grandes bancos de imagens, como *"caminhões à noite"* ou *"carro ultrapassando com chuva"*, e devolve as imagens que atendem à busca. Foi desenvolvido na minha iniciação científica na UFES e originou um artigo aceito no **IEEE Sensors Applications Symposium (SAS) 2026**.

O foco são eventos raros (*long-tail*) em dados de direção autônoma, que são difíceis de encontrar com filtros tradicionais.

## Como funciona

```
Pergunta do usuário
   │
   ▼
1. Gerador de perguntas ── quebra a busca em perguntas simples e reutilizáveis
   │                       ("quantos caminhões há na imagem?", "qual o período do dia?")
   ▼
2. Filtro ──────────────── descarta perguntas que o cache (data.csv) já responde
   │
   ▼
3. Leitor de imagens ───── modelo de visão responde as perguntas novas, imagem por imagem,
   │                       e as respostas viram colunas do data.csv
   ▼
4. Analista de dados ───── gera código Pandas para filtrar a tabela e
                           devolve a lista de imagens encontradas
```

Como as respostas ficam salvas, uma busca nova só processa as imagens de novo quando precisa de uma informação que ainda não existe na tabela. Buscas parecidas ficam praticamente instantâneas.

## Modelos (rodando localmente com Ollama)

| Papel | Modelo |
| --- | --- |
| Gerador e filtro de perguntas | Llama 3.1 70B Instruct |
| Leitura das imagens | LLaVA 1.6 34B |
| Geração de código Pandas | Qwen2.5-Coder 32B |

## Avaliação

Testado em cerca de 7 mil imagens do dataset [BDD100K](https://www.bdd100k.com/), comparando as respostas do sistema com as anotações oficiais. Alguns resultados de acurácia por pergunta:

| Pergunta | Acurácia |
| --- | --- |
| Quantas pessoas há na imagem? | 86,8% |
| Quantos ônibus há na imagem? | 90,5% |
| Há semáforo amarelo? | 89,8% |
| Há semáforo vermelho? | 80,6% |
| Quantos carros há na imagem? | 65,2% |

A análise completa, incluindo consultas com várias condições e análise qualitativa, está em `analise-resultados/`.

## Tecnologias

Python · [Agno](https://github.com/agno-agi/agno) (agentes e workflows) · Ollama · Pandas · Pydantic (saídas estruturadas) · Streamlit (interface de demonstração)

## Estrutura

| Arquivo | O que faz |
| --- | --- |
| `agents.py` | Definição dos agentes, prompts e esquemas de saída |
| `workflow.py` | Etapas do workflow: ler imagem, extrair respostas, salvar no CSV |
| `longtail_system.py` | Orquestra o fluxo completo de uma busca |
| `app.py` | Interface em Streamlit para buscar e ver as imagens |
| `analise-resultados/` | Notebook com as métricas contra o ground truth do BDD100K |

## Como rodar

```bash
pip install agno ollama pandas pydantic streamlit
ollama pull llama3.1:70b-instruct-q6_K
ollama pull llava:34b-v1.6-q8_0
ollama pull qwen2.5-coder:32b-instruct-q8_0
streamlit run app.py
```

Os modelos exigem uma GPU com bastante memória. Para testar em máquinas menores, troque os modelos em `agents.py` por versões menores.
