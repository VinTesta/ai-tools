# chunk-generator

Divide arquivos Markdown grandes em chunks, salva cada chunk em um arquivo próprio e gera um `index.md` com resumo e link para cada parte.

## Instalação

```bash
cd chunk-generator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Uso

```bash
export DEEPSEEK_API_KEY="sua-chave"
chunk-generator -f livro.md -s 4000
```

Sem resumo por IA:

```bash
chunk-generator -f livro.md -s 4000 --no-ai-index
```

Saída padrão: pasta única em `generated/<nome-do-arquivo>-<timestamp>/`, contendo `index.md` e arquivos `chunks/chunk-0001.md`, `chunks/chunk-0002.md`, etc.
# ai-tools
