# LH_CD_MATHEUSVENTURA

## Como rodar o projeto
1. Clone o repositório na sua máquina
2. Crie um ambiente virtual em python
3. Instale as bibliotecas pelo arquivo `requirements.txt`
```
python -m pip install -r requirements.txt
```
4. Com o arquivo na pasta `data/raw` com o nome `cars_test.csv`
5. Execute o arquivo `run.py`
   * Após isso, é esperado que na pasta `data/processed` o arquivo `predicted.csv` contendo id e preço dos carros, seja criado

## Estrutura
```.
├── run.py
├── README.md
├── requirements.txt
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── models
├── notebooks
├── references
├── reports
│   └── figures
```
