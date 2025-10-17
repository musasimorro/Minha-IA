# Arquivo: app.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Inicializa o Flask
app = Flask(__name__)

# 1. Carrega o modelo de IA (treinado no Colab e salvo como .pkl)
# O modelo está pronto para ser usado!
modelo = joblib.load('modelo_satisfacao.pkl')

# Define a página inicial (a interface)
@app.route('/')
def home():
    return render_template('index.html')

# Define a rota que a interface vai chamar para fazer a previsão
@app.route('/prever', methods=['POST'])
def prever():
    # 2. Recebe os dados enviados pelo formulário HTML
    dados = request.form

    # Converte os dados recebidos em um DataFrame do Pandas
    # A ordem e os nomes das colunas DEVEM ser os mesmos usados no treinamento
    df_previsao = pd.DataFrame({
        'Anos_de_Empresa': [float(dados['anos_empresa'])],
        'Horas_Extras_Mes': [float(dados['horas_extras'])],
        'Salario_Ajustado': [float(dados['salario'])]
    })
    
    # 3. Faz a previsão usando a IA
    previsao = modelo.predict(df_previsao)
    
    # Converte o resultado Sim/Nao para uma mensagem mais amigável
    resultado = str(previsao[0])
    
    # Retorna o resultado para a interface
    return render_template('index.html', previsao_ia=resultado)

if __name__ == "__main__":
    app.run(debug=True)