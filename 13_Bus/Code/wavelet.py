import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mude o nome para os dados vindos do MatLAB
df = pd.read_csv('19_06_20s_13Bus_200_linhas_10mile_seg.csv', sep=';', header=None)

# Passo 1: Escolha um valor de referência
valor_referencia = df.abs().max().max()  # Máximo valor absoluto em todo o DataFrame

# Passo 2: Normalização dos dados em relação ao valor de referência
df_pu = df / valor_referencia  # Convertendo para unidades de percentual (PU)

# Exibindo o DataFrame com o sinal em PU
print(df_pu)

df2 = df_pu.copy()

# Criar um dataframe vazio para armazenar os coeficientes
new_df = pd.DataFrame()

# Passo 1: Iterar sobre as linhas do dataframe
for index, row in df_pu.iterrows():
    # Passo 2: Converter os valores em um array numpy
    signal = row.values
    
    # Aplicar a Transformada Wavelet Contínua (CWT)
    wavelet = 'mexh'  # Selecionar a wavelet Morlet  mexh morl shan fbsp cmor
    scales = np.arange(1, 7)  # Escalas da wavelet
    coef, freqs = pywt.cwt(signal, scales, wavelet)
   
    df_ax = pd.DataFrame([coef[5]])
    
    new_df = pd.concat([new_df, df_ax], axis=0)

# Criando uma lista com os valores desejados
data = [0] * 20 + [1] * 21 + [0] * 39 + [1] * 21 + [0] * 29 + [1] * 21 + [0] * 19 + [1] * 21 + [0] * 9

# Verificando o comprimento da lista (deve ser 200)
print(len(data))  # Deve ser 200

# Criando o DataFrame com a lista de dados
df_co = pd.DataFrame(data)

# Exibindo as primeiras e últimas 20 linhas do DataFrame para verificação
print(df_co.head(20))
print(df_co.tail(20))

df2 = df_co.copy()

new_df = new_df.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

df_salvar = pd.concat([new_df, df2], axis=1)

df_salvar.to_csv('mexh_pu_13bus_19_06_corf_7_200linhas_100mile.csv', sep=';', index=True)
