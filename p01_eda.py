import time
import os
os.system('cls' if os.name == 'nt' else 'clear')
start_time = time.time()

#######################################################################################################
# imports
#######################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#######################################################################################################
# EDA
#######################################################################################################

base_2023 = pd.read_csv('base_2023.csv',sep=';', decimal = ',')
base_2024 = pd.read_csv('base_2024.csv',sep=';', decimal = ',')

base_2023['ano'] = 2023
base_2024['ano'] = 2024

dados = pd.concat([base_2023, base_2024], ignore_index=True)

print('\n'+' Conjuntos de dados '.center(80,'#')+'\n')
print('Base 2023: Matriculados e ingressantes de 2023')
print(base_2023.head())

print('\nBase 2024: Matriculados e ingressantes de 2024')
print(base_2024.head())

print('\n'+' Exploratory Data Analysis - (EDA)'.center(80,'#')+'\n')

#------------------------------------------------------------------------------------------------------

dados_2023 = base_2023.copy()
grupo = dados_2023.groupby(['curso', 'situacao_estudante']).size().unstack(fill_value=0)

for col in ['Ingressante', 'Matriculado']:
    if col not in grupo.columns:
        grupo[col] = 0

evasao = base_2023[base_2023['evadiu'] == 1]
evasao_por_curso = evasao['curso'].value_counts().sort_index()

total_por_curso = grupo.sum(axis=1)
percentual_evasao = [
    (evasao_por_curso.get(curso, 0) / total) * 100 if total > 0 else 0
    for curso, total in zip(grupo.index, total_por_curso)
]

grupo = grupo.sort_index()
cursos = grupo.index.tolist()
matriculados = grupo['Matriculado'].values
ingressantes = grupo['Ingressante'].values
x = np.arange(len(cursos))

#fig, ax1 = plt.subplots(figsize=(16, 6))  
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.bar(x, matriculados, label='Matriculados 2023', color='#004488')
ax1.bar(x, ingressantes, bottom=matriculados, label='Ingressantes 2023', color='#AACCFF')
ax1.set_ylabel('Número de Estudantes')
ax1.set_xticks(x)
ax1.set_xticklabels(cursos, rotation=45, ha='right')
ax1.set_ylim(0, max((matriculados + ingressantes)) * 1.3)  
ax2 = ax1.twinx()
l1, = ax2.plot(x, percentual_evasao, 'o-', color='red', label='% Evasão 2023', markersize=6)
ax2.set_ylabel('Evasão (%)')
ax2.set_ylim(10, max(percentual_evasao) * 1.2 if percentual_evasao else 20)

plt.subplots_adjust(right=0.75)

handles_1, labels_1 = ax1.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
fig.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    loc='center left',
    bbox_to_anchor=(0.6, 0.8),
    title="Legenda",
    fontsize=11,  
    title_fontsize=12
)

plt.title('Estudantes por Curso (2023) e Percentual de Evasão')
plt.savefig('img/alunos.png', dpi=300, bbox_inches='tight')

#plt.show()

#------------------------------------------------------------------------------------------------------

print('\n' + ' Tabela Cruzada: Renda Familiar x Evasão '.center(80, '#') + '\n')

if 'renda_familiar' in base_2023.columns and 'evadiu' in base_2023.columns:
    
    if pd.api.types.is_numeric_dtype(base_2023['renda_familiar']):
        base_2023['faixa_renda'] = pd.cut(
            base_2023['renda_familiar'],
            bins=[0, 1500, 3000, 6000, np.inf],
            labels=['Até R$1500', 'R$1501–3000', 'R$3001–6000', 'Acima de R$6000']
        )
    else:
        base_2023['faixa_renda'] = base_2023['renda_familiar']
    
    crosstab = pd.crosstab(
        base_2023['faixa_renda'], 
        base_2023['evadiu'], 
        normalize='index'
    ) * 100

    crosstab = crosstab.rename(columns={0: '% Não Evadiu', 1: '% Evadiu'})
    print(crosstab.round(2))


#######################################################################################################
# end!
#######################################################################################################
print(' Fim do processamento! '.center(80,'#')+'\n')
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_minutes = int(execution_time_seconds // 60)
execution_seconds = int(execution_time_seconds % 60)
execution_time_str = f"{execution_minutes} minutos, {execution_seconds} segundos"
print("Tempo de execução:", execution_time_str)
