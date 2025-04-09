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
    title="Legenda"
)

plt.title('Estudantes por Curso (2023) e Percentual de Evasão')
plt.savefig('img/alunos.png', dpi=300, bbox_inches='tight')